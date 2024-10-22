import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
from source.utils import *

# Load pre-trained DeepLabV3 model
model = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.DEFAULT)
model.eval()

# Define image transformation
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((300, 300)),  # Resize for model input
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create a color map for different classes
def create_color_map(num_classes):
    color_map = np.zeros((num_classes, 3), dtype=np.uint8)
    for i in range(num_classes):
        color_map[i] = [np.random.randint(0, 255) for _ in range(3)]  # Random colors
    return color_map

# Number of classes in COCO (DeepLabV3 uses 21 classes by default for COCO)
NUM_CLASSES = 21
color_map = create_color_map(NUM_CLASSES)

# # Open the laptop camera
# cap = cv2.VideoCapture(0)  # 0 is the default camera index

# if not cap.isOpened():
#     print("Error: Could not open camera.")
#     exit()

pipeline = dai.Pipeline()

# Create RGB camera node
rgb_cam = pipeline.create(dai.node.ColorCamera)
rgb_cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
# rgb_cam = pipeline.createColorCamera()
# rgb_cam.setBoardSocket(dai.CameraBoardSocket.RGB)
rgb_cam.setInterleaved(False)
rgb_cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
rgb_cam.setPreviewSize(300,300)
rgb_cam.setPreviewKeepAspectRatio(False)
rgb_cam.setFps(30)

# Create outputs
rgb_output = pipeline.create(dai.node.XLinkOut)
rgb_output.setStreamName("rgb")

# Link nodes
rgb_cam.preview.link(rgb_output.input)

with dai.Device(pipeline) as device:
    rgb_queue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

    while True:
        # ret, frame = cap.read()
        # if not ret:
        #     print("Error: Failed to capture frame.")
        #     break

        in_rgb = rgb_queue.get()

        frame = None
        if in_rgb is not None:
            frame = in_rgb.getCvFrame()

        # Convert the frame for the segmentation model
        input_tensor = transform(frame).unsqueeze(0)

        # Run inference
        with torch.no_grad():
            output = model(input_tensor)['out'][0]
        output_predictions = output.argmax(0).byte().cpu().numpy()

        # Create an empty output image for visualization
        output_image = np.zeros((output_predictions.shape[0], output_predictions.shape[1], 3), dtype=np.uint8)
        
        # Find the "main object", for simplicity use the class with the largest number of pixels
        unique_classes, counts = np.unique(output_predictions, return_counts=True)
        if len(counts) == 0:
            continue

        main_class_id = unique_classes[np.argmax(counts)]
        keypoints = np.column_stack(np.where(output_predictions == main_class_id))

        if len(keypoints) > 0:
            # Calculate rotation angle and centroid
            angle, centroid = calculate_pca_rotation_angle(keypoints)

            if angle is not None and centroid is not None:
                print(f"Main Object Class ID: {main_class_id}")
                print(f"rotation Angle: {angle:.2f} degrees")
                print(f"Centroid: {centroid}")

        # Color the output image based on the predictions
        for class_id in range(NUM_CLASSES):
            output_image[output_predictions == class_id] = color_map[class_id]

        # Show the original frame and the segmented output
        cv2.imshow("RGB Frame with Segmentation", output_image)
        cv2.imshow("Original Frame", frame)
        
        if cv2.waitKey(1) == ord('q'):
            break

# Release resources and close windows
cap.release()
cv2.destroyAllWindows()
