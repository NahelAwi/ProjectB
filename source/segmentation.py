import depthai as dai
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.models.segmentation import deeplabv3_resnet101

# Load pre-trained DeepLabV3 model
model = deeplabv3_resnet101(pretrained=True)
model.eval()

# Define image transformation
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((520, 520)),  # Resize for model input
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create a color map for different classes
def create_color_map(num_classes):
    color_map = np.zeros((num_classes, 3), dtype=np.uint8)
    for i in range(num_classes):
        color_map[i] = [np.random.randint(0, 255) for _ in range(3)]  # Random colors
    return color_map

# Number of classes in COCO
NUM_CLASSES = 21
color_map = create_color_map(NUM_CLASSES)

# Create pipeline
pipeline = dai.Pipeline()

# Create RGB camera node
rgb_cam = pipeline.createColorCamera()
rgb_cam.setBoardSocket(dai.CameraBoardSocket.RGB)
rgb_cam.setInterleaved(False)
rgb_cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)

# Create outputs
rgb_output = pipeline.createXLinkOut()
rgb_output.setStreamName("video")

# Link nodes
rgb_cam.video.link(rgb_output.input)

# Start pipeline
with dai.Device(pipeline) as device:
    rgb_queue = device.getOutputQueue(name="video", maxSize=1, blocking=False)

    while True:
        rgb_frame = rgb_queue.get().getCvFrame()

        # Transform the image for the segmentation model
        input_tensor = transform(rgb_frame).unsqueeze(0)

        # Run inference
        with torch.no_grad():
            output = model(input_tensor)['out'][0]
        output_predictions = output.argmax(0).byte().cpu().numpy()

        # Create an empty output image for visualization
        output_image = np.zeros((output_predictions.shape[0], output_predictions.shape[1], 3), dtype=np.uint8)

        # Color the output image based on the predictions
        for class_id in range(NUM_CLASSES):
            output_image[output_predictions == class_id] = color_map[class_id]

        # Show the RGB frame with segmentation
        cv2.imshow("RGB Frame with Segmentation", output_image)

        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()
