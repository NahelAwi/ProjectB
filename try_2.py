import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from source.utils import *

# def calculate_pca_rotation_angle(keypoints):
#     """Calculate rotation angle and centroid using PCA."""
#     if keypoints.shape[0] < 2:  # Need at least 2 points for PCA
#         return None, None

#     keypoints = keypoints.astype(np.float32)
#     m, e = cv2.PCACompute(keypoints, mean=np.array([]))

#     # Calculate rotation angle in degrees
#     angle = np.arctan2(e[0][1], e[0][0]) * 180 / np.pi
#     centroid = tuple(m[0])
    
#     return angle, centroid

# Load pre-trained Mask R-CNN model
model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
model.eval()

# Define image transformation
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert to tensor
])

# Open the laptop camera
cap = cv2.VideoCapture(0)  # 0 is the default camera index

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Convert the frame for the segmentation model
    input_tensor = transform(frame)
    input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension

    # Run inference
    with torch.no_grad():
        predictions = model(input_tensor)

    # Extract masks and labels
    masks = predictions[0]['masks'].cpu().numpy()
    boxes = predictions[0]['boxes'].cpu().numpy()
    scores = predictions[0]['scores'].cpu().numpy()
    labels = predictions[0]['labels'].cpu().numpy()

    # Set a threshold to filter predictions
    threshold = 0.5
    output_predictions = np.zeros(frame.shape[:2], dtype=np.uint8)  # For storing output predictions

    for i in range(len(scores)):
        if scores[i] > threshold:
            mask = masks[i, 0]  # Get the mask for this object
            mask = (mask > 0.5).astype(np.uint8)  # Binarize the mask
            
            # Update output predictions for the main object
            output_predictions[mask == 1] = labels[i].item()

            # Draw bounding box and color the mask
            color_mask = np.zeros_like(frame)
            color_mask[mask == 1] = np.array([0, 255, 0])  # Green color
            frame = cv2.addWeighted(frame, 1, color_mask, 0.5, 0)

            # Draw bounding box
            x1, y1, x2, y2 = boxes[i].astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue bounding box

    # Find the "main object", for simplicity use the class with the largest number of pixels ?
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

    # Show the original frame with the segmentation overlays
    cv2.imshow("Mask R-CNN Segmentation", frame)
    
    if cv2.waitKey(1) == ord('q'):
        break

# Release resources and close windows
cap.release()
cv2.destroyAllWindows()
