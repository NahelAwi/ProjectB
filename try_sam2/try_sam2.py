import cv2
import torch
import numpy as np
import sys
import random
sys.path.append('C:/study/ProjectB/')
sys.path.append('C:/study/ProjectB/sam2')
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from collections import deque
from source.utils import *
from torch.quantization import quantize_dynamic

height = 320
width = 480

def sam2_segment_frame(predictor, frame, prompt):
    # Load your image
    # image_path = "images/sample_4.png"
    image = cv2.resize(frame, (width, height))#cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Run segmentation (CPU only)
    with torch.no_grad():  # No inference mode for CPU
        predictor.set_image(image_rgb)
        masks, scores, logits = predictor.predict(
            point_coords=prompt["points"],
            point_labels=prompt["labels"]
        )

    # Overlay and display the segmentation result
    # for mask in masks:
    # for now pick random mask
    mask = random.choice(masks)
    segmentation = (mask * 255).astype(np.uint8)
    overlay = cv2.addWeighted(image, 0.7, cv2.cvtColor(segmentation, cv2.COLOR_GRAY2BGR), 0.3, 0)
    
    center_x, center_y, angle = calculate_pca_rotation_angle_from_mask(mask)
    if angle is None:
        return

    # Draw the orientation arrow
    arrow_length = 50
    end_x = int(center_x - arrow_length * np.cos(np.deg2rad(angle)))
    end_y = int(center_y - arrow_length * np.sin(np.deg2rad(angle)))

    cv2.arrowedLine(overlay, (int(center_y), int(center_x)), (end_y, end_x), (0, 255, 0), 2, tipLength=0.3)
    
    return overlay
    # cv2.imshow("Main Object Segmentation", overlay)
    # cv2.waitKey("q")

    # cv2.destroyAllWindows()

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Paths for checkpoint and config
checkpoint = "C:/study/ProjectB/sam2/checkpoints/sam2.1_hiera_tiny.pt"
model_cfg = "C:/study/ProjectB/sam2/sam2/configs/sam2.1/sam2.1_hiera_t.yaml"

# Build the predictor
model = build_sam2(model_cfg, checkpoint, device="cpu")
model = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
predictor = SAM2ImagePredictor(model)

# Define the center point
center_x, center_y = width // 2, height // 2

# Define radius and number of points
radius = 25
num_points = 5

# Generate points around the center
angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)  # Divide circle into equal angles
circle_points = np.array([
    [int(center_x + radius * np.cos(angle)), int(center_y + radius * np.sin(angle))]
    for angle in angles
])

# Combine the center point with the circle points
all_points = np.vstack([[center_x, center_y], circle_points])
# # Calculate the center point of the image
# center_point = np.array([[width // 2, height // 2]])  # x, y coordinates at the center

# Define labels for each point (e.g., 1 indicates the object to segment)
labels = np.ones(all_points.shape[0], dtype=np.int32)  # All points are object points

# Define the prompt with points and labels
prompt = {
    "points": all_points,
    "labels": labels
}

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = sam2_segment_frame(predictor, frame, prompt)
        cv2.imshow("Real-Time Segmentation", processed_frame)
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
