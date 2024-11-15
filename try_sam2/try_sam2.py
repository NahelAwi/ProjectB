import cv2
import torch
import numpy as np
import sys
import random
from threading import Thread
sys.path.append('C:/study/ProjectB/')
sys.path.append('C:/study/ProjectB/sam2')
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from source.utils import *

# height = 240
# width = 320
height = 480
width = 640
confidence_threshold = 0.5

# SAM2 segmentation function
def sam2_segment_frame(predictor, frame, prompt):
    image = cv2.resize(frame, (width, height))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with torch.no_grad():
        predictor.set_image(image_rgb)
        masks, scores, logits = predictor.predict(
            point_coords=prompt["points"],
            point_labels=prompt["labels"]
        )

    # Filter masks based on confidence threshold
    high_confidence_masks = [(mask, score) for mask, score in zip(masks, scores) if score > confidence_threshold]

    if not high_confidence_masks:
        return image, None, None  # If no masks meet the threshold, return the original image

    # Choose the best mask based on scores
    best_mask, best_score = max(high_confidence_masks, key=lambda x: x[1])
    segmentation = (best_mask * 255).astype(np.uint8)
    overlay = cv2.addWeighted(image, 0.7, cv2.cvtColor(segmentation, cv2.COLOR_GRAY2BGR), 0.3, 0)

    return overlay, best_mask

# Paths for checkpoint and config
checkpoint = "C:/study/ProjectB/sam2/checkpoints/sam2.1_hiera_tiny.pt"
model_cfg = "C:/study/ProjectB/sam2/sam2/configs/sam2.1/sam2.1_hiera_t.yaml"

# Build the predictor
device = "cuda" if torch.cuda.is_available() else "cpu"
model = build_sam2(model_cfg, checkpoint, device=device)
predictor = SAM2ImagePredictor(model)

# Define the prompt (points and labels)
center_x, center_y = width // 2, height // 2
radius = 25
num_points = 5
angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
circle_points = np.array([[int(center_x + radius * np.cos(angle)), int(center_y + radius * np.sin(angle))] for angle in angles])
all_points = np.vstack([[center_x, center_y], circle_points])
labels = np.ones(all_points.shape[0], dtype=np.int32)
prompt = {"points": all_points, "labels": labels}

# Optical flow variables
cap = cv2.VideoCapture(0)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame, best_mask = sam2_segment_frame(predictor, frame, prompt)
        
        if best_mask is None:
            continue
        
        # Calculate orientation and draw arrow if valid
        center_x, center_y, angle = calculate_pca_rotation_angle_from_mask(best_mask)
        if angle is None:
            continue

        print("angle = ", angle)
        arrow_length = 50
        end_x = int(center_x - arrow_length * np.cos(np.deg2rad(angle)))
        end_y = int(center_y - arrow_length * np.sin(np.deg2rad(angle)))

        cv2.arrowedLine(processed_frame, (int(center_y), int(center_x)), (end_y, end_x), (0, 255, 0), 2, tipLength=0.3)
        cv2.imshow("Real-Time Segmentation and Tracking", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()

