from ultralytics import FastSAM
import cv2
import torch
import numpy as np
import sys
from source.utils import *

def process_frame(frame_rgb):
    frame_resized = cv2.resize(frame_rgb, (width, height)) / 255.0  # Normalize to [0, 1]
    input_tensor = torch.tensor(frame_resized).permute(2, 0, 1).unsqueeze(0).float().to(device)

    try:
        # Run inference with points prompt
        with torch.no_grad():
            results = model(input_tensor, points=all_points, labels=labels)
    except:
        return frame_rgb

    if not (results and len(results) > 0):
        return frame_rgb

    # Extract masks from results
    best_result = results[0]  # Take the first result (assuming sorted by confidence)
    if len(best_result.masks.data) == 0:
        return frame_rgb
    best_mask = best_result.masks.data[0]
    segmentation = (best_mask.cpu().numpy() * 255).astype(np.uint8)
    overlay = cv2.addWeighted(frame, 0.7, cv2.cvtColor(segmentation, cv2.COLOR_GRAY2BGR), 0.3, 0)
    processed_frame = overlay

    # Calculate orientation and draw arrow if valid
    center_x, center_y, angle = calculate_pca_rotation_angle_from_mask(best_mask)
    if angle is not None:
        arrow_length = 50
        end_x = int(center_x - arrow_length * np.cos(np.deg2rad(angle)))
        end_y = int(center_y - arrow_length * np.sin(np.deg2rad(angle)))
        cv2.arrowedLine(processed_frame, (int(center_y), int(center_x)), (end_y, end_x), (0, 255, 0), 2, tipLength=0.3)
    return processed_frame

# Load the FastSAM model
model_path = "FastSAM-s.pt"  # or FastSAM-x.pt
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = FastSAM(model_path).to(device)

height = 480
width = 640

# Define the prompt (points and labels)
center_x, center_y = width // 2, height // 2
radius = 40
num_points = 40
angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
circle_points = np.array([[int(center_x + radius * np.cos(angle)), int(center_y + radius * np.sin(angle))] for angle in angles])
all_points = np.vstack([[center_x, center_y], circle_points])
labels = np.ones(all_points.shape[0], dtype=np.int32)

# Optical flow variables
cap = cv2.VideoCapture(0)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Pre-process the frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        processed_frame = process_frame(frame_rgb)
        cv2.imshow("Real-Time Segmentation", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()