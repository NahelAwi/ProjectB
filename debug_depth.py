from ultralytics import FastSAM
import cv2
import torch
import numpy as np
import sys
from utils import *

height = 480
width = 640
device = 'cuda' if torch.cuda.is_available() else 'cpu'

pipeline = create_RGB_Depth_pipeline()
old_angle = 0
with dai.Device(pipeline) as devicex:
    rgb_queue = devicex.getOutputQueue(name="rgb", maxSize=15, blocking=False)
    depth_queue = devicex.getOutputQueue(name="depth", maxSize=15, blocking=False)
    
    while True:
        depth_frame = depth_queue.get().getFrame()  # Get the full depth frame
        rgb_frame = cv2.cvtColor(rgb_queue.get().getCvFrame(), cv2.COLOR_BGR2RGB)
        
        kernel_size = 50
        # Retrieve the center coordinates of the frame
        # height, width = depth_frame.shape
        center_x = width // 2
        center_y = height // 2

        # Define the region of interest around the center
        start_x = max(0, center_x - kernel_size // 2)
        start_y = max(0, center_y - kernel_size // 2)
        end_x = min(width, center_x + kernel_size // 2)
        end_y = min(height, center_y + kernel_size // 2)

        # Extract the region around the center
        center_region = depth_frame[start_y:end_y, start_x:end_x]

        # Get the indices of non-zero depth values
        non_zero_indices = np.where(center_region > 0)
        non_zero_coords = list(zip(non_zero_indices[0] + start_y, non_zero_indices[1] + start_x))
        non_zero_depths = center_region[non_zero_indices]

        # Sort by depth and take the 100 closest points
        sorted_indices = np.argsort(non_zero_depths)
        closest_indices = sorted_indices[:100]
        closest_depths = non_zero_depths[closest_indices]
        closest_coords = np.array([non_zero_coords[i] for i in closest_indices])
        
        if len(closest_coords) > 0:
            all_points = np.vstack([[center_x, center_y], closest_coords])

        # Calculate the mean of the 50 closest pixels
        depth = np.mean(closest_depths)
        points = closest_coords
        
        print("Depth:", depth)

        # Plot the points on the RGB frame
        for point in points:
            y, x = point  # Extract coordinates
            cv2.circle(rgb_frame, (x, y), radius=3, color=(0, 255, 0), thickness=-1)  # Draw points
        
        # Display the RGB frame with plotted points
        cv2.imshow("RGB Frame with Points", rgb_frame)
        
        # Break on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
        
        
        
    