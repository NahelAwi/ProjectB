import cv2
import torch
import numpy as np
import sys
import depthai as dai
from utils import *

# Use CUDA if available.
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Global parameters (tweak these to suit your needs)
FPS = 30
ROI_KERNEL_SIZE = 200  # size (in pixels) of the central ROI
MIN_VALID_PIXELS = 50  # require at least this many valid depth pixels in the ROI
SMOOTHING_ALPHA = 0.2  # smoothing factor for temporal filtering (0 = no update, 1 = immediate)

# Create the pipeline.
pipeline = create_RGB_Depth_pipeline()

# For temporal smoothing, we keep a running filtered depth value.
filtered_depth = None

with dai.Device(pipeline) as device:
    rgb_queue = device.getOutputQueue(name="rgb", maxSize=15, blocking=False)
    depth_queue = device.getOutputQueue(name="depth", maxSize=15, blocking=False)

    while True:
        # Retrieve frames from device.
        depth_frame = depth_queue.get().getFrame()  # depth frame (in millimeters)
        rgb_frame = rgb_queue.get().getCvFrame()

        depth_frame = cv2.resize(depth_frame, (width, height), interpolation=cv2.INTER_NEAREST)

        # Optionally apply an additional median blur on the depth frame.
        # (This is in addition to the stereo node’s internal median filter.)
        depth_frame = cv2.medianBlur(depth_frame, 5)

        print("depth_frame.shape = ", depth_frame.shape)

        # Compute the ROI in the center.
        center_x, center_y = width // 2, height // 2
        roi_half = ROI_KERNEL_SIZE // 2
        start_x = max(0, center_x - roi_half)
        start_y = max(0, center_y - roi_half)
        end_x = min(width, center_x + roi_half)
        end_y = min(height, center_y + roi_half)
        center_region = depth_frame[start_y:end_y, start_x:end_x]

        # Create a mask for valid (non-zero) depth readings.
        valid_mask = center_region > 0
        valid_depths = center_region[valid_mask]

        if valid_depths.size < MIN_VALID_PIXELS:
            # If there are too few valid points, assume no reliable object is present.
            current_depth = filtered_depth if filtered_depth is not None else 0
        else:
            # Sort valid depths (lowest values = closest points).
            sorted_depths = np.sort(valid_depths, axis=None)
            # You can choose to use a subset (e.g. the 2000 closest points) to avoid outliers.
            num_points = min(len(sorted_depths), 2000)
            closest_depths = sorted_depths[:num_points]
            current_depth = np.median(closest_depths)

        # Temporal smoothing: update our filtered depth value.
        if filtered_depth is None:
            filtered_depth = current_depth
        else:
            filtered_depth = SMOOTHING_ALPHA * current_depth + (1 - SMOOTHING_ALPHA) * filtered_depth

        # Optionally: reject a depth reading if it is suspiciously “low” (e.g. exactly ~120mm)
        # when the object is far. Here you might add logic comparing filtered_depth with current_depth,
        # or checking the variance of values in the ROI.

        # Draw the ROI rectangle on the RGB frame.
        cv2.rectangle(rgb_frame, (start_x, start_y), (end_x, end_y), (0, 255, 255), 2)
        # Optionally, display a small circle at the center.
        cv2.circle(rgb_frame, (center_x, center_y), 3, (0, 0, 255), -1)

        # Display the current depth reading on the image.
        cv2.putText(rgb_frame, f"Depth: {int(filtered_depth)} mm", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Debug prints.
        print(f"Current median depth (smoothed): {filtered_depth:.2f} mm")

        # Display the annotated RGB frame.
        cv2.imshow("RGB Frame with ROI", rgb_frame)

        # Press 'q' to quit.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
