from ultralytics import FastSAM
import cv2
import torch
import numpy as np
import sys
from utils import *

height = 480
width = 640
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the FastSAM model
model_path = "FastSAM-s.pt"  # or FastSAM-x.pt
model = FastSAM(model_path).to(device)

# Define the prompt (points and labels)
center_x, center_y = width // 2, height // 2
radius = 40
num_points = 40
angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
circle_points = np.array([[int(center_x + radius * np.cos(angle)), int(center_y + radius * np.sin(angle))] for angle in angles])
all_points = np.vstack([[center_x, center_y], circle_points])
labels = np.ones(all_points.shape[0], dtype=np.int32)

# Optical flow variables
# cap = cv2.VideoCapture(0)
    
def Calculate_Depth(depth_frame):
    # Retrieve only the center depth value
    center_x = width // 2
    center_y = height // 2

    kernel_size = 30  # Adjust the size of the window
    start_x = center_x - kernel_size // 2
    start_y = center_y - kernel_size // 2

    # Extract a small region around the center
    center_region = depth_frame[start_y:start_y + kernel_size, start_x:start_x + kernel_size]

    # Apply median filtering to stabilize the depth value
    # filtered_depth = cv2.medianBlur(center_region, 5)

    # Calculate the median depth in that region
    center_depth_median = np.median(center_region)

    return center_depth_median

def process_frame(frame_rgb):
    frame_resized = cv2.resize(frame_rgb, (width, height)) / 255.0  # Normalize to [0, 1]
    input_tensor = torch.tensor(frame_resized).permute(2, 0, 1).unsqueeze(0).float().to(device)

    try:
        # Run inference with points prompt
        with torch.no_grad():
            results = model(input_tensor, points=all_points, labels=labels)
    except:
        return frame_rgb, None

    if not (results and len(results) > 0):
        return frame_rgb, None

    # Extract masks from results
    best_result = results[0]  # Take the first result (assuming sorted by confidence)
    if len(best_result.masks.data) == 0:
        return frame_rgb, None
    best_mask = best_result.masks.data[0]
    segmentation = (best_mask.cpu().numpy() * 255).astype(np.uint8)
    overlay = cv2.addWeighted(frame_rgb, 0.7, cv2.cvtColor(segmentation, cv2.COLOR_GRAY2BGR), 0.3, 0)
    processed_frame = overlay

    # Calculate orientation and draw arrow if valid
    center_x, center_y, angle = calculate_pca_rotation_angle_from_mask(best_mask)
    if angle is not None:
        arrow_length = 50
        end_x = int(center_x - arrow_length * np.cos(np.deg2rad(angle)))
        end_y = int(center_y - arrow_length * np.sin(np.deg2rad(angle)))
        cv2.arrowedLine(processed_frame, (int(center_y), int(center_x)), (end_y, end_x), (0, 255, 0), 2, tipLength=0.3)
        print("angle = ", angle)
    return processed_frame, angle

def calc_angle_thread(queue):
    pipeline = create_RGB_Depth_pipeline()
    old_angle = 0
    with dai.Device(pipeline) as devicex:
        rgb_queue = devicex.getOutputQueue(name="rgb", maxSize=15, blocking=False)
        depth_queue = devicex.getOutputQueue(name="depth", maxSize=15, blocking=False)
        # left_queue = devicex.getOutputQueue(name="left")
        # right_queue = devicex.getOutputQueue(name="right")
        

        try:
            while True:
                depth_frame = depth_queue.get().getFrame()  # Get the full depth frame

                depth = Calculate_Depth(depth_frame)

                print(f"Center depth: {depth} mm")

                ret = rgb_queue.get()

                # # Get left frame
                # left_frame = left_queue.get().getCvFrame()
                # # Get right frame 
                # right_frame = right_queue.get().getCvFrame()

                # imOut = np.hstack((left_frame, right_frame))
                # imOut = np.uint8(left_frame/2 + right_frame/2)

                # Display output image
                # cv2.imshow("Stereo Pair", imOut)
                
                frame = None
                if ret:
                    frame = ret.getCvFrame()
                else:
                    continue

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                angle = None
                if(depth > 200 and depth < 300): 
                    # Pre-process the frame
                    processed_frame, angle = process_frame(frame_rgb)
                else:
                    processed_frame = frame_rgb
                    
                if angle:
                    if abs(angle-old_angle) <= 2:#margin
                        angle = 0
                    else:
                        tmp = angle
                        angle -= old_angle
                        old_angle = tmp
                    queue.put(angle)
                
                
                cv2.imshow("Real-Time Segmentation", processed_frame)
                # cv2.imshow("Depth", depth_frame)
                # cv2.imshow("left", left_frame)
                # cv2.imshow("right", right_frame)
                


                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            cv2.destroyAllWindows()