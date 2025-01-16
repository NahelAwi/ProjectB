# from ultralytics import FastSAM
from ultralytics.models.fastsam.model import FastSAM
import cv2
import torch
import numpy as np
import sys
from torch.cuda.amp import GradScaler, autocast
from utils import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = "cpu"
print("device = ", device)

# Load the FastSAM model
model_path = "FastSAM-s.pt"  # or FastSAM-x.pt
print("initializing model ...")
model = FastSAM(model_path).to(device).eval()
print("done")

#possible optimizations:
# 1- pick the segmentation mask in a "smarter" way to avoid segmrntation of : 1) background 2) small details in the object. --> for example take mask with middle size
# 2- change fastsam num_points and radius and see how it affects small objects

# Define the prompt (points and labels)
center_x, center_y = width // 2, height // 2
radius = 15
num_points = 30
angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
circle_points = np.array([[int(center_x - radius * np.cos(angle)), int(center_y - radius * np.sin(angle))] for angle in angles])
all_points = np.vstack([[center_x, center_y], circle_points])
labels = np.ones(all_points.shape[0], dtype=np.int32)

def Calculate_Depth(depth_frame):
    # Retrieve only the center depth value

    kernel_size = 250  # Adjust the size of the window
    start_x = center_x - kernel_size // 2
    start_y = center_y - kernel_size // 2

    # Extract a small region around the center
    center_region = depth_frame[start_y:start_y + kernel_size, start_x:start_x + kernel_size]

    # Flatten the region and filter out zero values
    flattened_depths = center_region.flatten()
    non_zero_depths = flattened_depths[flattened_depths > 0]

    # Sort the non-zero depths and take the 200 closest points
    sorted_depths = np.sort(non_zero_depths)
    closest_pixels = sorted_depths[:min(len(sorted_depths), 200)]

    # Calculate the median depth in that region
    center_depth_median = np.median(closest_pixels)

    return center_depth_median

def draw_mask(frame, masks):
    debug_frame = frame.copy()

    num_masks = len(masks)
    colors = [tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(num_masks)]

    for i, mask in enumerate(masks):
        points = torch.nonzero(mask == 1, as_tuple=False)
        print(f"mask_{i} size = ", points.size(0))
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()

        color_mask = np.zeros_like(debug_frame, dtype=np.uint8)
        color_mask[mask == 1] = colors[i]

        debug_frame = cv2.addWeighted(debug_frame, 0.7, color_mask, 0.3, 0)

    return debug_frame

def mask_points_size(mask):
    points = torch.nonzero(mask == 1, as_tuple=False)
    return points.size(0)

def calculate_compactness(mask):
    """
    Calculates compactness of the mask.

    Args:
        mask (numpy.ndarray): Binary mask (assumed 0s and 1s).

    Returns:
        float: Compactness score.
    """
    # Ensure the mask is in the correct type and range
    mask_binary = (mask > 0).astype(np.uint8)  # Convert to 8-bit binary

    # Find contours
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return float('inf')  # If no contours, return an infinite compactness

    # Use the largest contour (by area)
    largest_contour = max(contours, key=cv2.contourArea)

    # Calculate perimeter and area of the largest contour
    perimeter = cv2.arcLength(largest_contour, True)
    area = cv2.contourArea(largest_contour)

    # Compactness formula: Perimeter^2 / (4 * π * Area)
    compactness = (perimeter ** 2) / (4 * np.pi * area) if area > 0 else float('inf')
    return compactness

def distance_to_center(mask):
    """
    Calculates a score that increases for large masks with points far from the center.

    Args:
        mask (torch.Tensor): Binary mask (1 for object, 0 for background).

    Returns:
        float: Distance score (higher for larger masks farther from the center).
    """
    # Get all nonzero points in the mask
    coords = torch.nonzero(mask == 1, as_tuple=False)
    if coords.size(0) == 0:
        return 0  # No points in the mask, return 0

    # Calculate the Euclidean distance of each point to the center
    distances = torch.norm(coords - torch.tensor((center_x, center_y), dtype=torch.float), dim=1)

    # Calculate the average distance
    avg_distance = distances.mean().item()

    # # Multiply average distance by the size of the mask (number of points)
    # score = avg_distance * coords.size(0)

    return avg_distance

def calculate_mask_score(mask, index, weight_size=30, weight_compactness=10.0, weight_distance=4.0, weight_confidence=80.0):
    """
    Calculates a combined score for the mask based on multiple criteria.

    Args:
        mask (torch.Tensor): Binary mask.
        frame_center (tuple): (x, y) coordinates of the frame center.
        weight_size (float): Weight for mask size.
        weight_compactness (float): Weight for compactness.
        weight_distance (float): Weight for distance to center.

    Returns:
        float: Combined score.
    """
    mask_np = mask.cpu().numpy()
    mask_np = (mask_np > 0).astype(np.uint8)  # Convert to 8-bit binary
    size = torch.sum(mask)
    
    if (size.item() > 0.9*width*height or size.item() < 0.01*width*height):
        return float("-inf")
    
    compactness = calculate_compactness(mask_np)
    distance = distance_to_center(mask)
    
    # Logarithmic scaling for size
    scaled_size = torch.log1p(size).item()  # log1p(x) = log(1 + x), avoids log(0)
    
    print("size = ", scaled_size)
    print("distance = ", distance)
    print("compactness = ", compactness)

    # Normalize and combine scores
    score = (weight_size * scaled_size) - (weight_compactness * compactness) - (weight_distance * distance) - (index * weight_confidence)  # lower index has higher confidence
    return score


def process_frame(frame_rgb):
    frame_resized = cv2.resize(frame_rgb, (width, height)) / 255.0  # Normalize to [0, 1]
    input_tensor = torch.tensor(frame_resized).permute(2, 0, 1).unsqueeze(0).float().to(device)

    try:
        # Run inference with points prompt
        with torch.no_grad():
            with autocast():
                print("start inference")
                results = model(input_tensor, points=all_points, labels=labels)
                print("done inference")
    except Exception as e:
        print(f"Inference failed with error: {e}")
        return frame_rgb, None

    if not (results and len(results) > 0):
        return frame_rgb, None

    # Extract masks from results
    result = results[0]
    masks = result.masks.data
    if len(masks) == 0:
        return frame_rgb, None
    print("len(masks) = ", len(masks))
    
    # Calculate scores and select the best mask
    scores = [calculate_mask_score(mask, index) for index, mask in enumerate(masks)]
    best_mask_idx = np.argmax(scores)
    best_mask = masks[best_mask_idx]

    # points = mask_points_size(best_mask)

    segmentation = (best_mask.cpu().numpy() * 255).astype(np.uint8)
    overlay = cv2.addWeighted(frame_rgb, 0.7, cv2.cvtColor(segmentation, cv2.COLOR_GRAY2BGR), 0.3, 0)
    processed_frame = overlay
    
    # processed_frame = draw_mask(frame_rgb, masks)
    
    # Calculate orientation and draw arrow if valid
    center_x, center_y, angle = calculate_pca_rotation_angle_from_mask(best_mask)
    if angle is not None:
        arrow_length = 50
        end_x = int(center_x - arrow_length * np.cos(np.deg2rad(angle)))
        end_y = int(center_y - arrow_length * np.sin(np.deg2rad(angle)))
        for point in circle_points:
            x,y = point
            cv2.circle(processed_frame, (x,y), radius=3, color=(255,0,0), thickness=-1)
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

                frame_rgb = frame#cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                angle = None
                if(depth > 170 and depth < 250): 
                    # Pre-process the frame
                    processed_frame, angle = process_frame(frame_rgb)
                else:
                    processed_frame = frame_rgb
                    
                if angle:
                    if abs(angle-old_angle) <= 10:#margin
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