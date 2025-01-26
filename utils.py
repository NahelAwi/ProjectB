import depthai as dai
import cv2
import numpy as np
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import os
import sys
import matplotlib.pyplot as plt

width = 320
height = 256

# # Function to create DepthAI pipeline for RGB and depth streams
def create_RGB_pipeline():
    pipeline = dai.Pipeline()

    rgb_cam = pipeline.create(dai.node.ColorCamera)
    rgb_cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    # rgb_cam = pipeline.createColorCamera()
    # rgb_cam.setBoardSocket(dai.CameraBoardSocket.RGB)
    rgb_cam.setInterleaved(False)
    rgb_cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_400_P)
    rgb_cam.setPreviewSize(width,height)
    rgb_cam.setPreviewKeepAspectRatio(False)
    # rgb_cam.setFps(FPS)

    # Create outputs
    rgb_output = pipeline.create(dai.node.XLinkOut)
    rgb_output.setStreamName("rgb")

    # Link nodes
    rgb_cam.preview.link(rgb_output.input)

    return pipeline

labelMap = [
    "person",         "bicycle",    "car",           "motorbike",     "aeroplane",   "bus",           "train",
    "truck",          "boat",       "traffic light", "fire hydrant",  "stop sign",   "parking meter", "bench",
    "bird",           "cat",        "dog",           "horse",         "sheep",       "cow",           "elephant",
    "bear",           "zebra",      "giraffe",       "backpack",      "umbrella",    "handbag",       "tie",
    "suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball", "kite",          "baseball bat",
    "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",      "wine glass",    "cup",
    "fork",           "knife",      "spoon",         "bowl",          "banana",      "apple",         "sandwich",
    "orange",         "broccoli",   "carrot",        "hot dog",       "pizza",       "donut",         "cake",
    "chair",          "sofa",       "pottedplant",   "bed",           "diningtable", "toilet",        "tvmonitor",
    "laptop",         "mouse",      "remote",        "keyboard",      "cell phone",  "microwave",     "oven",
    "toaster",        "sink",       "refrigerator",  "book",          "clock",       "vase",          "scissors",
    "teddy bear",     "hair drier", "toothbrush"
]

# def create_pipeline():
#     # Create a pipeline
#     pipeline = dai.Pipeline()

#     # Define sources and outputs
#     cam_rgb = pipeline.create(dai.node.ColorCamera)
#     detectionNetwork = pipeline.create(dai.node.YoloDetectionNetwork)
#     xout_rgb = pipeline.create(dai.node.XLinkOut)
#     xout_nn = pipeline.create(dai.node.XLinkOut)

#     xout_rgb.setStreamName("rgb")
#     xout_nn.setStreamName("detectionNetwork")

#     # Configure camera properties
#     cam_rgb.setPreviewSize(416, 416)  # Input size for the model
#     cam_rgb.setInterleaved(False)
#     cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
#     cam_rgb.setFps(30)

#     # This will automatically download the blob and return the path to it
#     blob_path = blobconverter.from_zoo(name="yolo-v3-tiny-tf", shaves=6)
#     print(f"Blob path: {blob_path}")

#     # Set model path
#     detectionNetwork.setBlobPath(blob_path)
#     detectionNetwork.input.setBlocking(False)
#     detectionNetwork.input.setQueueSize(1)
#     # Network specific settings
#     detectionNetwork.setConfidenceThreshold(0.5)
#     detectionNetwork.setNumClasses(80)
#     detectionNetwork.setCoordinateSize(4)
#     detectionNetwork.setAnchors([10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326])

#     detectionNetwork.setAnchorMasks({
#         "side52": [0, 1, 2],  # Small objects
#         "side26": [3, 4, 5],  # Medium objects
#         "side13": [6, 7, 8]   # Large objects
#     })

#     detectionNetwork.setIouThreshold(0.5)
#     detectionNetwork.setNumInferenceThreads(2)


#     # Linking
#     cam_rgb.preview.link(detectionNetwork.input)
#     # detectionNetwork.passthrough.link(xout_rgb.input)
#     cam_rgb.preview.link(xout_rgb.input)
#     detectionNetwork.out.link(xout_nn.input)
    
#     return pipeline

def create_RGB_Depth_pipeline():
    pipeline = dai.Pipeline()

    # Create RGB camera node
    rgb_cam = pipeline.create(dai.node.ColorCamera)
    rgb_cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    rgb_cam.setInterleaved(False)
    rgb_cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    rgb_cam.setPreviewSize(width,height)
    rgb_cam.setPreviewKeepAspectRatio(False)
    # rgb_cam.setFps(FPS)

    # Create outputs
    rgb_output = pipeline.create(dai.node.XLinkOut)
    rgb_output.setStreamName("rgb")

    # Link nodes
    rgb_cam.preview.link(rgb_output.input)


    # Set up stereo depth
    mono_left = pipeline.create(dai.node.MonoCamera)
    mono_right = pipeline.create(dai.node.MonoCamera)

    mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
    mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)

    mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)


    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.initialConfig.setConfidenceThreshold(200)
    stereo.setRectifyEdgeFillColor(0)  # Black, to better see the cutout
    stereo.setExtendedDisparity(True)
    stereo.setSubpixel(True)  # Enable subpixel precision for finer depth
    stereo.initialConfig.setMedianFilter(dai.StereoDepthProperties.MedianFilter.KERNEL_5x5)
    # stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.setLeftRightCheck(True)  # Helps resolve depth for near objects
    stereo.initialConfig.setDisparityShift(80) # TODO - does this add noise ? Tune this to optimal value (after defining camera place on the hand) - and change hand and fastsam to use the new min depth

    # Link mono cameras to stereo node
    mono_left.out.link(stereo.left)
    mono_right.out.link(stereo.right)

    # Create depth output
    xout_depth = pipeline.create(dai.node.XLinkOut)
    xout_depth.setStreamName("depth")
    stereo.depth.link(xout_depth.input)


    # xout_left = pipeline.create(dai.node.XLinkOut)
    # xout_left.setStreamName("left")
    # xout_right = pipeline.create(dai.node.XLinkOut)
    # xout_right.setStreamName("right")

    # # Attach cameras to output Xlink
    # mono_left.out.link(xout_left.input)
    # mono_right.out.link(xout_right.input)

    # Enable infrared (IR) projector for better depth perception
    # ir_led = pipeline.create(dai.node.LED)
    # ir_led.setBoardSocket(dai.CameraBoardSocket.AUTO)  # Control the IR LED automatically
    # ir_led.setBrightness(100)  # Set brightness level (0-100)

    return pipeline

def Calculate_Depth(depth_frame):
    # Retrieve only the center depth value

    center_x, center_y = width // 2, height // 2

    kernel_size = 150  # Adjust the size of the window
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

def frameNorm(frame, bbox):
    normVals = np.full(len(bbox), frame.shape[0])
    normVals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

def displayFrame(name, frame, detections):
    color = (255, 0, 0)
    for detection in detections:
        bbox = (detection.xmin, detection.ymin, detection.xmax, detection.ymax)
        print("bbox = ", bbox)
        bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
        cv2.putText(frame, labelMap[detection.label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
        cv2.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
    # Show the frame
    cv2.imshow(name, frame)


def calculate_pca_rotation_angle(keypoints):
    """Calculate the rotation angle of the object using PCA on all keypoints."""
    if len(keypoints) < 2:
        return None

    # Center the keypoints by subtracting the centroid
    centroid = np.mean(keypoints, axis=0)
    centered_keypoints = keypoints - centroid

    # Perform PCA on the keypoints
    cov_matrix = np.cov(centered_keypoints, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # The principal axis is the eigenvector with the largest eigenvalue
    principal_axis = eigenvectors[:, np.argmax(eigenvalues)]

    # Calculate the angle between the principal axis and the x-axis
    angle_radians = np.arctan2(principal_axis[1], principal_axis[0])
    angle_degrees = np.degrees(angle_radians)

    # Normalize the angle to be between 0 and 180 degrees
    if angle_degrees < 0:
        angle_degrees += 180

    return angle_degrees, centroid


def get_rotation_angle_using_hough_lines(edge_image):
    # Step 1: Ensure binary image by thresholding if necessary
    _, binary_image = cv2.threshold(edge_image, 127, 255, cv2.THRESH_BINARY)

    # Step 2: Detect lines using Hough Line Transform
    lines = cv2.HoughLines(binary_image, rho=1, theta=np.pi/180, threshold=100)

    if lines is None:
        print("No lines detected.")
        return None

    # Step 3: Calculate angles of detected lines
    angles = []
    for line in lines:
        rho, theta = line[0]
        angle = np.rad2deg(theta) - 90  # Convert from radians and adjust to typical rotation angle
        angles.append(angle)

    # Step 4: Calculate the median angle as the overall rotation angle
    rotation_angle = np.median(angles)
    return rotation_angle

def calculate_pca_rotation_angle_from_edge_image(edge_image):
    # Extract edge points from the final isolated edges image
    edge_points = np.column_stack(np.where(edge_image > 0))
    
    if len(edge_points) == 0:
        return None, None, None

    # Apply PCA to find the primary orientation
    pca = PCA(n_components=2)
    pca.fit(edge_points)

    # The principal component gives the orientation
    angle_pca = np.arctan2(pca.components_[0, 1], pca.components_[0, 0]) * 180 / np.pi  # Convert to degrees

    # Ensure the angle is within [-90,90] degrees
    angle_pca = angle_pca if angle_pca >= 0 else angle_pca + 180
    angle_pca = angle_pca if angle_pca <= 90 else angle_pca - 180

    center_x, center_y = pca.mean_

    return center_x, center_y, angle_pca

def calculate_pca_rotation_angle_from_mask(mask):
    """
    Calculate PCA rotation angle from a mask using PyTorch tensors.
    Assumes mask is a binary tensor (1 for edges, 0 for background).
    """
    # Extract edge points from the mask (coordinates where mask == 1)
    points = torch.nonzero(mask == 1, as_tuple=False).float()
    return calculate_pca_rotation_angle_from_points(points)

def calculate_pca_rotation_angle_from_points(points):
    """
    Calculate PCA rotation angle from a set of points using PyTorch tensors.
    """
    if points.size(0) == 0:  # Check if there are no points
        return None, None, None

    # Compute the mean of the points
    mean = points.mean(dim=0)
    centered_points = points - mean

    # Compute the covariance matrix
    cov_matrix = centered_points.T @ centered_points / (points.size(0) - 1)

    # Perform eigen decomposition of the covariance matrix
    eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)

    # The principal component is the eigenvector with the largest eigenvalue
    principal_component = eigenvectors[:, -1]  # Last column corresponds to the largest eigenvalue

    # Calculate the angle of the principal component
    angle_pca = torch.atan2(principal_component[1], principal_component[0]) * 180 / torch.pi  # Convert to degrees

    # Ensure the angle is within [-90, 90] degrees
    if angle_pca < 0:
        angle_pca += 180
    if angle_pca > 90:
        angle_pca -= 180

    center_x, center_y = mean.tolist()

    return center_x, center_y, angle_pca.item()