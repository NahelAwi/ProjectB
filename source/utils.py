import depthai as dai
import cv2
import numpy as np
import math
import blobconverter
from pathlib import Path
import time
import argparse
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
from hed_net import *
import os
import torch.optim as optim
import getopt
import PIL
import PIL.Image
import sys
import matplotlib.pyplot as plt
import torchvision.transforms as T

# # Function to create DepthAI pipeline for RGB and depth streams
# def create_pipeline():
#     pipeline = dai.Pipeline()

#     # RGB Camera
#     cam_rgb = pipeline.create(dai.node.ColorCamera)
#     cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
#     cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
#     cam_rgb.setInterleaved(False)
#     cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

#     # Depth Camera (Stereo pair)
#     mono_left = pipeline.create(dai.node.MonoCamera)
#     mono_right = pipeline.create(dai.node.MonoCamera)
#     stereo = pipeline.create(dai.node.StereoDepth)

#     mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
#     mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)

#     stereo.setLeftRightCheck(True)
#     stereo.setExtendedDisparity(False)
#     stereo.setSubpixel(True)

#     # Linking
#     cam_rgb_out = pipeline.createXLinkOut()
#     depth_out = pipeline.createXLinkOut()

#     cam_rgb_out.setStreamName("rgb")
#     depth_out.setStreamName("depth")

#     cam_rgb.video.link(cam_rgb_out.input)
#     stereo.depth.link(depth_out.input)

#     return pipeline

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

def create_pipeline():
    # Create a pipeline
    pipeline = dai.Pipeline()

    # Define sources and outputs
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    detectionNetwork = pipeline.create(dai.node.YoloDetectionNetwork)
    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_nn = pipeline.create(dai.node.XLinkOut)

    xout_rgb.setStreamName("rgb")
    xout_nn.setStreamName("detectionNetwork")

    # Configure camera properties
    cam_rgb.setPreviewSize(416, 416)  # Input size for the model
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam_rgb.setFps(30)

    # This will automatically download the blob and return the path to it
    blob_path = blobconverter.from_zoo(name="yolo-v3-tiny-tf", shaves=6)
    print(f"Blob path: {blob_path}")

    # Set model path
    detectionNetwork.setBlobPath(blob_path)
    detectionNetwork.input.setBlocking(False)
    detectionNetwork.input.setQueueSize(1)
    # Network specific settings
    detectionNetwork.setConfidenceThreshold(0.5)
    detectionNetwork.setNumClasses(80)
    detectionNetwork.setCoordinateSize(4)
    detectionNetwork.setAnchors([10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326])

    detectionNetwork.setAnchorMasks({
        "side52": [0, 1, 2],  # Small objects
        "side26": [3, 4, 5],  # Medium objects
        "side13": [6, 7, 8]   # Large objects
    })

    detectionNetwork.setIouThreshold(0.5)
    detectionNetwork.setNumInferenceThreads(2)


    # Linking
    cam_rgb.preview.link(detectionNetwork.input)
    # detectionNetwork.passthrough.link(xout_rgb.input)
    cam_rgb.preview.link(xout_rgb.input)
    detectionNetwork.out.link(xout_nn.input)
    
    return pipeline


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

netNetwork = None
def estimate(hed_input):
    global netNetwork

    if netNetwork is None:
        netNetwork = HedNetwork().eval()
        netNetwork = torch.jit.script(netNetwork)

    intWidth = hed_input.shape[2]
    intHeight = hed_input.shape[1]

    # assert(intWidth == 480) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue
    # assert(intHeight == 320) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue
    with torch.no_grad():
        return netNetwork(hed_input.view(1, 3, intHeight, intWidth))[0, :, :, :].cpu()

torch.set_grad_enabled(False)
torch.backends.cudnn.enabled = True

def process_frame(img_in):
    
    hed_input = torch.FloatTensor(
        np.ascontiguousarray(
            img_in[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)
        )
    )
    
    hed_output = estimate(hed_input)

    hed_output = (hed_output.clip(0.0, 1.0).numpy().transpose(1, 2, 0)[:, :, 0] * 255.0).astype(np.uint8)

    # Use contour detection to locate the central object without relying on segmentation
    # Apply Gaussian Blur to smooth the image and reduce noise
    blurred_image = cv2.GaussianBlur(hed_output, (5, 5), 0)

    # Detect edges using Canny Edge Detector
    canny_edges = cv2.Canny(blurred_image, 50, 150)

    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(canny_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Define center region for the cube (based on location in the initial image)
    width = hed_output.shape[1]
    height = hed_output.shape[0]
        
    center_x, center_y = width // 2, height // 2
    center_region_radius = min(width, height) // 5

    # Filter contours based on proximity to center
    filtered_contours = []
    for cnt in contours:
        # Get the center of each contour
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            # Check if the contour center is within the central region
            if abs(cx - center_x) < center_region_radius and abs(cy - center_y) < center_region_radius:
                filtered_contours.append(cnt)

    # Create an empty mask to draw filtered contours
    contour_mask = np.zeros_like(hed_output)
    cv2.drawContours(contour_mask, filtered_contours, -1, (255), thickness=cv2.FILLED)

    # Combine the contour mask with the original edges to isolate the "main" object
    isolated_edges = cv2.bitwise_and(hed_output, contour_mask)

    center_x, center_y, angle = calculate_pca_rotation_angle_from_edge_image(isolated_edges)
    if angle is None:
        return

    # Draw the orientation arrow
    arrow_length = 50
    end_x = int(center_x - arrow_length * np.cos(np.deg2rad(angle)))
    end_y = int(center_y - arrow_length * np.sin(np.deg2rad(angle)))

    output_image = cv2.cvtColor(isolated_edges, cv2.COLOR_GRAY2RGB)
    cv2.arrowedLine(output_image, (int(center_y), int(center_x)), (end_y, end_x), (0, 255, 0), 2, tipLength=0.3)

    # Display the refined result
    # fig, ax = plt.subplots(1, 3, figsize=(12, 6))
    # plt.imshow(hed_output, cmap='gray')
    # plt.title("Original Edge Detected Image")
    # plt.axis('off')

    # plt.imshow(isolated_edges, cmap='gray')
    # plt.title("Contour-Based Isolation")
    # plt.axis('off')
    
    # # Display the final image with orientation arrow
    # plt.imshow(output_image, cmap='gray')
    # plt.title("Object Orientation")
    # plt.axis('off')

    # plt.savefig(img_out, bbox_inches='tight', dpi=300)
    # plt.show()
    
    cv2.imshow("Original Edge Detected Image", hed_output)
    cv2.imshow("Contour-Based Isolation", isolated_edges)
    cv2.imshow("Object Orientation", output_image)

    print("angle = ", angle)    
    