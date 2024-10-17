import depthai as dai
import cv2
import numpy as np
import math
import blobconverter
from pathlib import Path
import time
import argparse

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