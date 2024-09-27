import depthai as dai
import cv2
import numpy as np
import math
import blobconverter

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

def create_pipeline():
    # Create a pipeline
    pipeline = dai.Pipeline()

    # Define sources and outputs
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    pose_nn = pipeline.create(dai.node.NeuralNetwork)
    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_nn = pipeline.create(dai.node.XLinkOut)

    xout_rgb.setStreamName("rgb")
    xout_nn.setStreamName("nn")

    # Properties
    cam_rgb.setPreviewSize(456, 256)  # Input size for the model
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

    # This will automatically download the blob and return the path to it
    blob_path = blobconverter.from_zoo(name="human-pose-estimation-0001", shaves=6)
    print(f"Blob path: {blob_path}")

    # Set model path
    pose_nn.setBlobPath(blob_path)
    pose_nn.input.setBlocking(False)
    pose_nn.input.setQueueSize(1)

    # Linking
    cam_rgb.preview.link(pose_nn.input)
    cam_rgb.preview.link(xout_rgb.input)
    pose_nn.out.link(xout_nn.input)
    
    return pipeline


def calculate_pca_tilt_angle(keypoints):
    """Calculate the tilt angle of the object using PCA on all keypoints."""
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