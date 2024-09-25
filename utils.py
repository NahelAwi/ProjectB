import depthai as dai
import cv2
import numpy as np
import math

# Function to create DepthAI pipeline for RGB and depth streams
def create_pipeline():
    pipeline = dai.Pipeline()

    # RGB Camera
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

    # Depth Camera (Stereo pair)
    mono_left = pipeline.create(dai.node.MonoCamera)
    mono_right = pipeline.create(dai.node.MonoCamera)
    stereo = pipeline.create(dai.node.StereoDepth)

    mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
    mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)

    stereo.setLeftRightCheck(True)
    stereo.setExtendedDisparity(False)
    stereo.setSubpixel(True)

    # Linking
    cam_rgb_out = pipeline.createXLinkOut()
    depth_out = pipeline.createXLinkOut()

    cam_rgb_out.setStreamName("rgb")
    depth_out.setStreamName("depth")

    cam_rgb.video.link(cam_rgb_out.input)
    stereo.depth.link(depth_out.input)

    return pipeline
