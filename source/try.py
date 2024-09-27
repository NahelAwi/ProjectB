import depthai as dai
import cv2
import numpy as np
import blobconverter

# Download YOLO and keypoint models
yolo_blob_path = blobconverter.from_zoo(name="yolo-v3-tiny-tf", shaves=6)  # YOLOv3 Tiny for object detection
keypoint_blob_path = blobconverter.from_zoo(name="human-pose-estimation-0001", shaves=6)  # Keypoint detection

# Create pipeline
pipeline = dai.Pipeline()

# Define camera node
cam_rgb = pipeline.create(dai.node.ColorCamera)
cam_rgb.setPreviewSize(416, 416)  # Set to 416x416 to match YOLO input requirements
cam_rgb.setInterleaved(False)
cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
# cam_rgb.setNumFramesPool(3)  # Ensure 3-channel input (RGB)
cam_rgb.setPreviewKeepAspectRatio(False)

# YOLO object detection neural network
yolo_nn = pipeline.create(dai.node.YoloDetectionNetwork)
yolo_nn.setBlobPath(yolo_blob_path)
yolo_nn.setConfidenceThreshold(0.5)
yolo_nn.setNumInferenceThreads(2)  # Optimize inference speed
yolo_nn.input.setBlocking(False)
yolo_nn.input.setQueueSize(1)

# YOLO specific parameters
yolo_nn.setNumClasses(80)
yolo_nn.setCoordinateSize(4)
yolo_nn.setAnchors([10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319])
yolo_nn.setAnchorMasks({"side26": [1, 2, 3], "side13": [3, 4, 5]})
yolo_nn.setIouThreshold(0.5)

# Keypoint detection neural network
keypoint_nn = pipeline.create(dai.node.NeuralNetwork)
keypoint_nn.setBlobPath(keypoint_blob_path)
keypoint_nn.input.setBlocking(False)
keypoint_nn.input.setQueueSize(1)

# XLink outputs for object detection, keypoint detection, and RGB
xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("rgb")
xout_obj_det = pipeline.create(dai.node.XLinkOut)
xout_obj_det.setStreamName("det")
xout_keypoints = pipeline.create(dai.node.XLinkOut)
xout_keypoints.setStreamName("keypoints")

# Linking
cam_rgb.preview.link(yolo_nn.input)  # Direct link to YOLO
cam_rgb.preview.link(xout_rgb.input)
yolo_nn.out.link(xout_obj_det.input)

# Link the YOLO detection to keypoint NN directly
yolo_nn.out.link(keypoint_nn.input)
keypoint_nn.out.link(xout_keypoints.input)

def draw_bounding_boxes(frame, detections):
    """Draw bounding boxes around detected objects."""
    for detection in detections:
        x_min = int(detection.xmin * frame.shape[1])
        y_min = int(detection.ymin * frame.shape[0])
        x_max = int(detection.xmax * frame.shape[1])
        y_max = int(detection.ymax * frame.shape[0])
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    return frame

def decode_keypoints(output, frame_shape):
    """Decode keypoints from neural network output."""
    keypoints = np.array(output).reshape(-1, 3)  # x, y, confidence
    keypoints[:, 0] *= frame_shape[1]
    keypoints[:, 1] *= frame_shape[0]
    return keypoints

def draw_keypoints(frame, keypoints, threshold=0.5):
    """Draw keypoints on the frame."""
    for point in keypoints:
        x, y, confidence = point
        if confidence > threshold:
            cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)

# Connect to device and start the pipeline
with dai.Device(pipeline) as device:
    # Output queues for streams
    q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    q_det = device.getOutputQueue(name="det", maxSize=4, blocking=False)
    q_keypoints = device.getOutputQueue(name="keypoints", maxSize=4, blocking=False)

    while True:
        in_rgb = q_rgb.get()
        frame = in_rgb.getCvFrame()

        in_det = q_det.get()
        detections = in_det.detections

        # Draw bounding boxes for all detected objects
        frame = draw_bounding_boxes(frame, detections)

        # Process keypoints for each detected object
        in_keypoints = q_keypoints.get()
        keypoints_output = in_keypoints.getFirstLayerFp16()
        keypoints = decode_keypoints(keypoints_output, frame.shape)

        # Draw keypoints on the original image
        draw_keypoints(frame, keypoints)

        # Show the frame with bounding boxes and keypoints
        cv2.imshow("YOLO + Keypoint Detection", frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()
