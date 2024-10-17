import cv2
from source.utils import *

# Create pipeline
pipeline = dai.Pipeline()

# Create RGB camera node
rgb_cam = pipeline.create(dai.node.ColorCamera)
rgb_cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
# rgb_cam = pipeline.createColorCamera()
# rgb_cam.setBoardSocket(dai.CameraBoardSocket.RGB)
rgb_cam.setInterleaved(False)
rgb_cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
rgb_cam.setPreviewSize(640,480)
rgb_cam.setPreviewKeepAspectRatio(False)
rgb_cam.setFps(15)

# Create outputs
rgb_output = pipeline.create(dai.node.XLinkOut)
rgb_output.setStreamName("rgb")

# Link nodes
rgb_cam.preview.link(rgb_output.input)

with dai.Device(pipeline) as device:
    rgb_queue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

    while True:

        in_rgb = rgb_queue.get()

        frame = None
        if in_rgb is not None:
            frame = in_rgb.getCvFrame()
        
        # Taking only middle of the image
        height, width = frame.shape[:2]
        mid_height = height // 2
        mid_width = width // 2
        # Define the ROI, for example 200x200 pixels around the center
        roi_size = 400
        middle_section = frame[mid_height - roi_size//2 : mid_height + roi_size//2,
                                mid_width - roi_size//2 : mid_width + roi_size//2]

        # Convert to grayscale
        gray = cv2.cvtColor(middle_section, cv2.COLOR_BGR2GRAY)

        # Apply GaussianBlur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)

        # Show the detected edges
        cv2.imshow("Edges", edges)
        cv2.imshow("blurred", blurred)
        # Find contours in the edge-detected image
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_contour_area = 200

        for contour in contours:
            if len(contour) >= 5 and cv2.contourArea(contour) > min_contour_area:  # Need at least 5 points to fit an ellipse
                ellipse = cv2.fitEllipse(contour)
                
                # Draw the ellipse
                cv2.ellipse(blurred, ellipse, (0, 255, 0), 2)

                # Get the tilt angle of the ellipse
                angle = ellipse[2]
                print(f"Tilt (angle): {angle} degrees")

                cv2.imshow("Fitted Ellipse", blurred)

        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()
