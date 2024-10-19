import cv2
from source.utils import *

FPS = 15

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
rgb_cam.setFps(FPS)

# Create outputs
rgb_output = pipeline.create(dai.node.XLinkOut)
rgb_output.setStreamName("rgb")

# Link nodes
rgb_cam.preview.link(rgb_output.input)

# # Create depth camera node
# mono_left = pipeline.create(dai.node.MonoCamera)
# mono_right = pipeline.create(dai.node.MonoCamera)

# mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
# mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)

# mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
# mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)

# # Create stereo depth node
# stereo = pipeline.create(dai.node.StereoDepth)
# # stereo.setOutputDepth(True)
# # stereo.setConfidenceThreshold(200)
# # stereo.setRectifyEdgeFillColor(0)  # Black, to better see the cutout
# # stereo.setExtendedDisparity(True)
# stereo.setSubpixel(True)  # Enable subpixel precision for finer depth
# # stereo.setMedianFilter(dai.StereoDepthProperties.MedianFilter.KERNEL_7x7)
# # stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
# stereo.setLeftRightCheck(True)  # Helps resolve depth for near objects



# # Link mono cameras to the stereo depth
# mono_left.out.link(stereo.left)
# mono_right.out.link(stereo.right)

# # Create output streams for depth
# xout_depth = pipeline.create(dai.node.XLinkOut)
# xout_depth.setStreamName("depth")
# stereo.depth.link(xout_depth.input)


with dai.Device(pipeline) as device:
    rgb_queue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    depth_queue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

    i = 0
    angle = 0

    while True:

        # # Get depth frame
        # depth_frame = depth_queue.get().getFrame()
        
        # # Normalize depth frame for visualization
        # depth_frame_vis = (depth_frame * (255 / stereo.getMaxDisparity())).astype('uint8')

        # # Get the depth value at the center of the image
        # depth_at_center = depth_frame[depth_frame.shape[0] // 2, depth_frame.shape[1] // 2]
        # # print(f"Depth at the center: {depth_at_center} mm")

        # cv2.imshow("Depth", depth_frame_vis)

        # Get RGB frame
        in_rgb = rgb_queue.get()

        frame = None
        if in_rgb is not None:
            frame = in_rgb.getCvFrame()
        
        # Taking only middle of the image
        height, width = frame.shape[:2]
        mid_height = height // 2
        mid_width = width // 2
        # Define the ROI, for example 200x200 pixels around the center
        roi_size = 300
        middle_section = frame[mid_height - roi_size//2 : mid_height + roi_size//2,
                                mid_width - roi_size//2 : mid_width + roi_size//2]

        # Convert to grayscale
        gray = cv2.cvtColor(middle_section, cv2.COLOR_BGR2GRAY)

        # Apply GaussianBlur to reduce noise
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)

        # Show the detected edges
        # cv2.imshow("Edges", edges)

        # Show the blurred
        # cv2.imshow("blurred", blurred)

        kernel = np.ones((3, 3), np.uint8)

        # Dilation to strengthen the edges
        edges_dilated = cv2.dilate(edges, kernel, iterations=3)
        # cv2.imshow("Edges_dialated", edges_dilated)

        # Erosion to remove small unwanted edges
        edges_eroded = cv2.erode(edges_dilated, kernel, iterations=1)
        cv2.imshow("Edges_eroded", edges_eroded)

        # Find contours in the edge-detected image
        contours, _ = cv2.findContours(edges_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_contour_area = 200

        if(len(contours) > 0):
        # for contour in contours:
            largest_contour = max(contours, key=cv2.contourArea)

            if len(largest_contour) >= 5 and cv2.contourArea(largest_contour) > min_contour_area:  # Need at least 5 points to fit an ellipse
                ellipse = cv2.fitEllipse(largest_contour)
                
                # Draw the ellipse
                cv2.ellipse(frame[mid_height - roi_size//2 : mid_height + roi_size//2,
                                mid_width - roi_size//2 : mid_width + roi_size//2], ellipse, (0, 255, 0), 2)

                # Get the tilt angle of the ellipse
                angle += ellipse[2]
                i += 1
                if(i == FPS):
                    angle = angle / FPS
                    print(f"Tilt (angle): {angle} degrees")
                    # print(f"Depth at the center: {depth_at_center} mm")
                    angle = 0
                    i = 0

        cv2.imshow("Fitted Ellipse", frame)

        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()
