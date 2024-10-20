import cv2
import depthai as dai

FPS = 15

# Create pipeline
pipeline = dai.Pipeline()

# Create RGB camera node
rgb_cam = pipeline.create(dai.node.ColorCamera)
rgb_cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
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


# Set up stereo depth
mono_left = pipeline.create(dai.node.MonoCamera)
mono_right = pipeline.create(dai.node.MonoCamera)

mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)

mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)

stereo = pipeline.create(dai.node.StereoDepth)
stereo.initialConfig.setConfidenceThreshold(180)
stereo.setRectifyEdgeFillColor(0)  # Black, to better see the cutout
# stereo.setExtendedDisparity(True)
# stereo.setSubpixel(True)  # Enable subpixel precision for finer depth
# stereo.initialConfig.setMedianFilter(dai.StereoDepthProperties.MedianFilter.KERNEL_7x7)
# stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
# stereo.setLeftRightCheck(True)  # Helps resolve depth for near objects

# Link mono cameras to stereo node
mono_left.out.link(stereo.left)
mono_right.out.link(stereo.right)

# Create depth output
xout_depth = pipeline.create(dai.node.XLinkOut)
xout_depth.setStreamName("depth")
stereo.depth.link(xout_depth.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    depth_queue = device.getOutputQueue(name="depth", maxSize=15, blocking=False)
    rgb_queue = device.getOutputQueue(name="rgb", maxSize=15, blocking=False)
    
    while True:

        in_rgb = rgb_queue.get()

        frame = None
        if in_rgb is not None:
            frame = in_rgb.getCvFrame()
        
        cv2.imshow("frame", frame)
        
        depth_frame = depth_queue.get().getFrame()  # Get the full depth frame

        # Retrieve only the center depth value
        height, width = depth_frame.shape
        center_depth = depth_frame[height // 2, width // 2]

        center_x = width // 2
        center_y = height // 2


        kernel_size = 3  # Adjust the size of the window
        start_x = center_x - kernel_size // 2
        start_y = center_y - kernel_size // 2

        # Extract a small region around the center
        center_region = depth_frame[start_y:start_y + kernel_size, start_x:start_x + kernel_size]

        # Calculate the average depth in that region
        center_depth_avg = center_region.mean()

        print(f"Center depth: {center_depth_avg} mm")

        cv2.imshow("Fitted Ellipse", depth_frame)

        if cv2.waitKey(1) == ord('q'):
            break
