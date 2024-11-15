from utils import *

pipeline = create_pipeline()

# Connect to device and start the pipeline
with dai.Device(pipeline) as device:
    # Output queues for the streams
    q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    q_nn = device.getOutputQueue(name="detectionNetwork", maxSize=4, blocking=False)

    # Processing loop
    while True:
        in_rgb = q_rgb.get()
        in_nn = q_nn.get()

        # Get RGB image
        # frame = in_rgb.getCvFrame()

        # # Get NN output
        # keypoints = np.array(in_nn.getFirstLayerFp16()).reshape(-1, 2)

        # # Visualize keypoints on the frame
        # for idx, point in enumerate(keypoints):
        #     x, y = int(point[0] * frame.shape[1]), int(point[1] * frame.shape[0])
        #     cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        #     cv2.putText(frame, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # # Show the frame
        # cv2.imshow("Keypoints", frame)
        
        if in_rgb is not None:
            frame = in_rgb.getCvFrame()

            # cv2.putText(frame, "NN fps: {:.2f}".format(counter / (time.monotonic() - startTime)),
            #             (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color2)

        if in_nn is not None:
            detections = in_nn.detections

        if frame is not None:
            # print("frame.shape = ", frame.shape)
            displayFrame("rgb", frame, detections)



        # Break on 'q' key press
        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()