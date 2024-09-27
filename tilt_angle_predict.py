from utils import *

pipeline = create_pipeline()

# Connect to device and start the pipeline
with dai.Device(pipeline) as device:
    # Output queues for the streams
    q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    q_nn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

    # Processing loop
    while True:
        in_rgb = q_rgb.get()
        in_nn = q_nn.get()

        # Get RGB image
        frame = in_rgb.getCvFrame()

        # Get NN output and reshape to keypoints
        keypoints = np.array(in_nn.getFirstLayerFp16()).reshape(-1, 2)

        # Normalize keypoints coordinates to frame size
        keypoints[:, 0] = keypoints[:, 0] * frame.shape[1]  # Scale X
        keypoints[:, 1] = keypoints[:, 1] * frame.shape[0]  # Scale Y

        # Calculate the tilt angle using all keypoints (PCA method)
        if len(keypoints) >= 2:
            tilt_angle, centroid = calculate_pca_tilt_angle(keypoints)

            if tilt_angle is not None:
                # Display the tilt angle on the frame
                cv2.putText(frame, f"Tilt: {tilt_angle:.2f} degrees", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                # Visualize keypoints and centroid on the frame
                for idx, point in enumerate(keypoints):
                    x, y = int(point[0]), int(point[1])
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                    cv2.putText(frame, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # Visualize centroid
                cv2.circle(frame, (int(centroid[0]), int(centroid[1])), 8, (255, 0, 0), -1)

                # Draw principal axis
                p1 = (int(centroid[0]), int(centroid[1]))
                p2 = (int(centroid[0] + 100 * np.cos(np.radians(tilt_angle))), 
                      int(centroid[1] + 100 * np.sin(np.radians(tilt_angle))))
                cv2.line(frame, p1, p2, (0, 0, 255), 2)

        # Show the frame
        cv2.imshow("Keypoints with Tilt (PCA)", frame)

        # Break on 'q' key press
        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()