from multiprocessing import Process, Queue
import time
import random

from fastsam import calc_angle, init_fastsam
from hand_control import hand_control_thread
from utils import *

if __name__ == "__main__":

    angle_queue = Queue()
    p2 = Process(target=hand_control_thread, args=(angle_queue,))
    p2.start()

    init_fastsam()

    pipeline = create_RGB_Depth_pipeline()

    with dai.Device(pipeline) as devicex:

        rgb_queue = devicex.getOutputQueue(name="rgb", maxSize=15, blocking=False)
        depth_queue = devicex.getOutputQueue(name="depth", maxSize=15, blocking=False)

        try:
            old_angle = 0
            while True:
                depth_frame = depth_queue.get().getFrame()  # Get the full depth frame

                ret = rgb_queue.get()

                # # Get left frame
                # left_frame = left_queue.get().getCvFrame()
                # # Get right frame 
                # right_frame = right_queue.get().getCvFrame()

                # imOut = np.hstack((left_frame, right_frame))
                # imOut = np.uint8(left_frame/2 + right_frame/2)

                # Display output image
                # cv2.imshow("Stereo Pair", imOut)
                
                frame = None
                if ret:
                    frame = ret.getCvFrame()
                else:
                    continue

                frame_rgb = frame#cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                processed_frame = frame_rgb
                angle = None

                depth = Calculate_Depth(depth_frame)

                print(f"Center depth: {depth} mm")

                angle, processed_frame = calc_angle(depth, frame_rgb, old_angle)

                if angle:
                    if abs(angle-old_angle) <= 10:#margin
                        angle = 0
                    else:
                        tmp = angle
                        angle -= old_angle
                        old_angle = tmp
                    angle_queue.put(angle)
                
                
                cv2.imshow("Real-Time Segmentation", processed_frame)
                # cv2.imshow("Depth", depth_frame)
                # cv2.imshow("left", left_frame)
                # cv2.imshow("right", right_frame)
                


                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            cv2.destroyAllWindows()
            p2.join()

