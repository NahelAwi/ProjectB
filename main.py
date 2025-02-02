from multiprocessing import Process, Queue
import time
import random
import argparse

from fastsam import calc_angle, init_fastsam
from hand_control import hand_control_thread
from utils import *

class DummyQ:
    def put(self, v):
        return
    def get(self):
        return None

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Example script with optional flags.")
    parser.add_argument("--no_hand", action="store_true")
    args = parser.parse_args()

    if args.no_hand:
        angle_queue = DummyQ()
        grip_queue = DummyQ()
        hand_is_open_queue = DummyQ()
    else:
        angle_queue = Queue(maxsize=1)
        grip_queue = Queue(maxsize=1)
        hand_is_open_queue = Queue(maxsize=1)
        p2 = Process(target=hand_control_thread, args=(angle_queue,grip_queue,hand_is_open_queue,))
        p2.start()

    init_fastsam()
    min_depth = 999
    filtered_depth = 0

    pipeline = create_RGB_Depth_pipeline()

    with dai.Device(pipeline) as devicex:

        rgb_queue = devicex.getOutputQueue(name="rgb", maxSize=15, blocking=False)
        depth_queue = devicex.getOutputQueue(name="depth", maxSize=15, blocking=False)

        try:
            old_angle = 0
            while True:
                depth_frame = depth_queue.get().getFrame()  # Get the full depth frame


                depth_frame = cv2.resize(depth_frame, (width, height), interpolation=cv2.INTER_NEAREST)
                depth_frame = cv2.medianBlur(depth_frame, 5)

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

                filtered_depth = Calculate_Depth(depth_frame, filtered_depth)

                min_depth = min(min_depth, filtered_depth)

                print(f"Center depth: {filtered_depth} mm")

                angle, processed_frame = calc_angle(filtered_depth, frame_rgb, old_angle)

                if angle:
                    if abs(angle-old_angle) <= 10:#margin
                        angle = 0
                        if filtered_depth < 140:
                            grip_queue.put(1)
                        hand_is_open_queue.get() # block until hand is open
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

