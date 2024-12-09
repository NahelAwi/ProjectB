from multiprocessing import Process, Queue
import time
import random

from try_fastsam import calc_angle_thread
from hand_control import hand_control_thread

if __name__ == "__main__":
    # Create a shared queue for communication
    angle_queue = Queue(maxsize=1)

    # Create the two processes
    p1 = Process(target=calc_angle_thread, args=(angle_queue,))
    p2 = Process(target=hand_control_thread, args=(angle_queue,))

    # Start the processes
    p1.start()
    p2.start()

    # Wait for the processes to finish (this won't happen in an infinite loop)
    p1.join()
    p2.join()