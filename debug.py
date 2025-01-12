from try_fastsam import *

class DummyQ:
    def __init__(self):
        pass

    def put(self, e):
        pass

queue = DummyQ()

calc_angle_thread(queue)
