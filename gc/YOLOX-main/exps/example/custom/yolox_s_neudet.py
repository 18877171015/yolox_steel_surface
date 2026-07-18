from yolox.exp import Exp as BaseExp

class Exp(BaseExp):
    def __init__(self):
        super().__init__()
        self.num_classes = 6
        self.depth = 0.33
        self.width = 0.50
        self.test_size = (640, 640)
        self.test_conf = 0.25
        self.nmsthre = 0.45