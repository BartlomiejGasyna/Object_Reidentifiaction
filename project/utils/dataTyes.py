import cv2

class Frame():
    ''' Frame class stores information about frame filename, number of bboxes, bbox coordinates'''
    def __init__(self, dir_name: str, filename: str, n: int, bboxes: list):
        self.dir_name = dir_name
        self.filename = filename
        self.n = n
        self.bboxes = bboxes
    
    def img(self):
        return cv2.imread(self.dir_name + self.filename)
    
    def __str__(self):
        return f"Filename: {self.filename}\nNumber of Boxes: {self.n}\nBounding Boxes: {self.bboxes}"

class FrameList(list):
    def append(self, frame):
        if isinstance(frame, Frame):
            super().append(frame)
        else:
            raise TypeError("Only instances of Frame class can be added to FrameList")