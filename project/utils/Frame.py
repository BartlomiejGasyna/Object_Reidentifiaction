import cv2
# import matplotlib.pyplot as plt
from pgmpy.models import FactorGraph
from pgmpy.factors.discrete import DiscreteFactor
import math

class Frame():
    ''' Frame class stores information about frame filename, number of bboxes, bbox coordinates'''
    def __init__(self, dir_name: str, filename: str, n: int, bboxes: list):
        self.dir_name = dir_name
        self.filename = filename
        self.n = n
        self.bboxes = bboxes
        
        self.hists = []
    def img(self):
        return cv2.imread(self.dir_name + self.filename)
    
    def __str__(self):
        return f"Filename: {self.filename}\nNumber of Boxes: {self.n}\nBounding Boxes: {self.bboxes}"

    def histograms(self):
        ''' compute histogram list from center part of bounding box '''

        hists = []
        for x, y, w, h in self.bboxes:
            # x, y, w, h = list(map(int, box))
            margin_x = 0.15 * w
            margin_y = 0.15 * h

            # Calculate the coordinates of the center region
            center_x = int(x + margin_x)
            center_y = int(y + margin_y)
            center_w = int(w - 2 * margin_x)
            center_h = int(h - 2 * margin_y)

            # Extract the center region from the image
            center_roi = self.img().copy()[center_y:center_y+center_h, center_x:center_x+center_w]

            # Calculate the histogram of the center region
            center_roi = cv2.cvtColor(center_roi, cv2.COLOR_BGR2HSV)

            h, s, _ = cv2.split(center_roi)
            center_roi_selected_channels = cv2.merge([h, s])

            hist = cv2.calcHist([center_roi_selected_channels], [0], None, [256], [0, 256])



            # Normalize the histogram
            hist /= hist.sum()


            hists.append(hist)
            

            # plt.plot(hist)
            # plt.show()
            # print('histogram: ', hist)
        self.hists = hists
        return hists

class FrameList(list):
    def append(self, frame):
        if isinstance(frame, Frame):
            super().append(frame)
        else:
            raise TypeError("Only instances of Frame class can be added to FrameList")


def IoU(boxA, boxB):
    '''
    Computes IoU's of two bboxes
    '''
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    
    iou = 0
    if xB > xA and yB > yA:
        intersection_area = (xB - xA) * (yB - yA)
        boxA_area = boxA[2] * boxA[3]
        boxB_area = boxB[2] * boxB[3]
        iou = intersection_area / float(boxA_area + boxB_area - intersection_area)
    return iou

def distance(boxA, boxB, imsize):
    xA = boxA[0] + boxA[2] / 2
    yA = boxA[1] + boxA[3] / 2

    xB = boxB[0] + boxB[2] / 2
    yB = boxB[1] + boxB[3] / 2

    distance = math.sqrt((xB - xA) ** 2 + (yB - yA) ** 2)
    dist_norm = 1.0 - (distance / imsize)
    # print('dist: ', dist_norm)
    return(dist_norm)

def compare_frames(frame1: Frame, frame2: Frame, G: FactorGraph):
    ''' 
    Ths function compares similarities between frames by computing weighted similarities based on:
    - Histogram similarity - weight - 0.75
    - IoU - weight - 0.25
    '''
    histograms_current = frame1.histograms()
    histograms_previous = frame2.histograms()


    for idx, current_hist in enumerate(histograms_current):
        similarity = []
        for id2, prev_hist in enumerate(histograms_previous):
            hist_sim = cv2.compareHist(current_hist, prev_hist, cv2.HISTCMP_CORREL)
            iou_sim = IoU(frame1.bboxes[idx], frame2.bboxes[id2])

            # distance is unsued - IoU gives better results
            # w, h = frame1.img().shape[:2]
            # dist = distance(frame1.bboxes[idx], frame2.bboxes[id2], math.sqrt(w**2 + h**2))

            similarity.append((hist_sim * 0.75 + iou_sim * 0.25) )
    
    # print('similarity: ', similarity) # DEBUG only

    # Adding factors to graph
        tmp = DiscreteFactor([str(idx)], [frame2.n + 1], [[0.395] + similarity])
        G.add_factors(tmp)
        G.add_edge(str(idx), tmp)

    return similarity