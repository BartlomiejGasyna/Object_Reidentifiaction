import cv2
import matplotlib.pyplot as plt

from pgmpy.models import FactorGraph
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import BeliefPropagation

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
            # hist /= center_w * center_h
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



def compute_intersection(bboxA, bboxB):
    iou_ = []
    boxes = []

    for boxA in bboxA:
        for boxB in bboxB:
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
            yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
            
            if xB > xA and yB > yA:
                intersection_area = (xB - xA) * (yB - yA)
                boxA_area = boxA[2] * boxA[3]
                boxB_area = boxB[2] * boxB[3]
                iou = intersection_area / float(boxA_area + boxB_area - intersection_area)
                iou_.append(iou)
                boxes.append([xA, yA, xB - xA, yB - yA])
                
    return iou_, boxes

# TODO: histograms can be optimized by saving them
def compare_histograms(frame1: Frame, frame2: Frame, G: FactorGraph):
    histograms_current = frame1.histograms()
    histograms_previous = frame2.histograms()


    hists_similarity = []
    for idx, current_hist in enumerate(histograms_current):
        hist_similarity = []
        for prev_hist in histograms_previous:
            sim = cv2.compareHist(current_hist, prev_hist, cv2.HISTCMP_CORREL)
            # hist_similarity.append(cv2.compareHist(current_hist, prev_hist, cv2.HISTCMP_CORREL))
            hist_similarity.append(sim)
    
    # print('similarity: ', hist_similarity) # DEBUG only

    # Adding factors to graph
        # print([len(histograms_previous) + 1])
        # print([[0.29] + hist_similarity])
        tmp = DiscreteFactor([str(idx)], [frame2.n + 1], [[0.29] + hist_similarity])
        G.add_factors(tmp)
        G.add_edge(str(idx), tmp)

    return hist_similarity

# def compare_histograms(frame1: Frame, frame2: Frame, G: FactorGraph):
#     hist1 = frame1.histograms()
#     hist2 = frame2.histograms()

# # In this function, histogram from both previous and current frame are compared against each other. Comparison value is
# # a mean value for comparing considered channels. For comparing I used ready Bhattacharyya method.

#     sum = 0
#     for i in range(len(hist1)):
#         mean_j = []
#         sum = 0
#         for j in range(len(hist2)):
#             comparison1 = cv2.compareHist(hist1[i][0],hist2[j][0],cv2.HISTCMP_BHATTACHARYYA)
#             comparison2 = cv2.compareHist(hist1[i][1],hist2[j][1],cv2.HISTCMP_BHATTACHARYYA)
#             # comparison3 = cv2.compareHist(hist1[i][2],hist2[j][2],cv2.HISTCMP_BHATTACHARYYA)
#             comparison = (comparison1 + comparison2) / 2
#             comparison = 1 - comparison
#             mean_j.append(comparison)

# # Adding DiscreteFactor to the FactorGraph()
#         tmp = DiscreteFactor([str(i)], [len(hist2) + 1], [[0.29] + mean_j])
#         G.add_factors(tmp)
#         G.add_edge(str(i),tmp)

#     return mean_j