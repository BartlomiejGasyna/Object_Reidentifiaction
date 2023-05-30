import cv2
import matplotlib.pyplot as plt
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

            hist = cv2.calcHist([center_roi], [0], None, [256], [0, 256])

                # Normalize the histogram
            # hist /= center_w * center_h
            hist /= hist.sum()


            hists.append(hist)

            # plt.plot(hist)
            # plt.show()
            # print('histogram: ', hist)
            
        return hists

class FrameList(list):
    def append(self, frame):
        if isinstance(frame, Frame):
            super().append(frame)
        else:
            raise TypeError("Only instances of Frame class can be added to FrameList")


def IoU(boxA, boxB):
        


	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
	yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
        
	# # compute the area of intersection rectangle
	# interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# # compute the area of both the prediction and ground-truth
	# # rectangles
	# boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	# boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# # compute the intersection over union by taking the intersection
	# # area and dividing it by the sum of prediction + ground-truth
	# # areas - the interesection area
	# iou = interArea / float(boxAArea + boxBArea - interArea)
	# # return the intersection over union value
	return (int(xA), int(yA)), (int(xB), int(yB))

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

