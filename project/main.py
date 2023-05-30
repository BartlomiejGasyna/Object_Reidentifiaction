# import os
import cv2
from utils.load_data import load_data

# import torch
# import numpy as np
from utils.dataTyes import IoU, compute_intersection

# dir = 'data/frames/'

# imgs = sorted([dir + filename for filename in os.listdir(dir)])


frames = load_data()


for idx, frame in enumerate(frames):
    img_curr = frame.img()
    if idx == 0:
        frame_prev = frame

    # show bboxes
    for bbox in frame.bboxes:
        x, y, w, h = list(map(int, bbox))
        cv2.rectangle(img_curr, (x, y), (x+w, y+h), (0, 255, 0) )

    for bbox in frame_prev.bboxes:
        x, y, w, h = list(map(int, bbox))
        cv2.rectangle(img_curr, (x, y), (x+w, y+h), (0, 255, 0) )
    
    # iou_, intersection_boxes = compute_intersection(frame.bboxes, frame_prev.bboxes)
    # print('ious ')
    # for iou, box in zip(iou_, intersection_boxes):
    #     cv2.rectangle(img_curr, (int(box[0]), int(box[1])), (int(box[0]+box[2]), int(box[1]+box[3])), (0, 0, 255))
    #     print(iou)

    frame.histograms()



    cv2.imshow('frame', img_curr)
    # cv2.imshow('prev', frame_prev.img())

    key = cv2.waitKey()
    if key == ord('q'):
        cv2.destroyAllWindows
        break

    ###############
    #  TODO: Process 
    ###############
    # process(frame, frame_prev)

    ###############
    frame_prev = frame