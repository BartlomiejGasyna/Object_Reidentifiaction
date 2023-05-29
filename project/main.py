import os
import cv2
from utils.load_data import load_data


# dir = 'data/frames/'

# imgs = sorted([dir + filename for filename in os.listdir(dir)])


frames = load_data()

for idx, frame in enumerate(frames):
    img_curr = frame.img()
    # print(img_curr)
    if idx == 0:
        continue
    
    # show bboxes
    for bbox in frame.bboxes:
        x, y, w, h = list(map(int, bbox))
        cv2.rectangle(img_curr, (x, y), (x+w, y+h), (0, 255, 0) )
    cv2.imshow('frame', img_curr)

    key = cv2.waitKey(100)
    if key == ord('q'):
        cv2.destroyAllWindows
        break

    ###############
    #  TODO: Process 
    ###############

    ###############
    img_prev = img_curr
    frame_prev = frame