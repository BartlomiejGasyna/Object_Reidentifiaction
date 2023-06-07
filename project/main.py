import cv2
from utils.load_data import load_data

from pgmpy.models import FactorGraph
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import BeliefPropagation
from itertools import combinations
import numpy as np
from utils.dataTyes import compare_histograms
from collections import OrderedDict

# dir = 'data/frames/'

# imgs = sorted([dir + filename for filename in os.listdir(dir)])


frames = load_data()


for idx, frame in enumerate(frames):
    img_curr = frame.img()
    if idx == 0:
        frame_prev = frame
        print(' '.join(['-1'] * frame.n))
        continue

    # show bboxes
    # for bbox in frame.bboxes:
    #     x, y, w, h = list(map(int, bbox))
    #     cv2.rectangle(img_curr, (x, y), (x+w, y+h), (0, 0, 100) )

    # for bbox in frame_prev.bboxes:
    #     x, y, w, h = list(map(int, bbox))
    #     cv2.rectangle(img_curr, (x, y), (x+w, y+h), (0, 0, 255) )
    
    # iou_, intersection_boxes = compute_intersection(frame.bboxes, frame_prev.bboxes)
    # print('ious ')
    # for iou, box in zip(iou_, intersection_boxes):
    #     cv2.rectangle(img_curr, (int(box[0]), int(box[1])), (int(box[0]+box[2]), int(box[1]+box[3])), (0, 0, 255))
    #     print(iou)

    ###############
    #  TODO: Process 
    ###############
    # process(frame, frame_prev)



    G = FactorGraph()

    # no bboxes in frame
    if frame.n == 0:
        frame_prev = frame
        print(' '.join(['-1'] * frame.n))
        continue
    
    # if no bboxes in previous frame, for each bbox in current frame, print '-1'
    if frame_prev.n == 0:
        print(' '.join(['-1'] * frame.n))
        continue
    # number of bboxes defines number of nodes
    for i in range(frame.n):
        G.add_node(str(i))

    # IOU = compute_intersection(frame.bboxes, frame_prev.bboxes)
    compare_histograms(frame, frame_prev, G)

    ##############
    # M =
    # [[1. 1. 1.]
    # [1. 0. 1.]
    # [1. 1. 0.]
    # [1. 1. 1.]
    # [1. 1. 1.]]
    ##############

    # M = np.ones((frame_prev.n + 1, frame.n + 1), dtype=float)
    # np.fill_diagonal(M, 0)
    # M[0,0] = 1

    M = np.ones((len(frame_prev.hists)+1,len(frame_prev.hists)+1))

    for i in range(len(frame_prev.hists)+1):
        for j in range(len(frame_prev.hists)+1):
            if i != 0:
                if i == j:
                    M[i][j] = 0

    # current_histrogram1 = 0
    # current_histrogram2 = 1
    for current_histrogram1, current_histrogram2 in combinations(range(int(frame.n)), 2):
        # print('loop')
        # print([str(current_histrogram1), str(current_histrogram2)])
        # print()
        # print([frame_prev.n + 1, frame_prev.n + 1])
        # print()
        # print(M)
        tmp = DiscreteFactor([str(current_histrogram1), str(current_histrogram2)], [frame_prev.n + 1,
                                                                                frame_prev.n + 1],
                                                                                M)
        G.add_factors(tmp)
        G.add_edge(str(current_histrogram1), tmp)
        G.add_edge(str(current_histrogram2), tmp)


    # Belief propagation
    BP = BeliefPropagation(G)
    BP.calibrate()

    pre_results = (BP.map_query(G.get_variable_nodes(), show_progress=False))
    pre_results2 = OrderedDict(sorted(pre_results.items()))

    result = list(pre_results2.values())
    final = []

    colors = [(255,0,0), (0, 255,0), (0,0,255), (255,255,0,), (0,255,255), (255,255,255)]

    
    for idx, res in enumerate(result):
        value = res - 1
        final.append(value)

        cv2.putText(img_curr, str(value), (int(frame.bboxes[idx][0] + frame.bboxes[idx][2]/2), int(frame.bboxes[idx][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255,0))

        x, y, w, h = list(map(int, frame.bboxes[idx]))
        cv2.rectangle(img_curr, (x, y), (x+w, y+h), colors[idx] )

    print(*final, sep = ' ')
    ###############
    frame_prev = frame

    # cv2.imshow('frame', img_curr)
    # key = cv2.waitKey(100)
    # if key == ord('q'):
    #     cv2.destroyAllWindows
    #     break