# Object Reidentification

This project aims to perform object reidentification by comparing frames based on histogram similarity and Intersection over Union (IoU) of bounding boxes.

![Example](resources/example.gif)

Due to low framerate of input video, IoU method is not the best possible solution, if tested with videos with higher FPS, results should improve.

## Project Description

This project aims to create a system for tracking pedestrians using probabilistic graphical models. The system is designed to determine the location of pedestrians in consecutive frames of a camera image by assigning bounding boxes (BBoxes) to each person. The project assumptions are as follows:

- For each frame of the image, the coordinates of the BBoxes are provided, but it is not specified which BBox belongs to which pedestrian.
- The task is to determine which BBoxes from the previous frame correspond to the BBoxes in the current frame.
- For each frame, a sequence of N integers is outputted, separated by spaces and terminated by a newline character (`\n`), where N is the number of BBoxes in the current frame. For each BBox, the number represents the index of the corresponding BBox from the previous frame, or -1 if, for example, the pedestrian appears on the screen for the first time.

Example output for 2 frames with 2 and 3 BBoxes:

    1 2
    0 -1 1
    

## Usage

The project utilizes probabilistic graphical models, histogram similarity, and Intersection over Union (IoU) to compare frames and determine the correspondence between BBoxes in consecutive frames. The `Frame` class represents a frame and stores information about the frame's filename, number of BBoxes, and their coordinates. The `FrameList` class extends the list class and provides additional functionality for storing `Frame` objects.

The `compare_frames` function takes two `Frame` objects and a `FactorGraph` as input. It compares similarities between frames based on histogram similarity and IoU and adds factors to the `FactorGraph` accordingly. The Belief Propagation algorithm is then used to perform inference on the graph and obtain the results.

To use the project, you need to load the frame data, iterate over the frames, and call the `compare_frames` function to compare each frame with the previous frame. The results are then processed and outputted according to the specified format.

## Matrix M

The matrix `M` in the probabilistic graphical model plays a crucial role in capturing the pairwise relationships or similarities between variables (bounding boxes) within the model. It serves as a factor table or factor potential, enabling the definition of factors and their associated potentials in the graphical model.

In the provided code, `M` is a square matrix with dimensions `(len(frame_prev.hists) + 1, len(frame_prev.hists) + 1)`. The additional `+1` is introduced to accommodate a special case when there are no bounding boxes in the previous frame.

The elements of `M` are initially set to `1`, representing a default similarity or compatibility between any pair of bounding boxes. However, the diagonal elements are set to `0` to indicate that there is no similarity between a bounding box and itself.

The matrix `M` allows the incorporation of pairwise similarities or dissimilarities between bounding boxes into the factor graph. By adjusting the values in `M`, you can control the influence or weight of different factors in the probabilistic model. This reflects the relative importance of histogram similarity and other similarity measures in determining the correspondence between bounding boxes across frames.

The defined factors and their associated potentials, based on `M`, facilitate the inference algorithm (such as Belief Propagation) to compute the optimal configuration of variables (bounding boxes) in the graph. This optimization process yields the most likely associations between bounding boxes from the current frame and those from the previous frame.

## Classes

### Frame

This class represents a frame and stores information about the frame's filename, number of bounding boxes, and their coordinates.

#### Methods

- `__init__(self, dir_name: str, filename: str, n: int, bboxes: list)`: Initializes the Frame object with the provided parameters.
- `img(self)`: Returns the image corresponding to the frame.
- `__str__(self)`: Returns a string representation of the Frame object.
- `histograms(self)`: Computes histogram lists from the center part of the bounding boxes in the frame.

### FrameList

This class extends the built-in list class and provides additional functionality for storing Frame objects.

#### Methods

- `append(self, frame)`: Appends a Frame object to the FrameList. Only instances of the Frame class can be added.

## Functions

- `IoU(boxA, boxB)`: This function computes the Intersection over Union (IoU) of two bounding boxes.
- `distance(boxA, boxB, imsize)`: This function calculates the distance between the centers of two bounding boxes normalized by the image size.
- `compare_frames(frame1: Frame, frame2: Frame, G: FactorGraph)`: This function compares similarities between frames based on histogram similarity and IoU. It takes two Frame objects and a FactorGraph as input.

## Additional Files

- `load_data.py`: This file contains a function `load_data(path)` that loads frame data from a specified path.

- `score.py`: Computes score value (number of all correctly identified bboxes / number of bboxes) * 100%. Groud truth file: `data/qt.txt`.
Feel free to modify and extend the code to suit your specific needs.

## License

This project is licensed under the MIT License.
