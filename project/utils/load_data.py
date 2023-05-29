from utils.dataTyes import Frame, FrameList

def load_data(frames_dir: str = 'data/frames/', dir_name: str = 'data/bboxes.txt'):
    frames = FrameList()

    # Open the template file for reading
    with open(dir_name, 'r') as file:
        lines = file.readlines()

    # Iterate over the lines in the file
    line_count = len(lines)
    index = 0
    while index < line_count:
        filename = lines[index].strip()
        n = int(lines[index + 1].strip())
        bboxes = []

        # Iterate over the bounding box lines
        for i in range(n):
            bbox_line = lines[index + 2 + i].strip().split(' ')
            bbox = [float(value) for value in bbox_line]
            bboxes.append(bbox)

        # Create an instance of the Frame class
        frame = Frame(frames_dir, filename, n, bboxes)
        frames.append(frame)

        # Move to the next frame
        index += n + 2
    
    return frames


if __name__ == '__main__':
    
    frames = load_data()
    # Print the frames
    for frame in frames:
        print(frame)
        print()