import argparse
from pathlib import Path
from utils.process import launch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', nargs='?', type=str, default='data')
    args = parser.parse_args()

    path = Path(args.path)

    launch(path)

if __name__ == '__main__':
    main()
