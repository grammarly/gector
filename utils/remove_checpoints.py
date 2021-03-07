import argparse
import os
import glob

def main(args):
    path = args.source
    if path[-1] != '/':
        path += '/'
    all_checkpoints = glob.glob(path+"*.th")
    count_delated = 0
    for checkpoint in all_checkpoints:
        if checkpoint != path+'best.th':
            os.remove(checkpoint)
            count_delated += 1
    print("Deleted files:", count_delated)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source',
                        help='Path to the source folder',
                        required=True)

    args = parser.parse_args()
    main(args)
