import argparse
import re

from helpers import write_lines


def filter_line(line):
    if "-LRB-" in line and "-RRB-" in line:
        rep = re.sub(r'\-.*?LRB.*?\-.*?\-.*?RRB.*?\-', '', line)
        line_cleaned = rep
    elif ("-LRB-" in line and "-RRB-" not in line) or (
            "-LRB-" not in line and "-RRB-" in line):
        line_cleaned = line.replace("-LRB-", '"').replace("-RRB-", '"')
    else:
        line_cleaned = line
    return line_cleaned


def main(args):
    with open(args.source) as f:
        data = [row.rstrip() for row in f]
    
    write_lines(args.output, [filter_line(row) for row in data])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source',
                        help='Path to the source file',
                        required=True)
    parser.add_argument('-o', '--output',
                        help='Path to the output file',
                        required=True)
    args = parser.parse_args()
    main(args)