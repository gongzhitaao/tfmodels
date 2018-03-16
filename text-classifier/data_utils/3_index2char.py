import os
import logging
import argparse

from tqdm import tqdm
import numpy as np


logging.basicConfig(format='%(asctime)-15s %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
info = logger.info


def parse_args():
    parser = argparse.ArgumentParser(
        description='Get each character from its UTF-8 encoding with chr.')
    parser.add_argument('fname', type=str, help='index file')
    return parser.parse_args()


def main(args):
    mat = np.load(os.path.expanduser(args.fname))
    for vec in mat:
        sent = ''.join(chr(c) for c in sent)
        print(sent)


if __name__ == '__main__':
    info('THE BEGIN')
    main(parse_args())
    info('THE END')
