# Some IO codes are from:
# - https://github.com/facebookresearch/faiss/blob/master/contrib/vecs_io.py

import numpy as np
import argparse
import texmex_python

def process_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", help="The input file (.fvecs)")
    parser.add_argument("--dst", help="The output file (.fvecs)")
    parser.add_argument("--topk", type=int, help="The number of element to pick up")
    return parser.parse_args()


def ivecs_write(fname, m):
    n, d = m.shape
    m1 = np.empty((n, d + 1), dtype='int32')
    m1[:, 0] = d
    m1[:, 1:] = m
    m1.tofile(fname)

def fvecs_write(fname, m):
    m = m.astype('float32')
    ivecs_write(fname, m.view('int32'))

if __name__ == "__main__":
    args = process_args()

    # Reat topk vector one by one
    vecs = []
    with open(args.src, "rb") as f:
        for vec in texmex_python.reader.read_fvec_iter(f):
            vecs.append(vec)
            if len(vecs) == args.topk:
                break
    vecs = np.array(vecs, dtype=np.float32)
    assert vecs.shape[0] == args.topk
    print("vecs.shape:", vecs.shape)

    fvecs_write(args.dst, vecs)
