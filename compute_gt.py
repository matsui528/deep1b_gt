# Some IO codes are from:
# - https://github.com/facebookresearch/faiss/blob/master/benchs/link_and_code/datasets.py
# - https://github.com/facebookresearch/faiss/blob/master/contrib/vecs_io.py

import faiss
import numpy as np
import argparse
from pathlib import Path

def process_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="./", help="The path to the output directory")
    parser.add_argument("--szsufs", default=["1M", "10M", "100M"], nargs="*", help="The target sizes")
    parser.add_argument("--base_filename", default="./deep1b/base.fvecs", help="The path to the base veectors")
    parser.add_argument("--query_filename", default="./deep1b/deep1B_queries.fvecs", help="The path to the query vectors")
    return parser.parse_args()


def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def fvecs_read(fname):
    return ivecs_read(fname).view('float32')


def ivecs_mmap(fname):
    a = np.memmap(fname, dtype='int32', mode='r')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:]


def fvecs_mmap(fname):
    return ivecs_mmap(fname).view('float32')


def ivecs_write(fname, m):
    n, d = m.shape
    m1 = np.empty((n, d + 1), dtype='int32')
    m1[:, 0] = d
    m1[:, 1:] = m
    m1.tofile(fname)

def fvecs_write(fname, m):
    m = m.astype('float32')
    ivecs_write(fname, m.view('int32'))


def do_compute_gt(xb, xq, topk=100):
    nb, d = xb.shape
    index = faiss.IndexFlatL2(d)
    index.add(xb)
    _, ids = index.search(x=xq, k=topk)
    return ids.astype('int32')




if __name__ == '__main__':
    args = process_args()

    for szsuf in args.szsufs:
        print("Deep{}:".format(szsuf))
        dbsize = {"1M": 1000000, "10M": 10000000, "100M": 100000000}[szsuf]

        xb = fvecs_mmap(args.base_filename)
        xq = fvecs_read(args.query_filename)

        xb = np.ascontiguousarray(xb[:dbsize], dtype='float32')

        gt_fname = str(Path(args.out) / "deep{}_groundtruth.ivecs".format(szsuf))

        gt = do_compute_gt(xb, xq, topk=100)
        ivecs_write(gt_fname, gt)
