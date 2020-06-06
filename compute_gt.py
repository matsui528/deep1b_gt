# Some IO codes are from: https://github.com/facebookresearch/faiss/blob/master/benchs/link_and_code/datasets.py

import faiss
import numpy as np
import fire
from pathlib import Path

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


def do_compute_gt(xb, xq, topk=100):
    nb, d = xb.shape
    index = faiss.IndexFlatL2(d)
    index.add(xb)
    _, ids = index.search(x=xq, k=topk)
    return ids.astype('int32')


def run(root="./deep1b", out="./"):
    root = Path(root)
    out = Path(out)

    for szsuf in ["1M", "10M", "100M"]:
        print("Deep{}:".format(szsuf))
        dbsize = {"1M": 1000000, "10M": 10000000, "100M": 100000000}[szsuf]

        xb = fvecs_mmap(str(root / "base.fvecs"))
        xq = fvecs_read(str(root / "deep1B_queries.fvecs"))

        xb = xb[:dbsize]

        gt_fname = str(out / "deep{}_groundtruth.ivecs".format(szsuf))

        gt = do_compute_gt(xb, xq, topk=100)
        ivecs_write(gt_fname, gt)


if __name__ == '__main__':
    fire.Fire(run)
