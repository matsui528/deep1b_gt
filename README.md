# deep1b_gt

Compute the exact 100 nearest neighbors for deep1M, deep10M, and deep100M datasets. We can use these neighbors as the ground truth for the search task for deep{1, 10, 100}m datasets.

Note that deep{1, 10, 100}M datasets are the top {1, 10, 100}M vectors of deep1b dataset, respectively.
## How to run
```bash
git clone https://github.com/matsui528/deep1b_gt.git
cd deep1b_gt
pip install -r requirements.txt

# Download Deep1b data on ./deep1b.
# After downloading the data, the direcotry is: 
# ................
# The data will be concatenated as base.fvecs and learn.fvecs
# ./deep1b/base.fvecs will take 361.4 GB (md5sum: )
# ./deep1b/learn.fvecs will take 129.5 GB (md5sum: )
# This may take several days
python download_deep1b.py --root ./deep1b

# Compute groundtruth. You need faiss
conda install -c pytorch faiss-cpu
python compute_gt.py --root ./deep1b --out ./
```

## Result
You can download the result from here
- [deep1M_groundtruth.ivecs]()
- [deep10M_groundtruth.ivecs]()
- [deep100M_groundtruth.ivecs]()


## Reference
Several codes are from:
- [how to download deep1b](https://github.com/arbabenko/GNOIMI/blob/master/downloadDeep1B.py)
- [how to compute gt](https://github.com/facebookresearch/faiss/blob/master/benchs/link_and_code/datasets.py)
