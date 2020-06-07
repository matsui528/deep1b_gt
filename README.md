# deep1b_gt

Compute the exact 100 nearest neighbors for deep1M, deep10M, and deep100M datasets. We can use these neighbors as the ground truth for the search task for deep{1, 10, 100}M datasets.

Note that deep{1, 10, 100}M datasets are the top {1, 10, 100}M vectors of deep1b dataset, respectively.
## How to run
```bash
git clone https://github.com/matsui528/deep1b_gt.git
cd deep1b_gt
pip install -r requirements.txt

# Download Deep1b data on ./deep1b. This may take several days. I recommend preparing 2TB of the disk space.
# After downloading the data, the structure of the directory would be: 
# .
# ├── base
# │   ├── base_00
# │   ├── base_01
# │   ...
# │   ├── base_35
# │   └── base_36
# ├── base.fvecs                # 388,000,000,000 bytes
# ├── deep1B_groundtruth.ivecs
# ├── deep1B_queries.fvecs
# ├── learn
# │   ├── learn_00
# │   ├── learn_01
# │   ...
# │   ├── learn_12
# │   └── learn_13
# └── learn.fvecs               # 139,090,240,000 bytes
python download_deep1b.py --root ./deep1b

# Compute groundtruth. You need faiss
conda install -c pytorch faiss-cpu
python compute_gt.py --root ./deep1b --out ./
```

## Result
You can download the results from here [https://github.com/matsui528/deep1b_gt/releases/download/v0.1.0/gt.zip](https://github.com/matsui528/deep1b_gt/releases/download/v0.1.0/gt.zip)
- deep1M_groundtruth.ivecs
- deep10M_groundtruth.ivecs
- deep100M_groundtruth.ivecs


## Reference
Several codes are from:
- [how to download deep1b](https://github.com/arbabenko/GNOIMI/blob/master/downloadDeep1B.py)
- [how to compute gt](https://github.com/facebookresearch/faiss/blob/master/benchs/link_and_code/datasets.py)
