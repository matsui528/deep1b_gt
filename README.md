# deep1b_gt

Compute the exact 100 nearest neighbors for deep1M, deep10M, and deep100M datasets. We can use these neighbors as the ground truth for the search task for deep{1, 10, 100}M datasets.

Note that deep{1, 10, 100}M datasets are the top {1, 10, 100}M vectors of deep1b dataset, respectively.

## Result
You can download the results from here [https://github.com/matsui528/deep1b_gt/releases/download/v0.1.0/gt.zip](https://github.com/matsui528/deep1b_gt/releases/download/v0.1.0/gt.zip)
- deep1M_groundtruth.ivecs
- deep10M_groundtruth.ivecs
- deep100M_groundtruth.ivecs


## How to run
```bash
git clone https://github.com/matsui528/deep1b_gt.git
cd deep1b_gt

# Download Deep1b data on ./deep1b. This may take several days. I recommend preparing 2TB of the disk space.
python download_deep1b.py --root ./deep1b

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

# rm -rf ./deep1b/base ./deep1b/learn    # Optionally, you can delete base and learn, that should not be used anymore

# Compute groundtruth. You need faiss
conda install -c pytorch faiss-cpu
python compute_gt.py --out ./

# You'll get deep1M_groundtruth.ivecs, deep10M_groundtruth.ivecs, and deep100M_groundtruth.ivecs
```






## (Bonus) Deep1M
As the deep1b dataset is too huge, you may want to download its subset (top 1M vectors) only. The following script will
- pick up the first 1M vectors from `base_00` to construct `deep1M_base.fvecs`
- pick up the first 100K vectors from `learn_00` to construct `deep1M_learn.fvecs`
```bash
git clone https://github.com/matsui528/deep1b_gt.git
cd deep1b_gt

# Download base_00, learn_00, and query on ./deep1b. This may take some hours. I recommend preparing 25GB of the disk space.
python download_deep1b.py --root ./deep1b --base_n 1 --learn_n 1 --ops query base learn 

# Select top 1M vectors from base_00 and save it on deep1M_base.fvecs
python pickup_vecs.py --src ./deep1b/base/base_00 --dst ./deep1b/deep1M_base.fvecs --topk 1000000

# Select top 100K vectors from learn_00 and save it on deep1M_learn.fvecs
python pickup_vecs.py --src ./deep1b/learn/learn_00 --dst ./deep1b/deep1M_learn.fvecs --topk 100000

# After running the above commands, the structure of the directory would be: 
# .
# ├── base
# │   └── base_00
# ├── deep1M_base.fvecs                # 388,000,000 bytes
# ├── deep1B_queries.fvecs             
# ├── learn
# │   └── learn_00
# └── deep1M_learn.fvecs               # 38,800,000 bytes

# rm -rd ./deep1b/base ./deep1b/learn    # Optionally, you can delete base and learn, that should not be used anymore


# Compute groundtruth. You need faiss
conda install -c pytorch faiss-cpu
python compute_gt.py --out ./ --szsufs 1M --base_filename ./deep1b/deep1M_base.fvecs --query_filename ./deep1b/deep1B_queries.fvecs 

# You'll get deep1M_groundtruth.ivecs
```





## Reference
Several codes are from:
- [how to download deep1b](https://github.com/arbabenko/GNOIMI/blob/master/downloadDeep1B.py)
- [how to compute gt](https://github.com/facebookresearch/faiss/blob/master/benchs/link_and_code/datasets.py)
