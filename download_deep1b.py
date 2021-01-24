# The download codes are from: https://github.com/arbabenko/GNOIMI/blob/master/downloadDeep1B.py

import subprocess
from pathlib import Path
import argparse

def process_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="./deep1b", help="The path to the data directory")
    parser.add_argument("--base_n", default=37, type=int, help="The number of batches for base vectors to be downloaded")
    parser.add_argument("--learn_n", default=14, type=int, help="The number of batches for learn vectors to be downloaded")
    parser.add_argument("--ops", default=["all"], nargs="*", help="Operations to run",
                        choices=["all", "query", "gt", "base", "base_merge", "learn", "learn_merge"])
    return parser.parse_args()


def wget_yandisk(root, filename):
    """
    Wget data from the Yandisk space
    Args:
        root: The path to the data directory.
        filename: The filename to download. This will be saved on root/filename
    """
    # Obtain wget path
    command = 'curl "{url}{yadiskLink}&path=/{filename}"'.format(
        url="https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key=",
        yadiskLink="https://yadi.sk/d/11eDCm7Dsn9GA",
        filename=filename
        )
    print("command: {}".format(command))
    process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    (out, err) = process.communicate()
    wgetLink = out.decode('utf-8').split(',')[0][8:]

    # Run wget if the file does not exist
    filepath = Path(root) / filename
    if filepath.exists():
        print("{} already exists. Skip".format(filepath))
        return
    wgetCommand = 'wget {wgetLink} -O {filepath}'.format(wgetLink=wgetLink, filepath=str(filepath))
    print("Downloading {} ...".format(filename))
    process = subprocess.Popen(wgetCommand, stdin=subprocess.PIPE, shell=True)
    process.stdin.write('e'.encode('utf-8'))
    process.wait()


def download_batches(root, prefix, batch_n):
    """
    Helper function to download batches
    Args:
        root: The path to the data directory.
        prefix: "base" or "learn"
        batch_n: The number of batches to be downloaded
    """
    assert prefix in ["base", "learn"]
    for n in range(batch_n):
        filename = "{}/{}_{}".format(prefix, prefix, str(n).zfill(2))  # e.g., base/base_00
        wget_yandisk(root=root, filename=filename)

def merge_batches(root, prefix, batch_n):
    """
    Helper function to merge batches into a single file
    Args:
        root: The path to the data directory.
        prefix: "base" or "learn"
        batch_n: The number of batches to be downloaded
    """
    assert prefix in ["base", "learn"]
    cat = "cat "
    for n in range(batch_n):
        filename = "{}/{}_{}".format(prefix, prefix, str(n).zfill(2))  # e.g., base/base_00
        cat += " " + str(root / filename)
    cat += " > " + str(root / f"{prefix}.fvecs")  # e.g.: cat ./deep1b/base_00 ./deep1b/base_01 ... > ./deep1b/base.fvecs
    subprocess.run(cat, shell=True)



if __name__ == '__main__':
    args = process_args()
    root = Path(args.root)
    assert root.is_dir()
    ops = args.ops

    # If "all" is specified, run the all operations
    if "all" in ops:
        ops = ["query", "gt", "base", "base_merge", "learn", "learn_merge"]

    # Create sub directories
    (root / "base").mkdir(exist_ok=True, parents=True)
    (root / "learn").mkdir(exist_ok=True, parents=True)

    # Download queries
    if "query" in ops:
        wget_yandisk(root=str(root), filename="deep1B_queries.fvecs")

    # Download groundtruth
    if "gt" in ops:
        wget_yandisk(root=str(root), filename="deep1B_groundtruth.ivecs")

    # Download base features
    if "base" in ops: 
        download_batches(root, "base", args.base_n)

    # Merge base features into a single file
    if "base_merge" in ops:
        merge_batches(root, "base", args.base_n)

    # Download learn features
    if "learn" in ops: 
        download_batches(root, "learn", args.learn_n)

    # Merge learn features into a single file
    if "learn_merge" in ops:
        merge_batches(root, "learn", args.learn_n)
