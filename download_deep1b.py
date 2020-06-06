# The download codes are from: https://github.com/arbabenko/GNOIMI/blob/master/downloadDeep1B.py

import subprocess
from pathlib import Path
import fire


def download_yandisk(root, filename):
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


def run(root="./deep1b"):
    root = Path(root)

    # Create sub directories
    (root / "base").mkdir(exist_ok=True, parents=True)
    (root / "learn").mkdir(exist_ok=True, parents=True)

    # Download queries and gt
    for filename in ["deep1B_groundtruth.ivecs", "deep1B_queries.fvecs"]:
        download_yandisk(root=str(root), filename=filename)

    # Download base features
    cat = "cat "
    for n in range(37):
        filename = "base/base_{}".format(str(n).zfill(2))
        download_yandisk(root=str(root), filename=filename)
        cat += " " + str(root / filename)
    # Combine them into a single file (base.fvecs)
    cat += " > " + str(root / "base.fvecs")
    subprocess.run(cat, shell=True)

    # Download learn features
    cat = "cat "
    for n in range(14):
        filename = "learn/learn_{}".format(str(n).zfill(2))
        download_yandisk(root=str(root), filename=filename)
        cat += " " + str(root / filename)
    # Combine them into a single file (learn.fvecs)
    cat += " > " + str(root / "learn.fvecs")
    subprocess.run(cat, shell=True)


if __name__ == '__main__':
    fire.Fire(run)
