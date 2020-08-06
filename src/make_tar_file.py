import argparse
import glob
import tarfile
import os.path

from tqdm import tqdm


def make_tarfile(output_filename, files, n=2):
    with tarfile.open(output_filename, "w:gz") as tar:
        for i in files:
            tar.add(
                name=i, arcname="/".join(i.split("/")[n:]), recursive=False,
            )


parser = argparse.ArgumentParser(description="Make tar file")
parser.add_argument("--paths", nargs="+", type=str)
parser.add_argument("--output")
parser.add_argument("--step", default=1000, type=int)
if __name__ == "__main__":
    args = parser.parse_args()

    files = []
    for path in args.paths:
        files.extend(glob.glob(os.path.join(path, "**"), recursive=True))
    print(f"Count files - {len(files)}")

    for idx, i in enumerate(tqdm(range(0, len(files), args.step))):
        if not os.path.exists(args.output + f"_{idx}.tar"):
            make_tarfile(
                os.path.join(args.output + f"_{idx}.tar.gz"), files[i : i + args.step],
            )
