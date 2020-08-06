import argparse
import os
from multiprocessing.pool import ThreadPool

import imagehash
import pandas as pd
from PIL import Image
from tqdm import tqdm


def get_hash(path):
    try:
        return hashfunc(Image.open(path)), path
    except Exception:
        return "0", path


parser = argparse.ArgumentParser(description="Path")
parser.add_argument("--path")
parser.add_argument("--out")
if __name__ == "__main__":
    args = parser.parse_args()

    df = pd.read_csv(args.path)
    images = list(df["image_name"].values)
    images = list(
        map(lambda x: os.path.join("../data/external_data", x + ".jpg"), images)
    )
    is_duplicate = []
    print(f"Count images - {len(images)}")

    hashfunc = imagehash.phash

    hash_dict = {}
    deleted_images = 0
    results = ThreadPool(20).imap(get_hash, images)
    for i, (hash, path) in enumerate(tqdm(results)):
        if hash in hash_dict:
            is_duplicate.append(1)
            os.remove(path)
            deleted_images += 1
        else:
            hash_dict[hash] = path
            is_duplicate.append(0)
    print(f"Images deleted - {deleted_images}")

    df["is_duplicate"] = is_duplicate
    df = df[df["is_duplicate"] == 0]
    df.to_csv(args.out, index=False)
