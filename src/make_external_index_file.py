import os

import click
import pandas as pd


@click.command()
@click.option("--paths", "-p", type=str, multiple=True)
@click.option("--image-columns", "-i", type=str, multiple=True)
@click.option("--target-columns", "-t", type=str, multiple=True)
@click.option("--prefixs", "-f", type=str, multiple=True)
@click.option(
    "--output", type=str, default="external_train.csv", help="Output directory"
)
def main(paths, image_columns, target_columns, prefixs, output):
    label_mapping = {0: "benign", 1: "malignant"}
    label_mapping_inv = {"benign": 0, "malignant": 1}
    df = pd.DataFrame()

    image_name, benign_malignant = [], []
    for path, image_column, target_column, prefix in zip(
        paths, image_columns, target_columns, prefixs
    ):
        print(prefix)
        df_tmp = pd.read_csv(path)
        df_tmp[image_column] = df_tmp[image_column].map(
            lambda x: os.path.join(prefix, x)
        )
        image_name.extend(list(df_tmp[image_column].values))
        benign_malignant.extend(
            list(map(lambda x: label_mapping.get(x, x), df_tmp[target_column].values))
        )

    df["image_name"] = image_name
    df["benign_malignant"] = benign_malignant
    df["target"] = df["benign_malignant"].map(label_mapping_inv)
    df.to_csv(output, index=False)


if __name__ == "__main__":
    main()
