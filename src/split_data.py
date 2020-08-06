import os
from collections import defaultdict
from typing import Optional, NoReturn, List, Tuple, Iterable

import click
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection._split import _BaseKFold
from sklearn.utils.validation import check_array, check_random_state


class StratifiedGroupKFold(_BaseKFold):
    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = False,
        random_state: Optional[int] = None,
    ) -> NoReturn:
        super().__init__(n_splits, shuffle=shuffle, random_state=random_state)

    def _iter_test_indices(
        self,
        x: Optional[Iterable] = None,
        y: Optional[Iterable] = None,
        groups: Optional[Iterable] = None,
    ) -> List[int]:
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")

        unique_groups, groups = np.unique(groups, return_inverse=True)
        n_groups = len(unique_groups)

        if self.n_splits > n_groups:
            raise ValueError(
                f"Cannot have number of splits n_splits={self.n_splits} greater than the number of groups: {n_groups}."
            )

        n_labels = np.unique(y).shape[0]
        n_samples_per_label = dict(enumerate(np.bincount(y)))
        labels_per_group = defaultdict(lambda: np.zeros(n_labels))
        for label, group in zip(y, groups):
            labels_per_group[group][label] += 1
        groups_and_labels = list(labels_per_group.items())

        if self.shuffle:
            check_random_state(self.random_state).shuffle(groups_and_labels)

        labels_per_fold = defaultdict(lambda: np.zeros(n_labels))
        groups_per_fold = defaultdict(set)

        for group, labels in sorted(groups_and_labels, key=lambda x: -np.std(x[1])):
            best_fold = None
            min_eval = None
            for fold in range(self.n_splits):
                labels_per_fold[fold] += labels
                std_per_label = list()

                for label in range(n_labels):
                    label_std = np.std(
                        [
                            labels_per_fold[i][label] / n_samples_per_label[label]
                            for i in range(self.n_splits)
                        ]
                    )
                    std_per_label.append(label_std)
                labels_per_fold[fold] -= labels
                fold_eval = np.mean(std_per_label)

                if min_eval is None or fold_eval < min_eval:
                    min_eval = fold_eval
                    best_fold = fold
            labels_per_fold[best_fold] += labels
            groups_per_fold[best_fold].add(group)

        for fold in range(self.n_splits):
            test_groups = groups_per_fold[fold]
            test_indices = [
                index for index, group in enumerate(groups) if group in test_groups
            ]

            yield test_indices

    def split(
        self, x, y: Optional[Iterable] = None, groups: Optional[Iterable] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        y = check_array(y, ensure_2d=False, dtype=None)
        groups = check_array(groups, ensure_2d=False, dtype=None)
        return super().split(x, y, groups)


def split_holdout(
    df: pd.DataFrame, labels: np.ndarray, holdout_size: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    gkf = GroupShuffleSplit(n_splits=1, test_size=holdout_size, random_state=17)
    groups_by_patient_id_list = df["patient_id"].copy().tolist()

    result = []
    for train_idx, val_idx in gkf.split(df, labels, groups=groups_by_patient_id_list):
        train_fold = df.iloc[train_idx]
        val_fold = df.iloc[val_idx]
        result.append((train_fold, val_fold))

    train, holdout = result[0][0], result[0][1]
    return train, holdout


def split_cross_val(
    df: pd.DataFrame, labels: np.ndarray, n_folds: int
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    gkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=17)

    groups_by_patient_id_list = df["patient_id"].copy().tolist()

    splits = []
    for train_idx, val_idx in gkf.split(df, labels, groups=groups_by_patient_id_list):
        train_fold = df.iloc[train_idx]
        val_fold = df.iloc[val_idx]
        splits.append((train_fold, val_fold))
    return splits


def save_dataframes(
    splits: List[Tuple[pd.DataFrame, pd.DataFrame]], save_dir: str
) -> NoReturn:
    for idx, (train, val) in enumerate(splits):
        train.to_csv(os.path.join(save_dir, f"train_{idx}.csv"), index=False)
        val.to_csv(os.path.join(save_dir, f"val_{idx}.csv"), index=False)


@click.command()
@click.option("--path", type=str, default="../data/train.csv")
@click.option("--target_column", type=str, default="target")
@click.option("--n_folds", type=int, default=10, help="Count splits")
@click.option("--holdout_size", type=float, default=0.15, help="Holdout size")
@click.option("--output", type=str, default="../data/splits", help="Output directory")
def main(path, target_column, n_folds, holdout_size, output):
    df = pd.read_csv(path)
    labels = df[target_column].values
    train, holdout = split_holdout(df, labels, holdout_size)

    train_labels = train[target_column].values
    splits = split_cross_val(train, train_labels, n_folds)

    os.makedirs(output, exist_ok=True)
    save_dataframes(splits, output)
    holdout.to_csv(os.path.join(output, "holdout.csv"), index=False)


if __name__ == "__main__":
    main()
