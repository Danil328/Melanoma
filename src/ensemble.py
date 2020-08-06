import os
import os.path as osp
from collections import defaultdict
import pandas as pd

import torch
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2
from ok_ppln.experiment import BaseExperiment
from ok_ppln.utils.checkpoint import load_checkpoint
from ok_ppln.utils.config import Config
from ok_tasks.core.task import get_task_factory, TaskFactory
from ok_tasks.datasets.classification import BinaryClsPlatformDataset
from torch.utils.data import DataLoader
from tqdm import tqdm


DATA_PATH = "/home/danil.akhmetov/ODKL/dvc-experiments/experiments/kaggle_melanoma_image_classification/data"


def make_dataloader(shape):
    transforms = Compose([Resize(shape, shape), Normalize(), ToTensorV2()])
    dataset = BinaryClsPlatformDataset(
        file_name=osp.join(DATA_PATH, "splits/holdout.csv"),
        path_column="image_name",
        label_column="benign_malignant",
        img_root=osp.join(DATA_PATH, "jpeg/train"),
        label_mapping_path=osp.join(DATA_PATH, "label_mapping.json"),
        transform=transforms,
        sep=",",
        extension=".jpg",
    )
    return DataLoader(
        dataset, batch_size=32, drop_last=False, num_workers=16, shuffle=False
    )


def create_models(model_names):
    MODELS = {256: dict(), 512: dict(), 768: dict()}
    for key in tqdm(MODELS.keys(), desc="Loading models"):
        for model_name in model_names:
            if str(key) in model_name:
                cfg = Config.fromfile(
                    osp.join(path_to_models, model_name, "config.json")
                )
                cfg["model"]["activation"] = "sigmoid"
                cfg["model"]["type"] = cfg["model"]["type"].split(".")[-1]
                cfg["model"]["encoder"][
                    "type"
                ] = "efficientnet_pytorch.EfficientNet.from_name"
                cfg["model"]["encoder"]["override_params"] = {
                    "num_classes": cfg["model"]["encoder"].pop("num_classes")
                }
                task_factory: TaskFactory = get_task_factory(cfg)
                experiment: BaseExperiment = task_factory.make_experiment(
                    cfg, device="cuda"
                )

                model = experiment.model
                load_checkpoint(
                    model,
                    osp.join(path_to_models, model_name, "model.pth"),
                    map_location="cuda",
                )
                _ = model.eval()
                MODELS[key][model_name] = model
    return MODELS


def create_dataloaders():
    DATALOADERS = {256: dict(), 512: dict(), 768: dict()}
    for key in tqdm(DATALOADERS.keys(), desc="Loading datasets"):
        DATALOADERS[key] = make_dataloader(key)
    return DATALOADERS


if __name__ == "__main__":
    path_to_models = "../models"
    model_names = os.listdir(path_to_models)

    MODELS = create_models(model_names)
    DATALOADERS = create_dataloaders()
    PREDICTS = defaultdict(list)
    TARGETS = defaultdict(list)

    with torch.no_grad():
        for key in tqdm(DATALOADERS.keys()):
            for batch in tqdm(DATALOADERS[key]):
                for model_name, model in MODELS[key].items():
                    predict = model(batch["image"].cuda())
                    PREDICTS[model_name].append(predict)
                TARGETS[key].append(batch["target"])
                # break

    PREDICTS = {
        key: torch.cat(value).squeeze().cpu().numpy() for key, value in PREDICTS.items()
    }

    df = pd.DataFrame.from_dict(PREDICTS)
    for key, value in TARGETS.items():
        df[f"target_{key}"] = torch.cat(value).cpu().numpy()
    df.to_csv("../models/predicts.csv", index=False)
