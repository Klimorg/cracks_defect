import os
import random
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import typer
from loguru import logger
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42

app = typer.Typer()


def set_seed(RANDOM_SEED: int):
    os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)


set_seed(RANDOM_SEED)


@app.command()
def get_files_and_labels(source_path: str, extension: str = ".jpg"):
    """[summary]

    Args:
        source_path (str): [description]
        extension (str, optional): [description]. Defaults to ".jpg".

    Returns:
        [type]: [description]
    """
    images = []
    labels = []

    FOLDERS_TO_LABELS = [x for x in Path(source_path).iterdir() if x.is_dir()]
    logger.info(
        f"Found {len(FOLDERS_TO_LABELS)} subfolders : {FOLDERS_TO_LABELS}"
    )
    logger.info(f"Searching {extension} files")
    images_paths = sorted(
        [x for x in Path(source_path).glob(f"**/*{extension}") if x.is_file()]
    )
    logger.info(f"Found {len(images_paths)} files")

    for image_path in images_paths:
        filename = image_path.absolute()
        folder = image_path.parent.name
        if folder in FOLDERS_TO_LABELS:
            images.append(filename)
            label = FOLDERS_TO_LABELS[folder]
            labels.append(label)

    return images, labels


def save_as_csv(filenames: List[str], labels: List[str], destination: str):
    """[summary]

    Args:
        filenames (List[str]): [description]
        labels (List[str]): [description]
        destination (str): [description]
    """

    data_dictionary = {"filename": filenames, "label": labels}
    data_frame = pd.DataFrame(data_dictionary)
    data_frame.to_csv(destination)


def main(repo_path: str):
    """[summary]

    Args:
        repo_path (str): [description]
    """

    data_path = repo_path / "data"
    train_path = data_path / "raw/train"
    test_path = data_path / "raw/val"

    train_files, train_labels = get_files_and_labels(train_path)
    test_files, test_labels = get_files_and_labels(test_path)

    prepared = data_path / "prepared"

    save_as_csv(train_files, train_labels, prepared / "train.csv")
    save_as_csv(test_files, test_labels, prepared / "test.csv")


@app.command()
def get_datasets(source_folder: str, ratio: float = 0.25):
    dic = {}
    source = Path(source_folder)
    subfolders = [x for x in source.iterdir() if x.is_dir()]

    for i, subfolder in enumerate(subfolders):
        dic[subfolder.stem] = i

    print(f"{dic}")

    subfiles = sorted([x for x in source.glob("**/*") if x.is_file()])
    random.shuffle(subfiles)

    train, val = train_test_split(
        subfiles, test_size=ratio, random_state=RANDOM_SEED
    )
    test, val = train_test_split(val, test_size=0.5, random_state=RANDOM_SEED)
    print(
        f"images in train : {len(train)},\n"
        f"images in val : {len(val)},\n"
        f"images in test : {len(test)}"
    )


if __name__ == "__main__":
    app()
