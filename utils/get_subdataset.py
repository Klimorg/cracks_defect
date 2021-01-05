import os
import random
from pathlib import Path
from typing import List
from collections import Counter

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


@logger.catch()
def get_files_and_labels(
    source_path: str, extension: str = ".jpg"
) -> List[str]:
    """Liste l'ensemble des images existantes suivant l'extension choisie
    dans tous les sous dossiers présents dans source_path.

    Retourne deux listes en bijections avec

    - images : url des images
    - labels : sous dossier où l'image est présente, pris comme label

    Args:
        source_path (str): adresse racine du dossier où chercher les images
        dans les sous dossiers
        extension (str, optional): type d'image que l'on cherche.

    Returns:
       Types de deux listes d'adresse des images et labels correspondants.
    """
    images = []
    labels = []

    FOLDERS = [x for x in Path(source_path).iterdir() if x.is_dir()]
    logger.info(f"Found {len(FOLDERS)} subfolders : {FOLDERS}")

    logger.info(f"Searching {extension} files")
    images_paths = sorted(
        [x for x in Path(source_path).glob(f"**/*{extension}") if x.is_file()]
    )
    logger.info(f"Found {len(images_paths)} files")

    logger.info("Creating images, labels full datasets.")
    for image_path in images_paths:
        filename = image_path.absolute()
        folder = image_path.parent.name
        if image_path.parent in FOLDERS:
            images.append(filename)
            labels.append(folder)

    return images, labels


@logger.catch()
def save_as_csv(filenames: List[str], labels: List[str], destination: str):
    """[summary]

    Args:
        filenames (List[str]): [description]
        labels (List[str]): [description]
        destination (str): [description]
    """
    logger.info(
        f"Saving dataset in {destination} with labels ratio {Counter(labels)}"
    )
    data_dictionary = {"filename": filenames, "label": labels}
    data_frame = pd.DataFrame(data_dictionary)
    data_frame.to_csv(destination)


@logger.catch()
@app.command()
def main(repo_path: str = Path(__file__).parent.parent, ratio: float = 0.25):
    """[summary]

    Args:
        repo_path (str): [description]
    """

    data_path = repo_path / "datas" / "raw_datas"
    prepared = repo_path / "datas" / "prepared_datas"

    raw_images, raw_labels = get_files_and_labels(data_path)

    dataset = list(zip(raw_images, raw_labels))
    random.shuffle(dataset)
    shuffled_images, shuffled_labels = zip(*dataset)

    images_train, images_val, labels_train, labels_val = train_test_split(
        shuffled_images,
        shuffled_labels,
        test_size=ratio,
        random_state=RANDOM_SEED,
    )
    images_val, images_test, labels_val, labels_test = train_test_split(
        images_val, labels_val, test_size=0.5, random_state=RANDOM_SEED
    )

    save_as_csv(images_train, labels_train, prepared / "train.csv")
    save_as_csv(images_val, labels_val, prepared / "val.csv")
    save_as_csv(images_test, labels_test, prepared / "test.csv")


if __name__ == "__main__":
    app()
