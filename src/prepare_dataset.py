import csv
import random
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

import typer
import yaml
from loguru import logger
from sklearn.model_selection import train_test_split
from utils import set_seed

config = yaml.safe_load(open("configs/params.yaml"))["prepare"]

random_seed = config["seed"]
ratio = config["split"]

app = typer.Typer()


def get_files_paths(
    root_directory: Path, extension: Optional[str] = ".jpg"
) -> Tuple[List[Path], List[Path]]:
    """Given an extension, gives a list of files and a list of subdirectories.

    Starting from `root_directory`, recursively search all subdirs of
    `root_directory` for all files of the given `extension`.

    Args:
        root_directory (Path): Root directory where to start the recursive
        search.
        extension (Optional[str], optional): Type of files we want to list
        during the recursive search, eg "find all jpg images in all the subdirs
        or this root directory". Defaults to ".jpg".

    Returns:
        Tuple[List[Path], List[str]]: The list of all files of the given
        extension, and the list of all the subdirectories.
    """
    subdirs = [
        subdir for subdir in Path(root_directory).iterdir() if subdir.is_dir()
    ]

    logger.info(f"Found subfolders : {subdirs}")

    logger.info(f"Searching {extension} files")
    files_paths = sorted(
        image
        for image in Path(root_directory).glob(f"**/*{extension}")
        if image.is_file()
    )

    num_files = len(files_paths)
    logger.info(f"Found {num_files} files")

    return files_paths, subdirs


@logger.catch()
def get_files_and_labels(
    root_directory: Path, extension: Optional[str] = ".jpg"
) -> Tuple[List[Path], List[str]]:
    """Liste l'ensemble des images existantes suivant l'extension choisie.

    dans tous les sous dossiers présents dans source_path.

    Retourne deux listes en bijections avec

    - images : url des images
    - labels : sous dossier où l'image est présente, pris comme label

    Args:
        root_directory (Path): adresse racine du dossier où chercher les images
        extension (str, optional): type d'image que l'on cherche.

    Returns:
        Tuple[List[Path], List[str]]: [description]
    """
    images = []
    labels = []

    images_paths, folders = get_files_paths(root_directory, extension)

    logger.info("Creating images, labels full datasets.")
    for image_path in images_paths:
        filename = image_path.absolute()
        folder = image_path.parent.name
        if image_path.parent in folders:
            images.append(filename)
            labels.append(folder)

    return images, labels


def save_as_csv(
    filenames: List[Path], labels: List[str], destination: Path
) -> None:
    """Save a given dataframe as csv.

    Args:
        filenames (List[str]): Liste des adresses des images, première colonne.
        labels (List[str]): Liste des labels correspondants, seconde colonne.
        destination (Path): adresse du dossier où est sauvegardé le csv.
    """
    labels_distribution = Counter(labels)

    logger.info(f"Saving dataset in {destination}.")
    logger.info(f"Labels distribution {labels_distribution}.")

    header = ["filename", "label"]
    with open(destination, "w", newline="") as saved_csv:
        writer = csv.writer(saved_csv, delimiter=",")
        writer.writerow(header)
        writer.writerows(zip(filenames, labels))


@app.command()
def main(
    repo_path: Path = Path(__file__).parent.parent, ratio: float = ratio
) -> None:
    """Main function.

    - Liste toutes les images via `get_files_and_labels` dans le dossier racine
    `repo_path`, par défaut `Path(__file__).parent.parent`.
    - Divise en 3 jeux train, val, & test suivant le `ratio`.
    - Sauvegarde ces 3 datasets sous la forme d'un csv via `save_as_csv`.

    Args:
        repo_path (Path, optional): [description]. Defaults to Path(__file__).parent.parent.
        ratio (float, optional): [description]. Defaults to ratio.
    """
    set_seed(random_seed)

    data_path = repo_path / "datas" / "raw_dataset"
    prepared = repo_path / "datas" / "prepared_datas"

    raw_images, raw_labels = get_files_and_labels(data_path)

    dataset = list(zip(raw_images, raw_labels))
    random.shuffle(dataset)
    shuffled_images, shuffled_labels = zip(*dataset)

    images_train, images_val, labels_train, labels_val = train_test_split(
        shuffled_images,
        shuffled_labels,
        test_size=ratio,
        random_state=random_seed,
    )
    images_val, images_test, labels_val, labels_test = train_test_split(
        images_val, labels_val, test_size=0.5, random_state=random_seed
    )

    save_as_csv(images_train, labels_train, prepared / "train.csv")
    save_as_csv(images_val, labels_val, prepared / "val.csv")
    save_as_csv(images_test, labels_test, prepared / "test.csv")


if __name__ == "__main__":
    app()
