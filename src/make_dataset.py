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

with open("configs/params.yaml") as reproducibility_params:
    config = yaml.safe_load(reproducibility_params)["prepare"]

with open("configs/datasets/datasets.yaml") as datasets:
    address = yaml.safe_load(datasets)

raw_dataset_address = address["raw_dataset"]
train_dataset_address = address["prepared_dataset"]["train"]
val_dataset_address = address["prepared_dataset"]["val"]
test_dataset_address = address["prepared_dataset"]["test"]

random_seed = config["seed"]
split = config["split"]

app = typer.Typer()


def get_files_paths(
    root_directory: Path, extension: Optional[str] = ".jpg"
) -> Tuple[List[Path], List[Path]]:
    """Given an extension, gives a list of files and a list of subdirectories.

    Starting from `root_directory`, recursively search all subdirs of
    `root_directory` for all files of the given `extension`.

    We suppose here that each subdir corresponds to a different class in a
    classification problem, ie :

    root_directory

    |

    |___ subdir_A (class_A)

    |___ subdir_B (class_B)

    |___ ...

    Args:
        root_directory (Path): Root directory where to start the recursive search.
        extension (Optional[str], optional): Type of files we want to list
            during the recursive search. Defaults to ".jpg".

    Returns:
        The list of all files of the given extension, and the list of
        all the subdirectories.
    """
    subdirs = [subdir for subdir in Path(root_directory).iterdir() if subdir.is_dir()]

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


def get_images_paths_and_labels(
    images_paths: List[Path], folders: List[Path]
) -> Tuple[List[Path], List[str]]:
    """Creation of a representation of a dataset as lists (image, label).

    Given a list of `images_paths` and a list of `folders`, create two lists
    in bijection with each other, meaning to each row (an image path) in the
    first list corresponds a row (a label) in the second list.

    Also checks if the labels in the two lists correspond.

    Args:
        images_paths (List[Path]): List containing all the images_paths.
        folders (List[Path]): List constaining the folders (labels).

    Note:
        The paths given here are the absolute ones, not the relative ones. Be sure to
        have the rights paths when changing from one workspace to another.

    Raises:
        ValueError: Class deduced from image_path which isn't in 'folders' list.

    Note:
        `image_path.parent` has to be in `folders`. Otherwise that
        would mean that the first list `images_paths` contains more classes
        than the list `folders`. Raise an error if so.

    Returns:
        The two lists being in bijection image <-> label.
    """
    images = []
    labels = []

    logger.info("Creating images <-> labels representation.")
    for image_path in images_paths:
        filename = image_path.absolute()
        folder = image_path.parent.name
        if image_path.parent in folders:
            images.append(filename)
            labels.append(folder)
        else:
            raise ValueError(
                "Class deduced from image_path which isn't in 'folders' list."
            )

    return images, labels


def save_as_csv(filenames: List[Path], labels: List[str], destination: Path) -> None:
    """Save two lists of observations, labels as a csv files.

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


observations_list = List[Path]
labels_list = List[str]
Datasets = Tuple[
    observations_list,
    labels_list,
    observations_list,
    labels_list,
    observations_list,
    labels_list,
]


def create_train_val_test_datasets(
    raw_images: List[Path],
    raw_labels: List[str],
    test_size: Optional[float] = split,
) -> Datasets:
    """Creation of datasets.

    Create three image datasets (train, validation, and test) given `raw_images`
    and `raw_labels`.

    The first step is to gather `raw_images` and `raw_labels` in a single
    `dataset` entity, then shuffle it, this is to ensure that the dataset is
    already well shuffled before the before using the `scikit_learn` module
    `train_test_split` (for example `dataset` could be alphabetically sorted
    before the shuffling).

    Then `dataset` passes into `train_test_split` to first get the `images_train`
    and `labels_train` and an intermediate `images_val` and `labels_val`.

    The intermediate `images_val` and `labels_val` is then again split in half
    to get the actual `images_val`, `labels_val`, `images_test`, `labels_test`.

    Args:
        raw_images (List[Path]): Full list of the images used for the three
            datasets.
        raw_labels (List[str]): Full list of the labels used for the three
            datasets.
        test_size (Optional[float], optional): Ratio used in the first use of
            `train_test_split`. Defaults to split.

    Returns:
        Datasets: The three "datasets" returned as lists of images, labels.

    Note:
        Datasets is the alias for the following type.
        ```python
        Datasets = Tuple[
            List[Path], List[str], List[Path], List[str], List[Path], List[str]
            ]```
    """
    set_seed(random_seed)

    dataset = list(zip(raw_images, raw_labels))
    random.shuffle(dataset)
    shuffled_images, shuffled_labels = zip(*dataset)

    images_train, images_val, labels_train, labels_val = train_test_split(
        shuffled_images,
        shuffled_labels,
        test_size=test_size,
        random_state=random_seed,
    )
    images_val, images_test, labels_val, labels_test = train_test_split(
        images_val, labels_val, test_size=0.5, random_state=random_seed
    )

    return (
        images_train,
        labels_train,
        images_val,
        labels_val,
        images_test,
        labels_test,
    )


@app.command()
def main() -> None:
    """Main function."""
    files_paths, subdirs = get_files_paths(raw_dataset_address)
    raw_images, raw_labels = get_images_paths_and_labels(files_paths, subdirs)

    datasets_components = create_train_val_test_datasets(raw_images, raw_labels)

    save_as_csv(datasets_components[0], datasets_components[1], train_dataset_address)
    save_as_csv(datasets_components[2], datasets_components[3], val_dataset_address)
    save_as_csv(datasets_components[4], datasets_components[5], test_dataset_address)


if __name__ == "__main__":
    app()
