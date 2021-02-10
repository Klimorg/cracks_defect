import random
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import typer
import yaml
from loguru import logger
from sklearn.model_selection import train_test_split  # type: ignore
from utils import set_seed  # type: ignore
import csv

params = yaml.safe_load(open("configs/params.yaml"))["prepare"]

random_seed = params["seed"]
ratio = params["split"]

app = typer.Typer()


@logger.catch()
def get_files_and_labels(
    source_path: Path, extension: str = ".jpg"
) -> Tuple[List[Path], List[str]]:
    """Liste l'ensemble des images existantes suivant l'extension choisie
    dans tous les sous dossiers présents dans source_path.

    Retourne deux listes en bijections avec

    - images : url des images
    - labels : sous dossier où l'image est présente, pris comme label

    Args:
        source_path (str): adresse racine du dossier où chercher les images
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
        # filename = image_path.relative_to(source_path)
        filename = image_path.absolute()
        folder = image_path.parent.name
        if image_path.parent in FOLDERS:
            images.append(filename)
            labels.append(folder)

    return images, labels


@logger.catch()
def save_as_csv(
    filenames: List[Path], labels: List[str], destination: Path
) -> None:
    """Sauvegarde sous la forme d'un csv une dataframe pandas où la première
    colonne correspond aux adresses des images et la seconde aux labels
    correspondants.

    Args:
        filenames (List[str]): Liste des adresses des images, première colonne.
        labels (List[str]): Liste des labels correspondants, seconde colonne.
        destination (str): adresse du dossier où est sauvegardé le csv.
    """
    logger.info(
        f"Saving dataset in {destination} with labels ratio {Counter(labels)}"
    )
    header = ["filename", "label"]
    with open(destination, "w", newline="") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow(header)
        writer.writerows(zip(filenames, labels))

    # data_frame = pd.DataFrame({"filename": filenames, "label": labels})
    # data_frame.to_csv(destination)


@logger.catch()
@app.command()
def main(
    repo_path: Path = Path(__file__).parent.parent, ratio: float = ratio
) -> None:
    """Fonction principale.

    - Liste toutes les images via `get_files_and_labels` dans le dossier racine
    `repo_path`, par défaut `Path(__file__).parent.parent`.
    - Divise en 3 jeux train, val, & test suivant le `ratio`.
    - Sauvegarde ces 3 datasets sous la forme d'un csv via `save_as_csv`.

    Args:
        repo_path (str): Dossier racine.
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
