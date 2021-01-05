import os
import random
from pathlib import Path

import numpy as np
import typer
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42

app = typer.Typer()


def set_seed(RANDOM_SEED: int):
    os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)


set_seed(RANDOM_SEED)


@app.command()
def get_dataset(source_folder: str, ratio: float = 0.25):
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
