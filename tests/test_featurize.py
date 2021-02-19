from pathlib import Path

import pandas as pd
import pytest
from src.featurize import Featurize


@pytest.fixture
def feat() -> Featurize:
    """Returns a test class.

    Returns:
        Featurize: The class we test here. Defined in `src.featurize.py`
    """
    return Featurize(n_classes=2, img_shape=(224, 224, 3), random_seed=42)


@pytest.fixture
def test_df() -> pd.DataFrame:
    """Returns a test dataframe.

    Returns:
        pd.DataFrame: Test dataframe with manually crafted rows to check
        behavior during testing. Has 20 rows : 10 with 'Negative' elements
        followed by 10 with 'Positive' elements.
    """
    return pd.read_csv("tests/test_datas/test_datas.csv")


def test_constructor():
    """Test that the constructor is weel defined.

    You should only need the three following parameters to initiate this
    class :

    1. The number of classes in the dataset.
    2. The dimensions of the images.
    3. The random seed for reproducibility.
    """
    ft = Featurize(n_classes=2, img_shape=(224, 224, 3), random_seed=42)

    assert isinstance(ft, Featurize)


def test_load_images(feat: Featurize, test_df: pd.DataFrame):
    """Test of the function `load_images`.

    The function should take the column 'filename' of the dataframe en return
    it as a list.

    Also checks that we have the right number of elements in the list.

    Args:
        feat (Featurize): [description]
        test_df (pd.DataFrame): [description]
    """
    filenames = feat.load_images(data_frame=test_df, column_name="filename")

    assert isinstance(filenames, list)

    assert len(filenames) == 20

    for idx in range(20):
        assert isinstance(filenames[idx], str)

        image_path = Path(filenames[idx])
        assert image_path.is_file()


def test_load_labels(feat: Featurize, test_df: pd.DataFrame):
    """Test load_labels function.

    Args:
        feat (Featurize): [description]
        test_df (pd.DataFrame): [description]
    """
    zeros = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ones = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    labels_test = zeros + ones

    labels_list = feat.load_labels(data_frame=test_df, column_name="label")

    assert len(labels_list) == 20

    for idx in range(20):
        assert labels_list[idx] == labels_test[idx]
