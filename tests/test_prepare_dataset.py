from pathlib import Path

import pytest
from src.prepare_dataset import get_files_paths, get_images_paths_and_labels


@pytest.fixture
def root_directory():
    """[summary].

    Returns:
        [type]: [description]
    """
    return Path("tests/test_datas")


def test_get_files_paths(root_directory):
    """[summary].

    Args:
        root_directory ([type]): [description]
    """
    files_paths, subdirs = get_files_paths(root_directory)

    for idx in range(20):
        assert isinstance(files_paths[idx], Path)

        image_path = Path(files_paths[idx])
        assert image_path.is_file()

    assert isinstance(subdirs, list)
    assert len(subdirs) == 2


def test_get_images_paths_and_labels(root_directory):
    """[summary].

    Args:
        root_directory ([type]): [description]
    """
    files_paths, subdirs = get_files_paths(root_directory)

    images, labels = get_images_paths_and_labels(files_paths, subdirs)

    assert len(images) == 20
    assert len(labels) == 20

    for idx in range(10):
        assert images[idx].parent.name == "Negative"
        assert images[10 + idx].parent.name == "Positive"
