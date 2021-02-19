from pathlib import Path

import pytest
from src.prepare_dataset import get_files_paths


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
