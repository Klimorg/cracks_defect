from typing import List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from sklearn.preprocessing import LabelEncoder

params = yaml.safe_load(open("configs/params.yml"))
n_classes = params["resnet"]["n_classes"]
img_shape = params["resnet"]["img_shape"]
random_seed = params["prepare"]["seed"]

AUTOTUNE = tf.data.experimental.AUTOTUNE


def load_images(data_frame: pd.DataFrame, column_name: str) -> List[str]:
    """[summary]

    Args:
        data_frame (pd.DataFrame): [description]
        column_name (str): [description]

    Returns:
        List[str]: [description]
    """
    filename_list = data_frame[column_name].to_list()

    return filename_list


def load_labels(data_frame: pd.DataFrame, column_name: str) -> List[int]:
    """[summary]

    `from sklearn.preprocessing import LabelEncoder` marche très bien pour
    encoder des labels, mais si le dataset est très grand, le temps d'attente
    pour l'encodage de tous les labels devient beaucoup trop long.

    Voir la question StackOverflow suivante :
    [Question](https://stackoverflow.com/questions/45321999/
    how-can-i-optimize-label-encoding-for-large-data-sets-sci-kit-learn)

    Args:
        data_frame (pd.DataFrame): [description]
        column_name (str): [description]

    Returns:
        List[int]: [description]
    """
    label_list = data_frame[column_name].to_list()
    # classes = list(set(label_list))

    # codec = LabelEncoder()
    # codec.fit(classes)
    # label_list = [codec.transform([label])[0] for label in label_list]

    labels = np.unique(label_list, return_inverse=True)[1]
    dic = dict(zip(label_list, labels))
    label_list = np.vectorize(dic.get)(label_list)

    return label_list


def load_data(data_path: str) -> Tuple[List[str], List[int]]:
    """[summary]

    Args:
        data_path (str): [description]

    Returns:
        Tuple[List[str], List[int]]: [description]
    """

    df = pd.read_csv(data_path)
    labels = load_labels(data_frame=df, column_name="label")
    filenames = load_images(data_frame=df, column_name="filename")

    return filenames, labels


def parse_image(filename: str, label: int) -> Tuple[np.ndarray, List[int]]:
    """[summary]

    Args:
        filename (str): [description]
        label (int): [description]

    Returns:
        Tuple[np.ndarray, List[int]]: [description]
    """
    # convert the label to one-hot encoding
    label = tf.one_hot(label, n_classes)
    # decode image
    image = tf.io.read_file(filename)
    # Don't use tf.image.decode_image, or the output shape will be undefined
    image = tf.image.decode_jpeg(image)
    # This will convert to float values in [0, 1]
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [img_shape[0], img_shape[1]])

    return image, label


def train_preprocess(
    image: np.ndarray, label: List[int]
) -> Tuple[np.ndarray, List[int]]:
    """[summary]

    Args:
        image (np.ndarray): [description]
        label (List[int]): [description]

    Returns:
        [type]: [description]
    """
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)

    return image, label


def create_dataset(
    data_path: str, batch: int, repet: int, prefetch: int, augment: bool
):
    """[summary]

    Args:
        data_path (str): [description]
        batch (int): [description]
        repet (int): [description]
        prefetch (int): [description]
        augment (bool): [description]

    Returns:
        [type]: [description]
    """

    features, labels = load_data(data_path)
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    dataset = dataset.shuffle(len(features), seed=random_seed)
    dataset = dataset.repeat(repet)
    dataset = dataset.map(parse_image, num_parallel_calls=AUTOTUNE)
    if augment:
        dataset = dataset.map(train_preprocess, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(prefetch)
    return dataset


if __name__ == "__main__":

    ds = create_dataset(
        "datas/prepared_datas/train.csv", 16, 1, 1, 42, augment=False
    )
