from typing import List, Tuple, TypeVar

import numpy as np
import pandas as pd
import tensorflow as tf
from loguru import logger

gen_type = TypeVar("gen_type")


class Tensorize(object):
    """Class used to create tensor datasets for TensorFlow.

    Args:
        object (object): The base class of the class hierarchy, used only to enforce
            WPS306. See https://wemake-python-stylegui.de/en/latest/pages/usage/
            violations/consistency.html#consistency.
    """

    def __init__(
        self, n_classes: int, img_shape: Tuple[int, int, int], random_seed: int
    ) -> None:
        """Initialization of the class Featurize.

        Initialize the class the number of classes in the datasets, the shape of the
        images and the random seed.

        Args:
            n_classes (int): Number of classes in the dataset.
            img_shape (Tuple[int, int, int]): Dimension of the image, format is (H,W,C).
            random_seed (int): Fixed random seed for reproducibility.
        """
        self.n_classes = n_classes
        self.img_shape = img_shape
        self.random_seed = random_seed
        self.AUTOTUNE = tf.data.experimental.AUTOTUNE

    def load_images(self, data_frame: pd.DataFrame, column_name: str) -> List[str]:
        """Load the images as a list.

        Take the dataframe containing the observations and the labels and the return the
        column containing the observations as a list.

        Args:
            data_frame (pd.DataFrame): Dataframe containing the dataset.
            column_name (str): The name of the column containing the observations.

        Returns:
            The list of observations deduced from the dataframe.
        """
        return data_frame[column_name].tolist()

    def load_labels(self, data_frame: pd.DataFrame, column_name: str) -> List[int]:
        """Load the labels as a list and encode them.

        Take the dataframe containing the observations and the labels and the return the
        column containing the labels as an encoded list.

        The encoding is done by taking the set of labels, alphabetically sorted, and
        then transforming them as integers starting from 0.

        `from sklearn.preprocessing import LabelEncoder` works well to encode labels,
        but if the dataset is huge, the time it takes to encode all the labels is
        growing fast. We use anumpy and vectorization to speed up the time.

        See the StackOverflow question :
        [Question](https://stackoverflow.com/questions/45321999/
        how-can-i-optimize-label-encoding-for-large-data-sets-sci-kit-learn)

        Args:
            data_frame (pd.DataFrame): Dataframe containing the dataset.
            column_name (str): The name of the column containing the labels.

        Returns:
            The list of encoded labels deduced from the dataframe.
        """
        label_list = data_frame[column_name].tolist()
        classes = sorted(set(label_list))
        logger.info(f"Found following labels {classes}")

        labels = np.unique(label_list, return_inverse=True)[1]
        dic = dict(zip(label_list, labels))
        logger.info(f"Dictionnary creation {dic}")
        vectorized_get = np.vectorize(dic.get)

        return vectorized_get(label_list)

    def parse_image_and_label(
        self, filename: str, label: int
    ) -> Tuple[np.ndarray, int]:
        """Transform image and label.

        Parse image to go from path to a resized np.ndarray, and parse the labels to
        one-hot encode them.

        Args:
            filename (str): The path of the image to parse.
            label (int): The label of the image, as an int, to one-hot encode.

        Returns:
            A np.ndarray corresponding to the image and the corresponding one-hot label.
        """
        resized_dims = [self.img_shape[0], self.img_shape[1]]
        # convert the label to one-hot encoding
        label = tf.one_hot(label, self.n_classes)
        # decode image
        image = tf.io.read_file(filename)
        # Don't use tf.image.decode_image,
        # or the output shape will be undefined
        image = tf.image.decode_jpeg(image)
        # This will convert to float values in [0, 1]
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, resized_dims)

        return image, label

    def train_preprocess(
        self, image: np.ndarray, label: List[int]
    ) -> Tuple[np.ndarray, List[int]]:
        """Augmentation preprocess, if needed.

        Args:
            image (np.ndarray): The image to augment.
            label (List[int]): The corresponding label.

        Returns:
            The augmented pair.
        """
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)

        return image, label

    def create_dataset(
        self,
        data_path: str,
        batch: int,
        repet: int,
        prefetch: int,
        augment: bool,
    ) -> tf.data.Dataset:
        """Creation of a tensor dataset for TensorFlow.

        Args:
            data_path (str): Path where the csv file containing the dataframe is
                located.
            batch (int): Batch size, usually 32.
            repet (int): How many times the dataset has to be repeated.
            prefetch (int): How many batch the CPU has to prepare in advance for the
                GPU.
            augment (bool): Does the dataset has to be augmented or no.

        Returns:
            A batch of observations and labels.
        """
        df = pd.read_csv(data_path)
        features = self.load_images(data_frame=df, column_name="filename")
        labels = self.load_labels(data_frame=df, column_name="label")

        dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        dataset = dataset.shuffle(len(features), seed=self.random_seed)
        dataset = dataset.repeat(repet)
        dataset = dataset.map(
            self.parse_image_and_label, num_parallel_calls=self.AUTOTUNE
        )
        if augment:
            dataset = dataset.map(
                self.train_preprocess, num_parallel_calls=self.AUTOTUNE
            )
        dataset = dataset.batch(batch)
        dataset = dataset.cache()
        return dataset.prefetch(prefetch)
