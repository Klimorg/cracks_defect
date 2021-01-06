import sys
from pathlib import Path

import pandas as pd
import tensorflow as tf
import yaml
from sklearn.preprocessing import LabelEncoder

from model.resnet import get_resnet

print(f"{sys.path}")

params = yaml.safe_load(open("configs/params.yml"))

random_seed = params["prepare"]["seed"]
n_classes = params["resnet"]["n_classes"]
bacth_size = params["train"]["batch_size"]
repetitions = params["train"]["repetitions"]
prefetch = params["train"]["prefetch"]

AUTOTUNE = tf.data.experimental.AUTOTUNE


def load_images(data_frame, column_name):
    filename_list = data_frame[column_name].to_list()

    return filename_list


def load_labels(data_frame, column_name):
    label_list = data_frame[column_name].to_list()
    classes = list(set(label_list))
    print(f"{classes}")

    codec = LabelEncoder()
    codec.fit(classes)
    label_list = [codec.transform([label])[0] for label in label_list]

    return label_list


def load_data(data_path):

    df = pd.read_csv(data_path)
    labels = load_labels(data_frame=df, column_name="label")
    filenames = load_images(data_frame=df, column_name="filename")

    return filenames, labels


def parse_image(filename, label):
    # convert the label to one-hot encoding
    label = tf.one_hot(label, n_classes)
    # decode image
    image = tf.io.read_file(filename)
    # Don't use tf.image.decode_image, or the output shape will be undefined
    image = tf.image.decode_jpeg(image)
    # This will convert to float values in [0, 1]
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [224, 224])
    return image, label


def train_preprocess(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)

    return image, label


def create_train_dataset(
    data_path, batch=bacth_size, repet=repetitions, prefetch=prefetch
):
    repo_path = Path(__file__).parent.parent
    csv_path = repo_path / data_path

    features, labels = load_data(csv_path)

    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    dataset = dataset.shuffle(len(features), seed=random_seed)
    dataset = dataset.repeat(repet)
    dataset = dataset.map(parse_image, num_parallel_calls=AUTOTUNE)
    dataset = dataset.map(train_preprocess, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(prefetch)
    return dataset


def create_val_dataset(
    data_path, batch=bacth_size, repet=repetitions, prefetch=prefetch
):
    repo_path = Path(__file__).parent.parent
    csv_path = repo_path / data_path

    features, labels = load_data(csv_path)

    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    dataset = dataset.shuffle(len(features), seed=random_seed)
    dataset = dataset.repeat(repet)
    dataset = dataset.map(parse_image, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(prefetch)
    return dataset


if __name__ == "__main__":
    data_path = "datas/prepared_datas/train.csv"
    dataset = create_train_dataset(data_path)

    for image, label in dataset.take(1):
        print(f"{image}, {label}")

    model = get_resnet()
    model.summary()
