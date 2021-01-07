from pathlib import Path

import hydra
import pandas as pd
import tensorflow as tf
import yaml
from sklearn.preprocessing import LabelEncoder

from model.resnet import get_resnet

# print(f"{sys.path}")

params = yaml.safe_load(open("configs/params.yml"))

AUTOTUNE = tf.data.experimental.AUTOTUNE

# def load_images(data_frame, column_name):
#     filename_list = data_frame[column_name].to_list()

#     return filename_list

# def load_labels(data_frame, column_name):
#     label_list = data_frame[column_name].to_list()
#     classes = list(set(label_list))

#     codec = LabelEncoder()
#     codec.fit(classes)
#     label_list = [codec.transform([label])[0] for label in label_list]

#     return label_list


@hydra.main(config_path="../configs/", config_name="params.yml")
def load_data(config):

    repo_path = hydra.utils.get_original_cwd()
    csv_path = Path(repo_path) / config.prepared_datas.train

    print(f"{csv_path}")

    data_frame = pd.read_csv(csv_path, index_col=0)
    print(f"{data_frame.head()}")

    filenames = data_frame["filename"].to_list()
    labels = data_frame["label"].to_list()

    filenames = filenames[:32]
    labels = labels[:32]
    classes = list(set(labels))

    codec = LabelEncoder()
    codec.fit(classes)
    labels = [codec.transform([label])[0] for label in label_list]

    # labels = load_labels(data_frame=df, column_name="label")
    # filenames = load_images(data_frame=df, column_name="filename")

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


def create_train_dataset(data_path, batch, repet, prefetch):
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


def create_val_dataset(data_path, batch, repet, prefetch):
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


# pour changer le lr https://stackoverflow.com/questions/59737875/keras-change-learning-rate

optim = {
    "rmsprop": tf.keras.optimizers.RMSProp(),
    "adam": tf.keras.optimizers.Adam(),
}


@hydra.main(config_path="../configs", config_name="params")
def train_loop(config):
    # ds, ds_val, epochs, learning_rate, loss_fn, metric_fn, optimizer

    ds = create_train_dataset(
        data_path=config.prepared_datas.train,
        batch=config.hyperparameters.batch_size,
        repet=config.hyperparameters.repetitions,
        prefetch=config.hyperparameters.prefetch,
    )
    ds_val = create_val_dataset(config.prepared_datas.train)

    model = get_resnet()

    optimizer = optimizer(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss=loss_fn, metrics=metric_fn)

    model.fit(ds, epochs=epochs, validation_data=ds_val)


if __name__ == "__main__":
    # train_loop()

    f, la = load_data()
