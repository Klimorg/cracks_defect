from pathlib import Path

import hydra
import pandas as pd
import tensorflow as tf
import yaml
from loguru import logger
from omegaconf import DictConfig
from sklearn.preprocessing import LabelEncoder

from model.resnet import get_resnet

# print(f"{sys.path}")

params = yaml.safe_load(open("configs/params.yml"))
n_classes = params["resnet"]["n_classes"]
img_shape = params["resnet"]["img_shape"]
random_seed = params["prepare"]["seed"]

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


def parse_image(filename, label):
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


"""Hydra
Lorsque que l'on travaille avec Hydra, toute la logique de la fonction doit
être contenu dans `main()`, on ne peut pas faire appel à des fonctions tierces
extérieures à `main()`, il faut tout coder dedans.

De même faire attention au dossier root : hydra modifie le dossier root :

Path(__file__).parent.parent donnera bien `.` mais cette racine est située
dans le dossier `outputs`, et non dans vrai dossier racine `cracks_defect`.

Il faut donc ici utiliser `hydra.utils.get_original_cwd()` pour pouvoir avoir
accès au dossier root `cracks_defect`.
"""


@logger.catch()
@hydra.main(config_path="../configs/", config_name="params.yml")
def load_data(config: DictConfig) -> None:

    logger.info("Data loading")

    repo_path = hydra.utils.get_original_cwd()
    csv_path = Path(repo_path) / config.prepared_datas.train

    logger.info(f"{csv_path.resolve()}")

    data_frame = pd.read_csv(csv_path, index_col=0)

    print(f"{data_frame.head()}")

    filenames = data_frame["filename"].to_list()
    labels = data_frame["label"].to_list()
    print(f"{filenames[:2]}, {labels[:2]}")

    filenames = filenames[:32]
    labels = labels[:32]

    logger.info("Featurize datas")

    classes = list(set(labels))

    codec = LabelEncoder()
    codec.fit(classes)
    labels = [codec.transform([label])[0] for label in labels]

    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.shuffle(len(filenames), seed=random_seed)
    dataset = dataset.repeat(config.hyperparameters.repetitions)
    dataset = dataset.map(parse_image, num_parallel_calls=AUTOTUNE)
    # dataset = dataset.map(train_preprocess, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(config.hyperparameters.batch_size)
    dataset = dataset.prefetch(config.hyperparameters.prefetch)

    # chargement du modèle

    # boucle d'entraînement

    # sauvegarde du modèle


# def train_preprocess(image, label):
#     image = tf.image.random_flip_left_right(image)
#     image = tf.image.random_flip_up_down(image)

#     return image, label


# def create_train_dataset(data_path, batch, repet, prefetch):
#     repo_path = Path(__file__).parent.parent
#     csv_path = repo_path / data_path

#     features, labels = load_data(csv_path)


#     return dataset


# def create_val_dataset(data_path, batch, repet, prefetch):
#     repo_path = Path(__file__).parent.parent
#     csv_path = repo_path / data_path

#     features, labels = load_data(csv_path)

#     dataset = tf.data.Dataset.from_tensor_slices((features, labels))
#     dataset = dataset.shuffle(len(features), seed=random_seed)
#     dataset = dataset.repeat(repet)
#     dataset = dataset.map(parse_image, num_parallel_calls=AUTOTUNE)
#     dataset = dataset.batch(batch)
#     dataset = dataset.prefetch(prefetch)
#     return dataset


# pour changer le lr :
# https://stackoverflow.com/questions/59737875/keras-change-learning-rate

# optim = {
#     "rmsprop": tf.keras.optimizers.RMSProp(),
#     "adam": tf.keras.optimizers.Adam(),
# }


# @hydra.main(config_path="../configs", config_name="params")
# def train_loop(config):
#     # ds, ds_val, epochs, learning_rate, loss_fn, metric_fn, optimizer

#     ds = create_train_dataset(
#         data_path=config.prepared_datas.train,
#         batch=config.hyperparameters.batch_size,
#         repet=config.hyperparameters.repetitions,
#         prefetch=config.hyperparameters.prefetch,
#     )
#     ds_val = create_val_dataset(config.prepared_datas.train)

#     model = get_resnet()

#     optimizer = optimizer(learning_rate=learning_rate)

#     model.compile(optimizer=optimizer, loss=loss_fn, metrics=metric_fn)

#     model.fit(ds, epochs=epochs, validation_data=ds_val)


if __name__ == "__main__":
    # train_loop()
    load_data()
