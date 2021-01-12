from pathlib import Path

import hydra
import mlflow
import mlflow.tensorflow
import tensorflow as tf
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from tensorflow.keras import backend as K
import os
import numpy as np
import random

from featurize import create_dataset
from model.resnet import get_resnet
from model.wide_resnet import get_wide_resnet
from tensorflow.keras.callbacks import Callback


class MlFlowCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        mlflow.log_metrics(logs, step=epoch)


@logger.catch()
@hydra.main(config_path="../configs/", config_name="params.yml")
def train(config: DictConfig) -> tf.keras.Model:
    """[summary]

    Lorsque que l'on travaille avec Hydra, toute la logique de la fonction doit
    être contenu dans `main()`, on ne peut pas faire appel à des fonctions
    tierces extérieures à `main()`, il faut tout coder dedans.

    De même faire attention au dossier root : hydra modifie le dossier root :

    Path(__file__).parent.parent donnera bien `.` mais cette racine est située
    dans le dossier `outputs`, et non dans vrai dossier racine `cracks_defect`.

    Il faut donc ici utiliser `hydra.utils.get_original_cwd()` pour pouvoir
    avoir accès au dossier root `cracks_defect`.

    Pour changer le learning rate d'un optimiseur
    **après avoir compilé le modèle**, voir la question StackOverflow suivante.

    [Modifier de lr](https://stackoverflow.com/questions/
    59737875/keras-change-learning-rate)

    Args:
        config (DictConfig): [description]

    Returns:
        tf.keras.Model: [description]
    """

    repo_path = hydra.utils.get_original_cwd()

    mlflow.set_tracking_uri(
        "file://" + hydra.utils.get_original_cwd() + "/mlruns"
    )
    mlflow.set_experiment(config.mlflow.experiment_name)
    # mlflow.entities.Experiment(
    #     artifact_location=hydra.utils.get_original_cwd() + "/mlruns/artifact"
    # )

    logger.info("MLFlow uris")
    print(f"{repo_path}")
    print(f"{mlflow.get_tracking_uri()}")
    # print(f"{mlflow.get_artifact_uri()}")

    logger.info(f"{OmegaConf.to_yaml(config)}")

    os.environ["PYTHONHASHSEED"] = str(config.prepare.seed)
    random.seed(config.prepare.seed)
    np.random.seed(config.prepare.seed)
    tf.random.set_seed(config.prepare.seed)

    logger.info("Data loading")
    # Enable auto-logging to MLflow to capture TensorBoard metrics.

    with mlflow.start_run():

        mlflow.tensorflow.autolog(every_n_iter=1)
        train_params = {
            "hp_batch_size": config.hyperparameters.batch_size,
            "hp_repetitions": config.hyperparameters.repetitions,
            "hp_prefetch": config.hyperparameters.prefetch,
            "hp_augment": config.hyperparameters.augment,
            # "epochs": config.hyperparameters.epochs,
            # "loss_fn": config.hyperparameters.loss_fn,
            # "metric_fn": config.hyperparameters.metric_fn,
        }
        model_params = {
            "model_name": config.resnet.name,
            "model_repetitions": config.resnet.repetitions_block,
            "model_n_classes": config.resnet.n_classes,
            "model_img_shape": config.resnet.img_shape,
        }

        mlflow.log_params(train_params)
        mlflow.log_params(model_params)

        ds = create_dataset(
            Path(repo_path) / config.prepared_datas.train,
            config.hyperparameters.batch_size,
            config.hyperparameters.repetitions,
            config.hyperparameters.prefetch,
            config.hyperparameters.augment,
        )

        ds_val = create_dataset(
            Path(repo_path) / config.prepared_datas.val,
            config.hyperparameters.batch_size,
            config.hyperparameters.repetitions,
            config.hyperparameters.prefetch,
            config.hyperparameters.augment,
        )

        logger.info("Loading model")

        model = get_wide_resnet()

        logger.info("Compile Optimizer, Loss, Metrics")
        optim = {
            "rmsprop": tf.keras.optimizers.RMSprop(),
            "adam": tf.keras.optimizers.Adam(),
            "nadam": tf.keras.optimizers.Nadam(),
            "sgd": tf.keras.optimizers.SGD(),
        }
        optimizer = optim[config.hyperparameters.optimizer]

        metrics = []
        for metric in config.hyperparameters.metric_fn:
            metrics.append(metric)

        model.compile(
            optimizer=optimizer,
            loss=config.hyperparameters.loss_fn,
            metrics=metrics,
        )

        # print(f"{model.optimizer}, {model.loss}, {model.metrics}")

        K.set_value(
            model.optimizer.learning_rate, config.hyperparameters.learning_rate
        )

        # mlflow.log_params(model.optimizer.get_config())

        logger.info("Start training")
        model.fit(
            ds,
            epochs=config.hyperparameters.epochs,
            validation_data=ds_val,
            # callbacks=[MlFlowCallback()],
        )

        # mlflow.keras.log_model(model, "models")

        # logger.info("Training done, saving model")
        # model.save("model.h5")


if __name__ == "__main__":
    train()
