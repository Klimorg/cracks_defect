from pathlib import Path

import hydra
import mlflow
import mlflow.tensorflow
import tensorflow as tf
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from tensorflow.keras import backend as K

from featurize import create_dataset
from model.resnet import get_resnet

mlflow.end_run()


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

    logger.info("MLFlow uris")
    print(f"{repo_path}")
    print(f"{mlflow.get_tracking_uri()}")
    # print(f"{mlflow.get_artifact_uri()}")

    logger.info(f"{OmegaConf.to_yaml(config)}")

    logger.info("Data loading")
    # Enable auto-logging to MLflow to capture TensorBoard metrics.
    mlflow.tensorflow.autolog(every_n_iter=1)

    with mlflow.start_run():

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

        model = get_resnet()

        logger.info("Setting hyperparapmeters")
        optim = {
            "rmsprop": tf.keras.optimizers.RMSprop(),
            "adam": tf.keras.optimizers.Adam(),
            "nadam": tf.keras.optimizers.Nadam(),
            "sgd": tf.keras.optimizers.SGD(),
        }
        optimizer = optim[config.hyperparameters.optimizer]

        model.compile(
            optimizer=optimizer,
            loss=config.hyperparameters.loss_fn,
            metrics=[config.hyperparameters.metric_fn],
        )

        K.set_value(
            model.optimizer.learning_rate, config.hyperparameters.learning_rate
        )

        logger.info("Start training")
        model.fit(
            ds, epochs=config.hyperparameters.epochs, validation_data=ds_val
        )

        logger.info("Training done, saving model")
        model.save("model.h5")


if __name__ == "__main__":
    train()
