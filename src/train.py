from pathlib import Path

import hydra
import tensorflow as tf
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from tensorflow.keras import backend as K

from featurize import create_dataset
from model.resnet import get_resnet


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

    Args:
        config (DictConfig): [description]

    Returns:
        tf.keras.Model: [description]
    """
    logger.info(f"{OmegaConf.to_yaml(config)}")

    logger.info("Data loading")

    repo_path = hydra.utils.get_original_cwd()

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
    model.fit(ds, epochs=config.hyperparameters.epochs, validation_data=ds_val)

    logger.info("Training done, saving model")
    model.save("model.h5")


# pour changer le lr :
# https://stackoverflow.com/questions/59737875/keras-change-learning-rate

if __name__ == "__main__":
    train()
