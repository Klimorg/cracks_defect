from pathlib import Path

import hydra
import mlflow
import tensorflow as tf
from loguru import logger
from mlflow import tensorflow as mltensorflow
from omegaconf import DictConfig, OmegaConf
from tensorize import Tensorize
from utils import config_to_hydra_dict, flatten_omegaconf, load_obj, set_seed

"""
https://stackoverflow.com/questions/59635474/
whats-difference-between-using-metrics-acc-and-tf-keras-metrics-accuracy

I'll just add that as of tf v2.2 in training.py the docs say
"When you pass the strings 'accuracy' or 'acc', we convert this to
one of tf.keras.metrics.BinaryAccuracy,
tf.keras.metrics.CategoricalAccuracy,
tf.keras.metrics.SparseCategoricalAccuracy based on the loss function
used and the model output shape. We do a similar conversion
for the strings 'crossentropy' and 'ce' as well."
"""


@logger.catch()
@hydra.main(config_path="../configs/", config_name="params.yaml")
def train(config: DictConfig) -> tf.keras.Model:
    """[summary].

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
    logger.add(f"logs_train_{config.log.timestamp}.log")

    logger.info(f"Training started at {config.log.timestamp}")
    conf_dict = config_to_hydra_dict(config)

    repo_path = hydra.utils.get_original_cwd()

    mlflow.set_tracking_uri("file://" + hydra.utils.get_original_cwd() + "/mlruns")
    mlflow.set_experiment(config.mlflow.experiment_name)

    logger.info(f"{OmegaConf.to_yaml(config)}")

    set_seed(config.prepare.seed)

    logger.info("Data loading")

    logger.info(f"Root path of the folder : {repo_path}")
    logger.info(f"MLFlow uri : {mlflow.get_tracking_uri()}")
    with mlflow.start_run(run_name=config.mlflow.run_name) as run:

        logger.info(f"Run infos : {run.info}")
        mltensorflow.autolog(every_n_iter=1)
        mlflow.log_params(flatten_omegaconf(config))

        ts = Tensorize(
            n_classes=config.datas.n_classes,
            img_shape=config.datasets.params.img_shape,
            random_seed=config.prepare.seed,
        )

        ds = ts.create_dataset(
            Path(repo_path) / config.datasets.prepared_datas.train,
            config.datasets.params.batch_size,
            config.datasets.params.repetitions,
            config.datasets.params.prefetch,
            config.datasets.params.augment,
        )

        ds_val = ts.create_dataset(
            Path(repo_path) / config.datasets.prepared_datas.val,
            config.datasets.params.batch_size,
            config.datasets.params.repetitions,
            config.datasets.params.prefetch,
            config.datasets.params.augment,
        )

        logger.info("Compiling model")

        cnn = load_obj(config.cnn.class_name)
        model = cnn(**conf_dict["cnn.params"])

        optim = load_obj(config.optimizer.class_name)
        optimizer = optim(**conf_dict["optimizer.params"])

        loss = load_obj(config.losses.class_name)
        loss = loss(**conf_dict["losses.params"])

        metric = load_obj(config.metrics.class_name)
        metric = metric()

        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=[metric],
        )

        logger.info("Start training")
        model.fit(
            ds,
            epochs=config.training.epochs,
            validation_data=ds_val,
        )


if __name__ == "__main__":
    train()
