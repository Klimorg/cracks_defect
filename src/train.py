from pathlib import Path

import hydra
import mlflow
import tensorflow as tf
from loguru import logger
from mlflow import tensorflow as mltensorflow
from omegaconf import DictConfig
from tensorize import Tensorize
from utils import flatten_omegaconf, load_obj, set_log_infos, set_seed


@logger.catch()
@hydra.main(config_path="../configs/", config_name="params.yaml")
def train(config: DictConfig):
    """Train loop of the classification model.

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

    https://stackoverflow.com/questions/59635474/
    whats-difference-between-using-metrics-acc-and-tf-keras-metrics-accuracy

    I'll just add that as of tf v2.2 in training.py the docs say
    "When you pass the strings 'accuracy' or 'acc', we convert this to
    one of tf.keras.metrics.BinaryAccuracy,
    tf.keras.metrics.CategoricalAccuracy,
    tf.keras.metrics.SparseCategoricalAccuracy based on the loss function
    used and the model output shape. We do a similar conversion
    for the strings 'crossentropy' and 'ce' as well."

    Args:
        config (DictConfig): [description]
    """
    conf_dict, repo_path = set_log_infos(config)

    logger.info("Setting training policy.")
    policy = tf.keras.mixed_precision.experimental.Policy("mixed_float16")
    tf.keras.mixed_precision.experimental.set_policy(policy)
    logger.info(f"Compute dtype : {policy.compute_dtype}")
    logger.info(f"Variable dtype : {policy.variable_dtype}")

    mlflow.set_tracking_uri(f"file://{repo_path}/mlruns")
    mlflow.set_experiment(config.mlflow.experiment_name)

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
            Path(repo_path) / config.datasets.prepared_dataset.train,
            config.datasets.params.batch_size,
            config.datasets.params.repetitions,
            config.datasets.params.prefetch,
            config.datasets.params.augment,
        )

        ds_val = ts.create_dataset(
            Path(repo_path) / config.datasets.prepared_dataset.val,
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
