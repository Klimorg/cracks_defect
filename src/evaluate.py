from pathlib import Path

import hydra
import mlflow
from featurize import featurize  # type: ignore
from loguru import logger
from omegaconf import DictConfig
from tensorflow.keras.models import load_model
from utils import set_seed


@hydra.main(config_path="../configs/", config_name="params.yaml")
def eval(config: DictConfig) -> None:

    set_seed(config.prepare.seed)
    repo_path = hydra.utils.get_original_cwd()

    mlflow.set_tracking_uri("file://" + hydra.utils.get_original_cwd() + "/mlruns")

    logger.info(f"Root path of the folder : {repo_path}")
    logger.info(f"MLFlow uri : {mlflow.get_tracking_uri()}")

    # artifact_url2 = "/home/vorph/work/cracks_defect/mlruns/0/84775389de534630b69779353b3dd7e0/artifacts"
    artifact_url1 = "/home/vorph/work/cracks_defect/mlruns/0/f85d7167e6624b92af6c90e73cc6dd10/artifacts"
    model_url = Path(artifact_url1) / Path("model/data/model.h5")

    model = load_model(model_url)

    ft = featurize(
        n_classes=config.datas.n_classes,
        img_shape=config.datas.img_shape,
        random_seed=config.prepare.seed,
    )

    ds_val = ft.create_dataset(
        Path(repo_path) / config.datasets.prepared_datas.val,
        config.datasets.params.batch_size,
        config.datasets.params.repetitions,
        config.datasets.params.prefetch,
        config.datasets.params.augment,
    )

    model.evaluate(ds_val)


if __name__ == "__main__":
    eval()
