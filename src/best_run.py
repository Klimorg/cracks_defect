from utils import get_sorted_runs
from tensorflow.keras.models import load_model
from featurize import featurize  # type: ignore
import hydra
import pandas as pd
import mlflow
from pathlib import Path

# params = yaml.safe_load(open("configs/params.yaml"))["mlflow"]

# experiment_name = params["experiment_name"]


@hydra.main(config_path="../configs/", config_name="params.yaml")
def evaluate(config):

    repo_path = hydra.utils.get_original_cwd()

    mlflow.set_tracking_uri(
        "file://" + hydra.utils.get_original_cwd() + "/mlruns"
    )

    all_runs = get_sorted_runs(
        experiment_name=config.mlflow.experiment_name,
        order_by=["metrics.val_loss ASC"],
    )

    print(
        all_runs[
            [
                "run_id",
                "tags.mlflow.runName",
                "metrics.val_categorical_accuracy",
                "metrics.val_loss",
            ]
        ]
    )

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

    recaps = []
    for run_id in all_runs["run_id"]:

        model_uri = (
            f"{repo_path}/mlruns/0/{run_id}/artifacts/model/data/model.h5"
        )
        print(f"loading model from run_id : {run_id}")
        model = load_model(model_uri)
        loss, acc = model.evaluate(ds_val)
        recap_run = {"run_id": run_id, "eval_loss": loss, "eval_acc": acc}
        recaps.append(recap_run)

    recap_df = pd.DataFrame(
        recaps, columns=["run_id", "eval_loss", "eval_acc"]
    )

    runs = all_runs.merge(recap_df, on=["run_id"], how="left", indicator=True)
    print(
        runs[
            [
                "run_id",
                "tags.mlflow.runName",
                "metrics.val_categorical_accuracy",
                "metrics.val_loss",
                "eval_loss",
                "eval_acc",
            ]
        ].sort_values("eval_loss")
    )


if __name__ == "__main__":
    evaluate()

# run_ids = [run["run_id"] for run in all_runs]

# print(f"{all_runs['run_id']}")
# best_run_id = all_runs.iloc[0].run_id
# best_run = mlflow.get_run(run_id=best_run_id)

# print(best_run)
