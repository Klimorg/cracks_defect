import yaml
from utils import get_sorted_runs

mlflow_config = yaml.safe_load(open("configs/params.yaml"))["mlflow"]
experiment_name = mlflow_config["experiment_name"]


def evaluate():

    all_runs = get_sorted_runs(
        experiment_name=experiment_name,
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

    best_run = all_runs.iloc[0]["run_id"]

    print(f"Best run id is : {best_run}")


if __name__ == "__main__":
    evaluate()
