import yaml
from utils import get_sorted_runs

params = yaml.safe_load(open("configs/params.yaml"))["mlflow"]
experiment_name = params["experiment_name"]


def evaluate():

    # repo_path = hydra.utils.get_original_cwd()

    # mlflow.set_tracking_uri(
    #     "file://" + hydra.utils.get_original_cwd() + "/mlruns"
    # )

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

# run_ids = [run["run_id"] for run in all_runs]

# print(f"{all_runs['run_id']}")
# best_run_id = all_runs.iloc[0].run_id
# best_run = mlflow.get_run(run_id=best_run_id)

# print(best_run)
