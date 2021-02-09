import mlflow
import yaml

params = yaml.safe_load(open("configs/params.yaml"))["mlflow"]

experiment_name = params["experiment_name"]

client = mlflow.tracking.MlflowClient()
experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
all_runs = mlflow.search_runs(
    experiment_ids=experiment_id, order_by=["metrics.val_loss ASC"]
)[:10]

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


# best_run_id = all_runs.iloc[0].run_id
# best_run = mlflow.get_run(run_id=best_run_id)

# print(best_run)
