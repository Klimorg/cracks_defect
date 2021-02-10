import collections
from collections import OrderedDict
import importlib
import os
import random
from pathlib import Path
from typing import Any, Dict, List
import mlflow  # type: ignore
import numpy as np
import tensorflow as tf  # type: ignore
import pandas as pd
from omegaconf import DictConfig, OmegaConf


# https://github.com/Erlemar/pytorch_tempest/blob/
# master/src/utils/technical_utils.py
def config_to_hydra_dict(cfg: DictConfig) -> Dict:
    """
    Convert config into dict with lists of values, where key is full name of
    parameter this function is used to get key names which can be used
    in hydra.
    Args:
        cfg:
    Returns:
        converted dict
    """
    experiment_dict = {}
    for k, v in cfg.items():
        for k1, v1 in v.items():
            experiment_dict[f"{k}.{k1}"] = v1

    return experiment_dict


# def load_object(object_path: Path, params: Dict) -> Any:

#     # for hp in hp_dict.keys():
#     #     optimizer.__setattr__(hp, hp_dict.get(hp))

#     p, m = object_path.rsplit(".", 1)

#     mod = importlib.import_module(p)
#     func = getattr(mod, m)

#     object = func(**params)

#     return object

# https://github.com/Erlemar/pytorch_tempest/blob/
# master/src/utils/technical_utils.py
def load_obj(obj_path: str, default_obj_path: str = "") -> Any:
    """
    Extract an object from a given path.
    https://github.com/quantumblacklabs/kedro/blob/
    9809bd7ca0556531fa4a2fc02d5b2dc26cf8fa97/kedro/utils.py
        Args:
            obj_path: Path to an object to be extracted, including the object
            name.
            default_obj_path: Default object path.
        Returns:
            Extracted object.
        Raises:
            AttributeError: When the object does not have the given named
            attribute.
    """
    obj_path_list = obj_path.rsplit(".", 1)
    # print(f"{obj_path_list}")
    obj_path = (
        obj_path_list.pop(0) if len(obj_path_list) > 1 else default_obj_path
    )
    # print(f"{obj_path}")
    obj_name = obj_path_list[0]
    # print(f"{obj_name}")
    module_obj = importlib.import_module(obj_path)
    if not hasattr(module_obj, obj_name):
        raise AttributeError(
            f"Object `{obj_name}` cannot be loaded from `{obj_path}`."
        )
    return getattr(module_obj, obj_name)


def set_seed(random_seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"


# https://github.com/Erlemar/pytorch_tempest/blob/
# master/src/utils/technical_utils.py
def flatten_omegaconf(d: Any, sep: str = "_") -> Dict[Any, str]:
    d = OmegaConf.to_container(d)

    obj = OrderedDict()

    def recurse(t, parent_key=""):

        if isinstance(t, list):
            for i, _ in enumerate(t):
                recurse(
                    t[i], parent_key + sep + str(i) if parent_key else str(i)
                )
        elif isinstance(t, dict):
            for k, v in t.items():
                recurse(v, parent_key + sep + k if parent_key else k)
        else:
            obj[parent_key] = t

    recurse(d)

    obj_txt = {
        k: v
        for k, v in obj.items()
        if isinstance(v, str) and not v.startswith("$")
    }
    obj_value = {k: v for k, v in obj.items() if isinstance(v, (int, float))}

    obj_txt.update(obj_value)

    res = dict(sorted(obj_txt.items()))
    res = {k: v for k, v in res.items()}

    return res


# https://github.com/GokuMohandas/applied-ml/blob/main/tagifai/utils.py
def get_sorted_runs(
    experiment_name: str, order_by: List, top_k: int = 10
) -> pd.DataFrame:
    """Get sorted list of runs from Experiment `experiment_name`.
    Usage:
    ```python
    runs = get_sorted_runs(experiment_name="best", order_by=["metrics.val_loss ASC"])
    ```
    Args:
        experiment_name (str): Name of the experiment to fetch runs from.
        order_by (List): List specification for how to order the runs.
    Returns:
        pd.DataFrame: Dataframe of ordered runs with their respective info.
    """
    # client = mlflow.tracking.MlflowClient()
    experiment_id = mlflow.get_experiment_by_name(
        experiment_name
    ).experiment_id
    runs_df = mlflow.search_runs(
        experiment_ids=experiment_id, order_by=order_by,
    )[:top_k]

    # Convert DataFrame to List[Dict]
    # runs = runs_df.to_dict("records")

    return runs_df
