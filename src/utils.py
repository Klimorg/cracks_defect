from pathlib import Path
from typing import Any, Dict
from omegaconf import DictConfig, OmegaConf
import os
import random
import tensorflow as tf
import numpy as np
import importlib
import collections

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


def load_object(object_path: Path, params: Dict) -> Any:

    # for hp in hp_dict.keys():
    #     optimizer.__setattr__(hp, hp_dict.get(hp))

    p, m = object_path.rsplit(".", 1)

    mod = importlib.import_module(p)
    func = getattr(mod, m)

    object = func(**params)

    return object


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


def flatten_omegaconf(d, sep="_"):
    d = OmegaConf.to_container(d)

    obj = collections.OrderedDict()

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
    obj = {k: v for k, v in obj.items() if isinstance(v, (int, float))}
    # obj = {k: v for k, v in obj.items()}

    return obj
