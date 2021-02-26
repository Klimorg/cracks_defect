import hydra
from omegaconf import DictConfig
from utils import flatten_omegaconf


@hydra.main(config_path="../configs/", config_name="params.yaml")
def main(config: DictConfig):
    """[summary].

    Args:
        config (DictConfig): [description]
    """
    res = flatten_omegaconf(config)
    print(f"{res}")


if __name__ == "__main__":
    main()
