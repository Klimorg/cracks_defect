import hydra
from omegaconf import DictConfig
from utils import config_to_hydra_dict, flatten_omegaconf


@hydra.main(config_path="../configs/", config_name="params.yaml")
def main(config: DictConfig):

    # print(f"{config}")
    res = flatten_omegaconf(config)
    print(f"{res}")


if __name__ == "__main__":
    main()
