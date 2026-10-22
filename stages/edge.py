from stage import Stage
from omegaconf import DictConfig, OmegaConf

class EdgeDetection(Stage):
    config_name = "edge"

    def __init__(self, config: DictConfig) -> None:
        super().__init__(config, stage="edge")

    def run(self) -> None:
        print(OmegaConf.to_yaml(self.config))
