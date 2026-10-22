from stage import Stage
from omegaconf import DictConfig
import time

class EdgeDetection(Stage):
    config_name = "edge"

    def __init__(self, config: DictConfig) -> None:
        super().__init__(config, stage="edge")

    def _run(self) -> None:
        time.sleep(1)
