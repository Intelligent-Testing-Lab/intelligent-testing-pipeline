from stage import Stage
from omegaconf import DictConfig, OmegaConf
import time
import logging

logger = logging.getLogger("pipeline.runner")

class EdgeDetection(Stage):
    config_name = "edge"

    def __init__(self, config: DictConfig) -> None:
        super().__init__(config, stage="edge")

    def _run(self) -> None:
        time.sleep(1)

        #logger.info(f"{OmegaConf.to_yaml(self.config)}")

        logger.info(f"{self.config.run.rgb_input}")
        logger.info(f"Writing to outputdir: {self.config.run.output_dir}")

        import carla
        from stages.carla import get_carla_options

        options = get_carla_options(self.config)

        client = carla.Client(options.host, options.world_port)
        logger.info(client.get_server_version())
