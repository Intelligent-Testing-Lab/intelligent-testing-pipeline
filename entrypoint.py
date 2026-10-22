import os
from omegaconf import DictConfig, OmegaConf
import hydra
from pathlib import Path
from stages.edge import EdgeDetection

PROJECT_ROOT = Path(os.getenv("COSMOS_ROOT", "/workspace/")) # defaults to the docker container root

STAGE_MAPPING: dict = {
    "edge": EdgeDetection
}

@hydra.main(
    config_path=str(
        (PROJECT_ROOT / "preprocessing" / "configs").resolve()
    ),
    config_name="config"
)

def main(cfg: DictConfig):
    stage = STAGE_MAPPING[cfg.stage.name]
    stage(cfg.stage).run()

if __name__ == '__main__':
    main()
