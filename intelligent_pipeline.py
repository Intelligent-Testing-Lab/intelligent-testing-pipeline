import importlib
import os
from pathlib import Path

import hydra
from omegaconf import DictConfig

PROJECT_ROOT = Path(os.getenv("COSMOS_ROOT", os.getcwd()))  # defaults to the docker container root

# Stage name -> (module path, class name).
#
# Stored as strings and resolved lazily (see ``load_stage``) so importing the
# entrypoint does *not* import every stage's dependencies. This is what allows
# each stage to run under its own Python interpreter / venv (driven by the
# pypyr pipeline) instead of forcing every dependency into one environment.
#
# NOTE: add a new entry here after implementing each stage.
STAGE_MAPPING: dict[str, tuple[str, str]] = {
    "edge": ("stages.edge", "EdgeDetection"),
}


def load_stage(name: str):
    """Lazily import and return the ``Stage`` class registered under ``name``."""
    try:
        module_name, class_name = STAGE_MAPPING[name]
    except KeyError as exc:
        known = ", ".join(sorted(STAGE_MAPPING))
        raise KeyError(f"Unknown stage {name!r}. Known stages: {known}") from exc
    return getattr(importlib.import_module(module_name), class_name)


@hydra.main(
    config_path=str((PROJECT_ROOT / "configs").resolve()),
    config_name="config",
)
def main(cfg: DictConfig):
    stage_cls = load_stage(cfg.stage.name)
    stage_cls(cfg).run()


if __name__ == "__main__":
    main()
