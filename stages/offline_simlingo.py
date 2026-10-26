from __future__ import annotations
from stage import Stage
from omegaconf import DictConfig
import importlib
from types import ModuleType
from typing import Union, Callable
import numpy as np
import sys

class OfflineSimlingoEval(Stage):

    def __init__(self, config: DictConfig, stage: str = "offline_simlingo") -> None:
        super().__init__(config, stage)

    def _predict(self, image: np.ndarray, speed: float, position: np.ndarray, heading: float, route: np.ndarray) -> dict:
        return self.agent.predict(
            image, speed=speed, position=position, heading=heading, route=route
        )

    def _run(self) -> None:
        sys.path.append(self.config.stage.pythonpath)
        importlib.invalidate_caches()

        OfflineLingoAgent = self._inject_dependency("team_code.offline_agent", "OfflineLingoAgent")

        self.agent = OfflineLingoAgent(                 # type: ignore
            checkpoint_path=self.config.stage.weights,
            eval_route_as="target_point",
            pretrained_cache_dir="pretrained",
            use_cot=None, # use COT, set False to disable,
            device="cuda"
        )

    def _inject_dependency(self, module_name, class_name: str="") -> Union[Callable, ModuleType]:
        """
        Load and return a module dependency. Uses the
        """
         # Let importlib through the import error
        module = importlib.import_module(module_name)
        return getattr(module, class_name) if class_name else module

    def verify_completion(self) -> bool:
        return super().verify_completion()
