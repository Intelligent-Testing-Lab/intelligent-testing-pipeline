# this class allows the lazy loading of specific libraries
# it also allows dyanmic configurations through hydra - changing the parameters of a single step
# is done through the corresponding config
#
import hydra
from pathlib import Path
from abc import ABC, abstractmethod
from omegaconf import DictConfig, OmegaConf
import os
import logging
import time

PROJECT_ROOT = os.getenv("COSMOS_ROOT", "/workspace/") # defaults to the docker container root

class PipeplineStageError(Exception):

    def __init__(self, message, name, time):
        self.message = f"Stage {name} failed after {time:.2f}. Traceback:\n{message}"
        super().__init__(self.message)

class Stage(object):
    config_name: str = "config"

    def __init__(self, config: DictConfig, stage: str = "") -> None:
        self.config = config
        self.stage = stage

    def run(self) -> None:
        name = type(self).__name__
        start_time = time.perf_counter()

        logger = logging.getLogger(name) # get the logger of the class being called

        logger.info("Starting stage: %s", name)

        try:

            self._run()
        except Exception as e:
            failure_time = time.perf_counter() - start_time
            logger.exception("Stage failed: %s after %.2fs", name, failure_time)
            raise PipeplineStageError(str(e), name, failure_time)
        else:
            completion_time = time.perf_counter() - start_time
            logger.info("Stage completed: %s in %.2fs", name, completion_time)

    @abstractmethod
    def _run(self) -> None:
        """Implement stage-specific work. Lazy load"""
        ...

    @abstractmethod
    def verify_completion(self) -> bool:
        """Implement stage-specific verification. Optional, always passes if unimplemented """
        return True
