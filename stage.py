from abc import ABC, abstractmethod
from omegaconf import DictConfig, OmegaConf
import logging
import time

class PipeplineStageError(Exception):

    def __init__(self, message, name, time):
        self.message = f"Stage {name} failed after {time:.2f}. Traceback:\n{message}"
        super().__init__(self.message)

class Stage(ABC):
    config_name: str = "config"

    def __init__(self, config: DictConfig, stage: str = "") -> None:
        self.config = config
        self.stage = stage
        self.name = type(self).__name__
        self.logger = logging.getLogger(self.name)

    def run(self) -> None:
        start_time = time.perf_counter()

        self.logger.info("Starting stage: %s", self.name)

        if self.config.debug:
            self.logger.info("Loaded Config for (with overrides) for %s:\n%s", self.name, OmegaConf.to_yaml(self.config.stage))

        try:

            self._run()
        except Exception as e:
            failure_time = time.perf_counter() - start_time
            self.logger.exception("Stage failed: %s after %.2fs", self.name, failure_time)
            raise PipeplineStageError(str(e), self.name, failure_time)
        else:
            completion_time = time.perf_counter() - start_time
            self.logger.info("Stage completed: %s in %.2fs", self.name, completion_time)

    @abstractmethod
    def _run(self) -> None:
        """Implement stage-specific work. Lazy load"""
        ...

    @abstractmethod
    def verify_completion(self) -> bool:
        """Implement stage-specific verification. Optional, always passes if unimplemented """
        return True
