import logging
import time
from abc import ABC, abstractmethod

from omegaconf import DictConfig, OmegaConf


class PipelineStageError(Exception):
    """Raised when a stage fails to run or fails its completion verification."""

    def __init__(self, message, name, time):
        self.message = f"Stage {name} failed after {time:.2f}. Traceback:\n{message}"
        super().__init__(self.message)


class Stage(ABC):
    """Base class for a pipeline stage.

    Subclasses must implement :meth:`_run`. Optionally override
    :meth:`verify_completion` to assert that the stage produced its expected
    artifacts; it is invoked by :meth:`run` after a successful ``_run``.
    """

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
            self.logger.info(
                "Loaded Config (with overrides) for %s:\n%s",
                self.name,
                OmegaConf.to_yaml(self.config.stage),
            )

        try:
            self._run()
        except Exception as e:  # noqa: BLE001 - re-raised as a stage error
            failure_time = time.perf_counter() - start_time
            self.logger.exception("Stage failed: %s after %.2fs", self.name, failure_time)
            raise PipelineStageError(str(e), self.name, failure_time) from e
        else:
            if not self.verify_completion():
                failure_time = time.perf_counter() - start_time
                self.logger.error(
                    "Completion verification failed for stage %s after %.2fs",
                    self.name,
                    failure_time,
                )
                raise PipelineStageError(
                    "Completion verification failed", self.name, failure_time
                )
            completion_time = time.perf_counter() - start_time
            self.logger.info("Stage completed: %s in %.2fs", self.name, completion_time)

    @abstractmethod
    def _run(self) -> None:
        """Implement stage-specific work. Required."""

    def verify_completion(self) -> bool:
        """Optional. Return ``False`` if the stage did not complete correctly."""
        return True
