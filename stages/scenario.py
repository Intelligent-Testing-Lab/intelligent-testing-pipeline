from stage import Stage
from omegaconf import DictConfig
import sys
from typing import Callable, Union
from types import ModuleType
import importlib
import logging
import yaml
from pathlib import Path
from stages.carla import get_carla_options
import os

logger = logging.getLogger("pipeline.runner")

def load_yaml(path: Path | str) -> dict:
    """
    Loads data from a YAML file specified
    """
    if not isinstance(path, Path):
        path = Path(path)

    if not path.exists():
        logger.error("Path does not exist")
        raise FileNotFoundError(f"Path {str(path)} does not exist. Could not load YAML file.")
    try:
        with open(path, "r") as f:
            data = yaml.safe_load(f)
            return data
    except yaml.YAMLError:
        raise RuntimeError(f"Could not load YAML from {str(path)}. Is it valid YAML?")

class Scenario(Stage):

    def __init__(self, config: DictConfig, stage: str = "scenario") -> None:
        super().__init__(config, stage)

    def _build_options(self, config: DictConfig) -> dict:
        return {
            "world": config.stage.world,
            "timestep": config.stage.timestep,
            "tick_budget": config.stage.tick_budget,
            "segment_pool_size": config.stage.segment_pool_size,
            "route_length_m": config.stage.route_length_m,
            "expert_config": config.stage.expert_config,
            "npc_mode": config.stage.npc_mode,
            "pid_gains": config.stage.pid_gains,
            "debug": config.stage.debug,
            "seed": config.stage.seed,
        }

    def _test_dependency(self, libary: str) -> bool:
        try:
            self._inject_dependency(libary)
            return True
        except ImportError:
            logger.error(f"Could not inject library: {libary} into current runner runtime.")
            return False

    def _expert_env(
        self,
        project_root: str | Path,
        carla_root: str | Path | None = None,
    ) -> dict[str, str]:
        """
        Build environment variables for running the PDM-Lite expert integration.

        Args:
            project_root: Project directory containing `simulator/carla_garage`.
            carla_root: CARLA installation root. Defaults to $CARLA_ROOT or
                `/home/davidg/data/carla/0.9.15`.
            python_executable: Python interpreter to use in the subprocess.
                Defaults to $PYTHON or `python3`.

        Returns:
            A dict of environment variables for `subprocess.run(..., env=...)`.
        """
        project_root = Path(project_root).expanduser().resolve()
        garage = project_root / "simulator" / "carla_garage"
        team_code = garage / "team_code"

        if not team_code.is_dir():
            raise FileNotFoundError(
                f"Garage not found: expected {team_code}. "
                "Check that `project_root` points to the repository root."
            )

        carla_root = Path(
            carla_root
            or os.environ.get("CARLA_ROOT")
            or "/home/davidg/data/carla/0.9.15"
        ).expanduser().resolve()

        carla_pythonapi = carla_root / "PythonAPI" / "carla"
        carla_pythonapi_parent = carla_root / "PythonAPI"

        # Exact trees only: avoid plain leaderboard/ and scenario_runner/ name collisions.
        garage_paths = [
            carla_pythonapi,
            carla_pythonapi_parent,
            garage / "scenario_runner_autopilot",
            garage / "leaderboard_autopilot",
            garage / "team_code",
        ]

        for p in garage_paths:
            sys.path.append(str(p.resolve())) # append to current python path

        env = os.environ.copy()
        env.update(
            {
                "CARLA_ROOT": str(carla_root),
                "CARLA_PYTHONAPI": str(carla_pythonapi),
            }
        )

        os.environ = env
        return env

    def _validate_genotype(self, genotype) -> bool:
        GenoTypeValidator = self._inject_dependency("genotypes.scenario", "GenoTypeValidator")

        bounds = load_yaml(self.config.stage.gene_bounds)

        validator = GenoTypeValidator()
        validator.load_bounds(bounds)
        for result in validator.validate(genotype):
            logger.info(f"   {result}")
        valid = validator.is_valid(genotype)
        logger.info(f"    overall is_valid = {valid}")
        return valid

    def _run(self) -> None:
        # extend python path (interpreter) with our custom scenario library
        # invalidate caches to allow importlib to see the module
        sys.path.append(self.config.stage.pythonpath)
        importlib.invalidate_caches()
        self._expert_env(self.config.stage.pythonpath, self.config.stage.carla_root)

        # load the libraries we need to build the genotype
        GeneFactory = self._inject_dependency("genotypes.gene", "GeneFactory")
        GenoType = self._inject_dependency("genotypes.scenario", "GenoType")

        config = load_yaml(self.config.stage.gene_path) # change this to target gene from in memory context
        genotype = GeneFactory().build_gene(GenoType, config)

        if not self._validate_genotype(genotype):
            raise SystemError(f"Provided Gene {self.config.stage.gene_path} is not valid, checked against {self.config.stage.gene_bounds}")

        options = self._build_options(self.config)

        carla_options = get_carla_options(self.config)
        options["host"] = carla_options.host
        options["port"] = carla_options.world_port
        options["sensor_config"] = self.config.stage.oracle
        options["timeout"] = 30

        # assemble things needed for env
        os.environ["DATAGEN"] = "1"
        os.environ["SAVE_PATH"] = self.config.run.output_dir
        os.environ["TOWN"] = options["world"].upper()
        os.environ["REPETITION"] = "1"
        os.environ["REPETITIONS"] = "1"
        os.environ["CHALLENGE_TRACK_CODENAME"] = "MAP"
        os.environ["ROUTES"] = ""

        ScenarioDecoder = self._inject_dependency("decoding.carla.decoder", "ScenarioDecoder")
        decoder = ScenarioDecoder(options)
        scenario = decoder.decode(genotype)

        # execute the main scenario loop
        try:
            for _ in range(scenario.tick_budget):
                scenario.tick()

            ts = scenario.world.get_snapshot().timestamp
            ego_tf = scenario.ego.get_transform()
            ego_v = scenario.ego.get_velocity()
            speed = (ego_v.x ** 2 + ego_v.y ** 2 + ego_v.z ** 2) ** 0.5
            logger.info(f"\nRan {scenario.frame} ticks (game time {ts.elapsed_seconds:.3f}s); "
                  f"ego at ({ego_tf.location.x:.1f}, {ego_tf.location.y:.1f}) moving {speed:.2f} m/s")

            collisions = scenario.check_npc_npc_collisions()
            logger.info(f"\nNPC-NPC collision pairs: {collisions if collisions else 'none'}")
            for v in scenario.background:   # pid mode: the control history the expert now reads via get_control()
                if v.actions:
                    th = [a[0] for a in v.actions]
                    br = [a[1] for a in v.actions]
                    st = [a[2] for a in v.actions]
                    logger.info(f"  {v.name}: {len(v.actions)} controls | throttle mean {sum(th)/len(th):.2f} / max {max(th):.2f} | "
                          f"brake mean {sum(br)/len(br):.2f} / max {max(br):.2f} | |steer| max {max(abs(s) for s in st):.2f}")
            logger.info(f"Transfer prompt: {scenario.get_prompt()!r}")
        finally:
            scenario.cleanup()

    def _inject_dependency(self, module_name, class_name: str="") -> Union[Callable, ModuleType]:
        """
        Load and return a module dependency. Uses the
        """
         # Let importlib through the import error
        module = importlib.import_module(module_name)
        return getattr(module, class_name) if class_name else module
