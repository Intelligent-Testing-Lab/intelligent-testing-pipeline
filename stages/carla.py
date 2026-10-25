from stage import Stage
from pathlib import Path
from omegaconf import DictConfig

import dataclasses
import logging
import yaml
import time
import socket
import psutil
import subprocess
import os


logger = logging.getLogger("pipeline.runner")

@dataclasses.dataclass
class CarlaOptions():
    run_cmd: str
    host: str
    quality: str

    # these are determined at run time - find a free port
    tm_port: int = 0
    streaming_port: int = 0
    world_port: int = 0
    pid: int = 0

    def build_launch_cmd(self) -> list:
        carla_process_cmd = ["bash", "-lc", f"exec {self.run_cmd} -RenderOffScreen -quality-level={self.quality} --world-port={self.world_port} --streaming-port={self.streaming_port}"]
        return carla_process_cmd

def get_carla_options(config: DictConfig) -> CarlaOptions:
    """
    Read carla_options.yaml from the output dir and return a CarlaOptions object to connect to a client
    """
    options_path = Path(config.run.output_dir) / "carla_options.yaml"

    if not options_path.exists():
        raise FileNotFoundError("Could not find carla_options.yaml")

    with open(options_path) as f:
        obj = yaml.safe_load(f)

    try:
        options = CarlaOptions(
            run_cmd="",
            host=obj["host"],
            quality="",
            world_port=obj["world_port"],
            tm_port=obj["tm_port"],
            streaming_port=obj["streaming_port"],
            pid=obj["pid"]
        )
    except KeyError as e:
        # we capture here to raise a more verbose error
        raise KeyError("Could not find valid carla option from keys (host, world_port, tm_port, streaming_port, pid). Check the keys are defined in output_dir/carla_options.yaml.")
    return options

def get_free_port(host: str, reserved_ports: list, port_min: int, port_max: int) -> int:
    for port in range(port_min, port_max):
        if port in reserved_ports:
            continue
        if is_port_bound(host, port):
            continue
        return port
    raise RuntimeError("Cannot find free socket for CARLA")

def is_port_bound(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
        except OSError:
            return True
        return False

def save_carla_options(config: DictConfig, options: CarlaOptions) -> None:
    """
    Save carla_options.yaml into the output dir
    """
    options_path = Path(config.run.output_dir) / "carla_options.yaml"

    if options_path.exists():
        raise FileExistsError(f"CARLA options already exist in {config.run.output_dir} - is CARLA running?")

    obj = {
        "host": options.host,
        "tm_port": options.tm_port,
        "world_port": options.world_port,
        "streaming_port": options.streaming_port,
        "pid": options.pid
    }

    with open(options_path, "w") as f:
        yaml.safe_dump(obj, f)

class CarlaLauncher(Stage):

    def __init__(self, config: DictConfig, stage: str = "") -> None:
        super().__init__(config, stage)

    def _run(self) -> None:
        """
        Launches a CARLA instance using subprocess, selecting random free ports at the time of launch. Saves
        the options (host, port, PID) as carla_options.yaml.

        Spawn this as a pipeline stage per item you want to process against CARLA - i.e. per scenario.
        """
        logger.info(self.config)

        reserved_ports = []
        for i in range(0,3):
            port = get_free_port(self.config.stage.host, reserved_ports, self.config.stage.port_min, self.config.stage.port_max)
            reserved_ports.append(port)

        options = CarlaOptions(
            run_cmd=self.config.stage.run_cmd,
            host=self.config.stage.host,
            quality=self.config.stage.quality,
            world_port=reserved_ports[0],
            tm_port=reserved_ports[1],
            streaming_port=reserved_ports[2]
        )

        carla_log = Path(self.config.run.output_dir) / "carla.log"

        with carla_log.open("wb") as log_file:
            print(options.build_launch_cmd())
            process = subprocess.Popen(
                options.build_launch_cmd(),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=log_file,
                start_new_session=True,
                close_fds=True,
            )
        self.pid = process.pid
        options.pid = self.pid

        time.sleep(10) # give carla time to launch

        save_carla_options(self.config, options)

    def verify_completion(self) -> bool:
        return psutil.pid_exists(self.pid)

class CarlaCleanup(Stage):

    def __init__(self, config: DictConfig, stage: str = "") -> None:
        super().__init__(config, stage)

    def _run(self) -> None:
        """
        Call terminate on the CARLA PID and wait for the process to finish.
        """
        self.options = get_carla_options(self.config)
        process = psutil.Process(self.options.pid)

        # clear the carla_options
        carla_options = Path(self.config.run.output_dir) / "carla_options.yaml"

        if carla_options.exists() and carla_options.is_file():
            os.remove(str(carla_options.resolve()))

        # check if streaming port is bound
        try:
            if process.is_running() and is_port_bound(self.options.host, self.options.streaming_port):
                process.terminate()
                process.wait(timeout=self.config.stage.timeout_wait) # wait 20 seconds for carla to cleanup
        except psutil.TimeoutExpired:
            logger.error(f"Could not gracefully terminate CARLA process {process.pid}. Attempting to force kill")
            process.kill() # raises an error - could not kill CARLA at all
            process.wait(timeout=self.config.stage.timeout_wait) # wait 20 seconds for carla to cleanup


    def verify_completion(self) -> bool:
        # if the port is free then CARLA has terminated - unless the port is used by another instance.
        # Need a way to check this - also verify process status?
        return not is_port_bound(self.options.host, self.options.streaming_port)
