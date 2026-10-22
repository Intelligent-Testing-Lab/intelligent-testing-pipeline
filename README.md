### intelligent pipeline
An internal library to make pipeline development easy. Configurable using [hydra-core](), with [pypyr]() as the execution engine. Complete with examples `.yaml` scripts on how to define pipelines, with multiple stages and batched artifacts. Automatically handles logging, performance metrics, and supports custom completion verification scripts. It also supports Ray integration for distribution over clusters (WIP), SLURM (WIP) and utility scripts for use cases such as autonomous driving (WIP), and support for executing stages as docker containers (WIP). 

### Dependencies
- UV
- Python 3.10
- Ubuntu 24.04+

### Usage

Install the requirements (using uv):
```bash
uv venv --python 3.10
uv pip install -r requirements.txt
```


You can inherit from the `Stage` class to create your own pipeline stage. Override the `_run()` (required) and `verify_completion()` (optional) methods to implement your custom functionality.:

For example, here is a script that runs edge detection on an input video and verifies it wrote successfully.
```python
from stage import Stage
from omegaconf import DictConfig
from utils import get_frame_buffer, save_mp4
from pathlib import Path
import cv2

class EdgeDetector(Stage):

    def __init__(self, config: DictConfig) -> None:
        super().__init__(config, stage="edge")

    # Override this -> required
    def _run(self) -> None:
        # stage specific config is accessed through self.config.stage
        input_video = Path(self.config.stage.input_dir)
        video_buffer = get_frame_buffer(str(input_video.resolve()))

        # run edge detection...
        edge_buffer = cv2.Canny(...)

        # save video
        ouput_video = Path(self.config.stage.output_dir)
        save_mp4(edge_buffer, str(ouput_video.resolve()), fps=20.0)

```

To configure the stage, create a new config file in `configs/stage/edge.yaml` and add the parameters you want. These can be accessed from the class directly using `self.config.stage`. Global config is accessed through `self.config`. 

To run the script, add the class to the stage mapping in the entrypoint (`intelliget_pipeline.py`):
```python
from stages.edge_detector import EdgeDetector

STAGE_MAPPING: dict = {
    "edge": EdgeDetector
}
```

And run the `intelliget_pipeline.py` with the corresponding stage you want to execute.
```bash
python3 -m intelligent_pipeline stage=edge
```

You can also override any config parameters. For example, here I can modify the target of the edge detector:
```bash
> python3 -m intelligent_pipeline stage=edge stage.input_dir=/path/to/new/target

name: edge
input_dir: /path/to/new/target
output_dir: /data/edges
```
