Read the README.md for an overview of the project. Ignore any WIP features.

The important files and directories are:
- `stages/`: Custom implementation of the each of the pipelines stages
- `configs/`: Config directory
  - `configs/config.yaml`: Global config and defaults
  - `configs/stage`: Stage specific configuration. Must match Stage name defined in class.
- `intelligent_pipeline.py`: Library entrypoint, used to execute individual stages. After each stage implementation, make sure to add it to the STATE_MAPPING.

When you implement a feature, you should:
- Create a plan that will be reviewed by the developer. You should get his explicity approval before continuuing. Ask to clarify any queries you have with the instructions and requirements supplied.
- After each stage of the planned implementation, launch a code review sub-agent to:
  - Run ruff
  - Check for any code or logic errors
  - Verify the code implementation matches the requirements of the stage.
