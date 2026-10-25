"""Run a single stage in-process (no subprocess), given hydra-style argv.

This is the unit of work a :mod:`ray_executor` actor/task executes. It composes
the stage's config the same way the Hydra entrypoint would -- base ``config.yaml``
merged with the stage's ``configs/stage/<name>.yaml`` plus the per-run overrides
(``run.*``, ``batch.*``) -- then instantiates the registered stage and runs it.

It is deliberately independent of Ray (Ray is imported lazily by the executor),
so the in-process execution path can be unit-tested without Ray installed.
"""
from __future__ import annotations

import os
from pathlib import Path


def _root(project_root) -> Path:
    return Path(project_root) if project_root else Path(os.getenv("COSMOS_ROOT", os.getcwd()))


def parse_argv(argv: list[str]) -> tuple[str | None, list[tuple[str, str]]]:
    """Split hydra-style ``key=value`` argv into (stage_name, [(dotted_key, value)]).

    ``stage=<name>`` selects the stage; everything else is an override applied to
    the composed config. ``+key``/``-key`` prefixes are stripped.
    """
    stage_name: str | None = None
    overrides: list[tuple[str, str]] = []
    for tok in argv:
        if "=" not in tok:
            continue
        key, _, value = tok.partition("=")
        key = key.lstrip("+-")
        if key == "stage":
            stage_name = value
        else:
            overrides.append((key, value))
    return stage_name, overrides


def build_stage_config(stage_name: str, overrides: list[tuple[str, str]], project_root=None):
    """Compose a stage's DictConfig: base config + its stage config + overrides.

    Override values are stored as strings, matching Hydra CLI semantics
    (``run.batch_size=8`` -> ``"8"`` on an untyped key); stages cast what they read
    (e.g. ``int(cfg.x)``), so this is consistent with the subprocess path.
    """
    from omegaconf import OmegaConf

    cfg_dir = _root(project_root) / "configs"
    base = OmegaConf.load(cfg_dir / "config.yaml")
    stage_cfg = OmegaConf.load(cfg_dir / "stage" / f"{stage_name}.yaml")
    config = OmegaConf.merge(base, {"stage": stage_cfg})
    for key, value in overrides:
        OmegaConf.update(config, key, value, merge=False)
    return config


def run_stage_in_process(argv: list[str], project_root=None) -> None:
    """Compose the config for ``argv``'s stage and run it in this process."""
    from intelligent_pipeline import load_stage

    stage_name, overrides = parse_argv(argv)
    if not stage_name:
        raise ValueError(f"no 'stage=' override in argv: {argv!r}")
    config = build_stage_config(stage_name, overrides, project_root)
    stage_cls = load_stage(stage_name)
    stage_cls(config).run()
