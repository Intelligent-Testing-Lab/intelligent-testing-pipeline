"""Per-item shared context dict that flows between pipeline stages.

Generalizes the ``results.json`` read-modify-write pattern into one JSON dict per
item, keyed by ``run.context_file`` (default ``<output_dir>/pipeline_context.json``).
A stage reads via :func:`get` (or :attr:`Stage.ctx`) and writes via :func:`update` /
:func:`save`; step *N* writes, step *N+1* reads.

Because the dict lives on disk per item, it flows exactly like the batched items and
works under the Local / SLURM / Ray executors and across machines -- no in-memory
object to pass between processes. Use it for common data stages need to share, e.g.
``launch_carla`` stores ``carla: {host, port, pid}`` so later stages connect their
own ``carla.Client`` to the same server.
"""
from __future__ import annotations

import json
from pathlib import Path

from omegaconf import OmegaConf

DEFAULT_REL = "pipeline_context.json"


def _path(run) -> Path:
    cf = OmegaConf.select(run, "context_file")
    if not cf:
        outdir = OmegaConf.select(run, "output_dir")
        cf = str(Path(outdir) / DEFAULT_REL) if outdir else DEFAULT_REL
    return Path(cf)


def get(run) -> dict:
    """Return the item's context dict (``{}`` if it has not been written yet)."""
    p = _path(run)
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def update(run, **kv) -> dict:
    """Merge ``kv`` (top-level keys) into the dict, write it back, and return it."""
    data = get(run)
    data.update(kv)
    save(run, data)
    return data


def save(run, data: dict) -> None:
    """Write ``data`` as the item's context dict (parent dirs created)."""
    p = _path(run)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2))
