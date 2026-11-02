"""CMA-ME quality-diversity mutation stage (pyribs).

Runs a CMA-ME search (CMA-ES emitter + :class:`~ribs.archives.GridArchive`) over a
subset of the scenario genotype's *numeric* genes, each normalised to ``[0, 1]`` using
the gene bounds. Each candidate pyribs proposes is materialised as a gene YAML and
evaluated by dispatching the existing pipeline stages
(``launch_carla -> scenario -> cleanup_carla -> offline_simlingo``) as subprocesses in
their own venvs -- the exact launch contract ``LocalExecutor`` uses. The objective
(steering MSE, maximised) and the behaviour characteristics are read back from disk and
returned to the archive via :class:`~ribs.schedulers.Scheduler.tell`.

The CMA-ME state stays in-process: this stage runs in the ``intelligent_pipeline`` venv,
where ``ribs`` is installed. Per-candidate evaluation is dispatched as subprocesses so
``scenario`` / ``offline_simlingo`` run under their own interpreters.

Developer extension point (intentionally left for the developer to fill in):

* :meth:`CmaMeStage._characteristics` -- returns the 4-vector of behaviour characteristics
  (it can reuse :meth:`CmaMeStage._per_frame_steering` to derive ``argmax_t err[t]``).

:meth:`CmaMeStage._fitness` (steering MSE) and :meth:`CmaMeStage._build_emitters` (default CMA-ME
emitter) are implemented and can be overridden.
"""
from __future__ import annotations

import copy
import csv
import gzip
import json
import logging
import re
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from omegaconf import DictConfig

from ribs import archives, emitters, schedulers
from executors import LocalExecutor, WorkUnit
from pipeline import StageDef
from stage import Stage

logger = logging.getLogger("pipeline.runner")


def _load_yaml(path: str | Path) -> dict:
    """Load a YAML file into a dict (``{}`` for an empty file)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Could not load YAML from {path} (does not exist).")
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data if data is not None else {}


class CmaMeStage(Stage):
    """CMA-ME quality-diversity mutation of the numeric scenario genes."""

    def __init__(self, config: DictConfig, stage: str = "cma_me") -> None:
        super().__init__(config, stage)
        # Lazy caches (populated on first use).
        self._base_gene: dict | None = None
        self._bounds: dict | None = None
        self._active_genes: list[str] | None = None
        self._archive: archives.GridArchive | None = None
        self._convergence: list[dict] = []

    # ------------------------------------------------------------------ #
    # Config + gene access
    # ------------------------------------------------------------------ #
    @property
    def _c(self) -> DictConfig:
        return self.config.stage

    def _base(self) -> dict:
        if self._base_gene is None:
            self._base_gene = _load_yaml(self._c.gene_path)
        return self._base_gene

    def _bounds_dict(self) -> dict:
        if self._bounds is None:
            self._bounds = _load_yaml(self._c.gene_bounds)
        return self._bounds

    # ------------------------------------------------------------------ #
    # Nested-dict helpers (genes are dotted keys over nested YAML)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _get_nested(d: dict, dotted: str) -> Any:
        cur: Any = d
        for part in dotted.split("."):
            cur = cur[part]
        return cur

    @staticmethod
    def _set_nested(d: dict, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        cur: Any = d
        for p in parts[:-1]:
            cur = cur[p]
        cur[parts[-1]] = value

    # ------------------------------------------------------------------ #
    # Active target genes (the dynamic solution dimension)
    # ------------------------------------------------------------------ #
    def active_genes(self) -> list[str]:
        """The ``target_genes`` that are actually mutable for this base genotype.

        A ``background_i.*`` param is active only when ``background_i.present`` is truthy;
        ``background_i.lane_change_onset_frame`` additionally requires ``lane_change`` to be
        truthy. ``present`` / ``lane_change`` are static and are never mutated themselves.
        """
        if self._active_genes is None:
            base = self._base()
            active: list[str] = []
            for key in self._c.target_genes:
                section, attr = key.rsplit(".", 1)
                if section.startswith("background"):
                    if not self._truthy(self._get_nested(base, f"{section}.present")):
                        continue
                    if attr == "lane_change_onset_frame":
                        if not self._truthy(self._get_nested(base, f"{section}.lane_change")):
                            continue
                active.append(key)
            self._active_genes = active
        return self._active_genes

    @staticmethod
    def _truthy(v: Any) -> bool:
        return bool(int(v))

    # ------------------------------------------------------------------ #
    # Normalisation (bound <-> [0, 1])
    # ------------------------------------------------------------------ #
    def _bounds_for(self, gene: str) -> tuple[float, float]:
        """Return the numeric ``[low, high]`` bound for ``gene``.

        Bounds may be shared across a family of genes (e.g. the bounds file groups the
        three background vehicles under one ``background`` section, while the gene file
        has ``background_1/2/3``). So we fall back from ``section.attr`` to the
        ``<section without trailing _N>.attr`` form.
        """
        bounds = self._bounds_dict()
        section, attr = gene.rsplit(".", 1)
        candidates = [gene]
        shared = re.sub(r"_\d+$", "", section)
        if shared != section:
            candidates.append(f"{shared}.{attr}")

        for cand in candidates:
            try:
                b = self._get_nested(bounds, cand)
            except (KeyError, TypeError):
                continue
            lo, hi = float(b[0]), float(b[1])
            if hi <= lo:
                raise ValueError(f"Invalid bound for {gene!r}: {b!r} (need upper > lower).")
            return lo, hi
        raise KeyError(f"No numeric bound found for gene {gene!r}; tried {candidates}.")

    @staticmethod
    def _normalize(v: float, lo: float, hi: float) -> float:
        return float(np.clip((v - lo) / (hi - lo), 0.0, 1.0))

    @staticmethod
    def _denormalize(x: float, lo: float, hi: float) -> float:
        return float(lo + x * (hi - lo))

    def _x0(self) -> np.ndarray:
        """Initial solution in the normalised ``[0, 1]^n`` space (from the base gene)."""
        base = self._base()
        active = self.active_genes()
        x0 = np.empty(len(active), dtype=np.float64)
        for i, gene in enumerate(active):
            lo, hi = self._bounds_for(gene)
            x0[i] = self._normalize(float(self._get_nested(base, gene)), lo, hi)
        return x0

    def _build_gene(self, x: np.ndarray) -> dict:
        """Denormalise ``x`` into a full gene dict (base gene with the active genes set)."""
        gene = copy.deepcopy(self._base())
        active = self.active_genes()
        for i, g in enumerate(active):
            lo, hi = self._bounds_for(g)
            self._set_nested(gene, g, self._denormalize(float(x[i]), lo, hi))
        return gene

    # ------------------------------------------------------------------ #
    # pyribs objects
    # ------------------------------------------------------------------ #
    def _build_archive(self, n: int) -> archives.GridArchive:
        dims = [int(d) for d in self._c.measure_dims]
        ranges = [(float(lo), float(hi)) for lo, hi in self._c.measure_ranges]
        if len(dims) != len(ranges):
            raise ValueError(
                f"measure_dims ({len(dims)}) and measure_ranges ({len(ranges)}) must be the same length."
            )
        return archives.GridArchive(
            solution_dim=n,
            dims=dims,
            ranges=ranges,
            seed=int(self._c.seed),
        )

    def _build_emitters(self, archive: archives.GridArchive, n: int) -> list:
        """Build the emitter list.

        Override this method to program your own emitters (pyribs emitter classes).
        The default is a single CMA-ME emitter: CMA-ES (``es="cma_es"``) with the
        ``"filter"`` selection rule, which drives adaptation from archive insertions.
        """
        return [
            emitters.EvolutionStrategyEmitter(
                archive,
                x0=self._x0(),
                sigma0=float(self._c.sigma0),
                es="cma_es",
                selection_rule="filter",
                bounds=[(0.0, 1.0)] * n,
                batch_size=(int(self._c.batch_size) if self._c.batch_size is not None else None),
                seed=int(self._c.seed),
            )
        ]

    def _default_measure(self) -> np.ndarray:
        """A guaranteed in-range measure point (used to fill in a failed candidate)."""
        return np.array([0.5 * (lo + hi) for lo, hi in self._c.measure_ranges], dtype=np.float64)

    # ------------------------------------------------------------------ #
    # Stage interface
    # ------------------------------------------------------------------ #
    def _run(self) -> None:
        n = len(self.active_genes())
        n_measure = len(list(self._c.measure_ranges))
        self.logger.info(
            "CMA-ME: %d active gene(s) of %d candidate(s); measure_dim=%d; max_evaluations=%d",
            n, len(list(self._c.target_genes)), n_measure, int(self._c.max_evaluations),
        )
        archive = self._build_archive(n)
        emitters = self._build_emitters(archive, n)
        self._archive = archive
        scheduler = schedulers.Scheduler(archive, emitters)
        scope = str(self._c.carla_scope)
        if scope != "per_candidate":
            self.logger.warning("carla_scope=%r is not implemented yet; falling back to per-candidate evaluation.", scope)
        self._init_run_csvs()
        self._run_loop(scheduler)
        self._write_outputs(archive)

    def _run_loop(self, scheduler: schedulers.Scheduler) -> None:
        max_evals = int(self._c.max_evaluations)
        sentinel = float(self._c.sentinel_objective)
        total = 0
        round_idx = 0
        while total < max_evals:
            sols = scheduler.ask()
            if sols.size == 0:
                self.logger.warning("scheduler.ask() returned no solutions; stopping.")
                break
            objectives, measures = self._run_candidates(sols, sentinel, round_idx)
            scheduler.tell(objective=objectives, measures=measures)
            total += int(len(sols))
            round_idx += 1
            stats = scheduler.archive.stats
            self.logger.info(
                "Round %d: evaluated %d (total %d/%d) | elites=%d | obj_max=%s | qd=%s",
                round_idx, int(len(sols)), total, max_evals, stats.num_elites, stats.obj_max, stats.qd_score,
            )
            self._convergence.append({
                "round": round_idx,
                "evaluated_total": total,
                "num_elites": int(stats.num_elites),
                "obj_max": float(stats.obj_max),
                "obj_mean": float(stats.obj_mean),
                "coverage": float(stats.coverage),
                "qd_score": float(stats.qd_score),
                "norm_qd_score": float(stats.norm_qd_score),
            })
            self._csv_append(self._out_dir() / "cma_me_convergence.csv", [
                round_idx, total, int(stats.num_elites),
                float(stats.obj_max), float(stats.obj_mean),
                float(stats.coverage), float(stats.qd_score), float(stats.norm_qd_score),
            ])

    def _run_candidates(
        self, sols: np.ndarray, sentinel: float, round_idx: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate every candidate in ``sols`` sequentially and return (objective, measures).

        This is the single sequential->parallel pivot seam: the ``for`` loop below is the
        only place that would be swapped for a pool over N free CARLA ports. ``tell`` must
        receive one (objective, measure) per candidate in the same order, so a failed
        candidate is filled with the sentinel objective and an in-range default measure.
        """
        objectives = np.full(len(sols), sentinel, dtype=np.float64)
        measures = np.tile(self._default_measure(), (len(sols), 1))
        for i, sol in enumerate(sols):
            candidate_dir = self._candidate_dir(round_idx, i)
            success = False
            try:
                gene = self._build_gene(sol)
                self._evaluate_candidate(candidate_dir, gene)
                objectives[i] = self._fitness(candidate_dir)
                measures[i] = self._characteristics(candidate_dir, gene)
                success = True
            except Exception:
                self.logger.exception("Candidate r%d_%d failed; recording sentinel objective.", round_idx, i)
                objectives[i] = sentinel
                # objectives[i] / measures[i] keep their sentinel / default values.
            row = [round_idx, i, f"r{round_idx}_{i}", success, float(objectives[i])]
            row += [float(v) for v in measures[i]]
            self._csv_append(self._out_dir() / "cma_me_candidates.csv", row)
        return objectives, measures

    def _candidate_dir(self, round_idx: int, i: int) -> Path:
        return self._out_dir() / "candidates" / f"r{round_idx}_{i}"

    def _evaluate_candidate(self, candidate_dir: Path, gene: dict) -> None:
        """Materialise the candidate gene and run the evaluation recipe in order.

        Each recipe step is launched as a subprocess in its own venv (the ``LocalExecutor``
        contract). Per-step dynamic overrides are injected here:
        * ``scenario``         -> ``stage.gene_path``  = the candidate gene YAML
        * ``offline_simlingo`` -> ``stage.input_dir``  = the candidate dir (its recorded data)
        ``launch_carla`` / ``cleanup_carla`` operate on ``run.output_dir`` (= candidate dir).

        Assumes the scenario records its RGB/measurements under ``run.output_dir`` (via its
        ``SAVE_PATH``), which ``offline_simlingo`` then reads from ``stage.input_dir``.
        """
        candidate_dir.mkdir(parents=True, exist_ok=True)
        gene_path = candidate_dir / "candidate_gene.yaml"
        self._write_gene_yaml(gene_path, gene)

        for step in self._c.evaluation_recipe:
            stage = str(step["stage"])
            interpreter = str(step["interpreter"])
            overrides = list(step.get("overrides", []) or [])
            if stage == "scenario":
                overrides.append(f"stage.gene_path={gene_path}")
            if stage == "offline_simlingo":
                overrides.append(f"stage.input_dir={candidate_dir}")
            self._spawn_stage(stage, str(candidate_dir), interpreter, overrides)

    def _write_gene_yaml(self, path: Path, gene: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(gene, f, sort_keys=False)

    def _spawn_stage(self, stage: str, output_dir: str, interpreter: str, overrides: list[str]) -> None:
        """Launch one pipeline stage as a subprocess in the given interpreter's venv.

        Mirrors the ``LocalExecutor`` launch contract: ``<interpreter> -m intelligent_pipeline
        stage=<stage> run.output_dir=<output_dir> <overrides...>``. Raises on non-zero exit.
        """
        argv = [f"stage={stage}", f"run.output_dir={output_dir}"] + list(overrides)
        unit = WorkUnit(name=f"cma_me_{stage}", interpreter=interpreter, argv=argv)
        results = LocalExecutor().run_units(StageDef(stage=stage), [unit])
        r = results[0]
        if not r.ok:
            raise RuntimeError(
                f"Stage {stage!r} failed (interpreter={interpreter}); rc!=0.\n"
                f"STDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
            )

    def _out_dir(self) -> Path:
        """The stage's output directory (``<output_dir>/cma_me``) for archives, CSVs and elites."""
        return Path(self.config.run.output_dir) / "cma_me"

    def _init_run_csvs(self) -> None:
        """Create the run CSV logs (headers only) before the search starts.

        ``cma_me_candidates.csv`` -- one row per evaluated candidate (objective + the
        configured characteristics; failed candidates keep their sentinel/default values
        and are flagged ``success=False``).
        ``cma_me_convergence.csv`` -- one row per round with the archive's quality/
        diversity stats (elites, obj max/mean, coverage, QD score).
        """
        char_names = [str(c) for c in self._c.characteristics]
        n_ranges = len(list(self._c.measure_ranges))
        if len(char_names) != n_ranges:
            raise ValueError(
                f"characteristics ({len(char_names)}) and measure_ranges ({n_ranges}) "
                "must be the same length (one characteristic per measure axis)."
            )
        self._csv_header(
            self._out_dir() / "cma_me_candidates.csv",
            ["round", "index", "candidate", "success", "objective", *char_names],
        )
        self._csv_header(
            self._out_dir() / "cma_me_convergence.csv",
            ["round", "evaluated_total", "num_elites", "obj_max", "obj_mean",
             "coverage", "qd_score", "norm_qd_score"],
        )

    def _csv_header(self, path: Path, header: list[str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(header)

    def _csv_append(self, path: Path, row: list) -> None:
        with open(path, "a", newline="") as f:
            csv.writer(f).writerow(row)

    def _write_outputs(self, archive: archives.GridArchive) -> None:
        """Export the archive, best elite, all elites (as gene YAMLs), and a convergence summary."""
        out = self._out_dir()
        out.mkdir(parents=True, exist_ok=True)

        # 1) full archive (all elites): JSON + CSV
        df = archive.data(return_type="pandas")
        df.to_json(out / "cma_me_archive.json", orient="records", indent=2)
        df.to_csv(out / "cma_me_archive.csv", index=False)

        # 2) best elite: JSON (objective / characteristics / gene) + gene YAML
        best = archive.best_elite
        best_measures = [float(m) for m in best["measures"]]
        best_gene = self._build_gene(np.asarray(best["solution"], dtype=np.float64))
        self._dump_json(out / "cma_me_best.json", {
            "objective": float(best["objective"]),
            "characteristics": dict(zip(self._c.characteristics, best_measures)),
            "gene": best_gene,
        })
        self._write_gene_yaml(out / "cma_me_best.yaml", best_gene)

        # 3) all elites as gene YAMLs
        if self._c.save_elites:
            elites_dir = out / "cma_me_elites"
            for k, sol in enumerate(archive.data()["solution"]):
                self._write_gene_yaml(elites_dir / f"elite_{k}.yaml", self._build_gene(np.asarray(sol, dtype=np.float64)))

        # 4) convergence summary (final stats + per-round log)
        stats = archive.stats
        self._dump_json(out / "cma_me_summary.json", {
            "num_elites": int(stats.num_elites),
            "obj_max": float(stats.obj_max),
            "obj_mean": float(stats.obj_mean),
            "coverage": float(stats.coverage),
            "qd_score": float(stats.qd_score),
            "norm_qd_score": float(stats.norm_qd_score),
            "convergence": self._convergence,
        })
        self.logger.info("Wrote CMA-ME outputs to %s (elites=%d, obj_max=%s)", out, stats.num_elites, stats.obj_max)

        # 5) optional archive heatmap (best-effort; requires matplotlib)
        if self._c.save_archive_plot:
            self._write_plot(archive, out / "cma_me_heatmap.png")

    @staticmethod
    def _dump_json(path: Path, obj) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(obj, f, indent=2)

    def _write_plot(self, archive: archives.GridArchive, path: Path) -> None:
        """Best-effort archive heatmap. Skips (with a warning) if matplotlib is unavailable."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from ribs import visualize
        except ImportError as e:
            self.logger.warning("save_archive_plot is enabled but plotting deps are unavailable (%s); skipping.", e)
            return
        try:
            fig, ax = plt.subplots()
            visualize.grid_archive_heatmap(archive, ax)
            fig.tight_layout()
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path)
            plt.close(fig)
            self.logger.info("Wrote archive heatmap to %s", path)
        except Exception:
            self.logger.exception("Failed to render archive heatmap to %s", path)

    # ------------------------------------------------------------------ #
    # Developer extension points (intentionally left for the developer)
    # ------------------------------------------------------------------ #
    def _fitness(self, candidate_dir: Path) -> float:
        """Return the steering-MSE objective for this candidate.

        The MSE of SimLingo's predicted steering (``simlingo_predictions.csv`` ``steering``
        column) against the recorded steering (``measurements/<frame>.json.gz`` ``steer``),
        averaged over the frames ``>= fitness_frame_index``. Higher = SimLingo is further from
        the recorded expert = a more challenging scenario (the quantity we maximise).
        """
        frame_index = int(self._c.fitness_frame_index)
        frames, predicted, recorded = self._per_frame_steering(candidate_dir)
        mask = frames >= frame_index
        if not mask.any():
            hi = int(frames.max()) if frames.size else "none"
            raise ValueError(
                f"no frames >= {frame_index} to compute steering MSE over (max frame = {hi})."
            )
        err = (predicted[mask] - recorded[mask]) ** 2
        return float(np.mean(err))

    def _per_frame_steering(self, candidate_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(frames, predicted, recorded)`` steering arrays aligned by frame.

        ``predicted`` comes from the ``offline_simlingo`` CSV (``steering``); ``recorded`` from
        the per-frame measurements (``steer``). Frames are the integer indices present in both
        (stems are zero-padded, so both are matched by ``int(stem)``). Exposed so
        :meth:`CmaMeStage._characteristics` can derive ``argmax_t err[t]`` with ``err = predicted - recorded``.
        """
        cand = Path(candidate_dir)
        csv_path = cand / "simlingo_predictions.csv"
        meas_dir = cand / "measurements"
        if not csv_path.is_file():
            raise FileNotFoundError(f"predictions CSV not found: {csv_path}")
        if not meas_dir.is_dir():
            raise FileNotFoundError(f"measurements dir not found: {meas_dir}")

        predicted: dict[int, float] = {}
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    predicted[int(row["frame"])] = float(row["steering"])
                except (KeyError, ValueError):
                    continue

        recorded: dict[int, float] = {}
        for p in sorted(meas_dir.glob("*.json*")):
            try:
                frame = int(p.name.split(".", 1)[0])
            except ValueError:
                continue
            recorded[frame] = float(self._load_measurement(p)["steer"])

        common = sorted(set(predicted) & set(recorded))
        frames = np.asarray(common, dtype=np.int64)
        return (
            frames,
            np.asarray([predicted[f] for f in common], dtype=np.float64),
            np.asarray([recorded[f] for f in common], dtype=np.float64),
        )

    @staticmethod
    def _load_measurement(path: Path) -> dict:
        raw = gzip.open(path, "rb").read() if path.name.endswith(".gz") else path.read_bytes()
        return json.loads(raw)

    def _characteristics(self, candidate_dir: Path, gene: dict) -> np.ndarray:
        """Return the 4-vector of behaviour characteristics for this candidate.

        Order (see config ``characteristics``):
            ``[occlusion_ratio(Seg_16..93), TTC(states), agents_in_corridor(states), argmax_t err[t]]``

        DRAFT: ``occlusion_ratio``, ``TTC`` and ``agents_in_corridor`` are computed from the
        per-frame ego + background-agent states in ``records.json.gz`` using simple geometric rules
        with the configurable ``char_*`` thresholds; ``argmax_t err[t]`` uses
        :meth:`CmaMeStage._per_frame_steering`. Replace the individual sub-methods with your
        exact definitions as needed.
        """
        states = self._frames_states(candidate_dir)
        if not states:
            self.logger.warning(
                "no per-frame states in records.json(.gz) at %s; state-based characteristics default to 0. "
                "Ensure the scenario records states (or implement _characteristics directly).",
                candidate_dir,
            )
        lo, hi = int(self._c.char_occl_frame_range[0]), int(self._c.char_occl_frame_range[1])
        occl = self._occlusion_ratio(
            states, lo, hi, float(self._c.char_occl_max_forward_m), float(self._c.char_occl_max_lateral_m)
        )
        ttc = self._ttc(states, float(self._c.char_ttc_horizon_s))
        corr = self._agents_in_corridor(
            states, float(self._c.char_corridor_forward_m), float(self._c.char_corridor_lateral_m)
        )
        argmax = self._argmax_err_t(candidate_dir, int(self._c.fitness_frame_index))
        return np.array([occl, ttc, corr, argmax], dtype=np.float64)

    # --- characteristic sub-methods (draft) ------------------------------------------- #
    def _frames_states(self, candidate_dir: Path) -> list:
        """Load the per-frame states from the candidate's records file.

        The scenario records to ``records.json.gz`` (``{meta_data, states, lights,
        route, ego_actions, adv_actions}``); a plain ``records.json`` is accepted as a
        fallback. Returns ``data["states"]`` -- one entry per frame, each a dict of
        per-agent fields (``pos``, ``vel``, ``yaw``, ``id``, ``type``, ``extent``,
        ...), first entry per frame = ego, count varies as actors spawn/despawn.
        """
        cand = Path(candidate_dir)
        for name in ("records.json.gz", "records.json"):
            rec = cand / name
            if not rec.is_file():
                continue
            try:
                raw = gzip.open(rec, "rb").read() if name.endswith(".gz") else rec.read_bytes()
                data = json.loads(raw)
            except (OSError, EOFError, ValueError, zlib.error):
                self.logger.warning("could not parse %s", rec)
                continue
            if not isinstance(data, dict):
                self.logger.warning("unexpected records structure in %s", rec)
                continue
            states = data.get("states") or []
            if states:
                return list(states)
        return []

    @staticmethod
    def _frame_agents(state: dict) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray] | None:
        """Parse one per-frame state into ``(ego_pos, ego_vel, ego_yaw, agents_pos, agents_vel)``.

        The recorder stores each frame as ``{pos, vel, yaw, id, type, extent, color, height,
        pitch, roll}`` with every field shaped ``[1, N, ...]`` (``pos``/``vel`` ``[1, N, 2]``,
        ``yaw`` ``[1, N, 1]``). ``N`` varies per frame (actors spawn/despawn) and the ego is
        the first entry. Returns ``None`` if the frame has no agents or an unexpected shape.
        """
        try:
            pos = np.asarray(state["pos"], dtype=np.float64)
            vel = np.asarray(state["vel"], dtype=np.float64)
            yaw = np.asarray(state["yaw"], dtype=np.float64)
        except (KeyError, ValueError, TypeError):
            return None
        if pos.ndim != 3 or vel.ndim != 3 or yaw.ndim != 3 or pos.shape[1] < 1:
            return None
        n = pos.shape[1]
        ego_pos = pos[0][0]
        ego_vel = vel[0][0]
        ego_yaw = float(yaw[0][0][0])
        agents_pos = pos[0][1:]
        agents_vel = vel[0][1:] if vel.shape[1] >= n else np.zeros((0, 2))
        return ego_pos, ego_vel, ego_yaw, agents_pos, agents_vel

    @staticmethod
    def _ego_frame(ego_yaw: float, agents_pos: np.ndarray, ego_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(forward, lateral)`` offsets of each agent in the ego frame (x forward, y left)."""
        rel = agents_pos - ego_pos
        c, s = np.cos(ego_yaw), np.sin(ego_yaw)
        fwd = c * rel[:, 0] + s * rel[:, 1]
        lat = -s * rel[:, 0] + c * rel[:, 1]
        return fwd, lat

    def _occlusion_ratio(self, states, lo: int, hi: int, max_fwd: float, max_lat: float) -> float:
        """Fraction of frames in ``[lo, hi]`` where an agent is directly ahead of the ego."""
        occ = total = 0
        for idx, state in enumerate(states):
            if not (lo <= idx <= hi):
                continue
            total += 1
            parsed = self._frame_agents(state)
            if parsed is None or parsed[3].size == 0:
                continue
            ego_pos, _, ego_yaw, agents_pos, _ = parsed
            fwd, lat = self._ego_frame(ego_yaw, agents_pos, ego_pos)
            if np.any((fwd > 0) & (fwd <= max_fwd) & (np.abs(lat) < max_lat)):
                occ += 1
        return occ / total if total else 0.0

    def _ttc(self, states, horizon: float) -> float:
        """Minimum time-to-collision with any approaching agent over the episode (else ``horizon``)."""
        best = horizon
        for state in states:
            parsed = self._frame_agents(state)
            if parsed is None or parsed[3].size == 0:
                continue
            ego_pos, ego_vel, _, agents_pos, agents_vel = parsed
            for i in range(agents_pos.shape[0]):
                rel = agents_pos[i] - ego_pos
                dist = float(np.linalg.norm(rel))
                approach = -float(np.dot(rel, agents_vel[i] - ego_vel))
                if approach > 1e-6:  # range decreasing -> the agent is closing in
                    best = min(best, dist / approach)
        return float(np.clip(best, 0.0, horizon))

    def _agents_in_corridor(self, states, fwd_len: float, lat_half: float) -> float:
        """Max number of agents inside the ego's corridor (ahead + within ``lat_half``) over the episode."""
        best = 0
        for state in states:
            parsed = self._frame_agents(state)
            if parsed is None or parsed[3].size == 0:
                continue
            ego_pos, _, ego_yaw, agents_pos, _ = parsed
            fwd, lat = self._ego_frame(ego_yaw, agents_pos, ego_pos)
            best = max(best, int(np.sum((fwd > 0) & (fwd <= fwd_len) & (np.abs(lat) < lat_half))))
        return float(best)

    def _argmax_err_t(self, candidate_dir: Path, frame_index: int) -> float:
        """Frame index of the largest ``|steering|`` error at/after ``frame_index`` (0 if none)."""
        frames, predicted, recorded = self._per_frame_steering(candidate_dir)
        mask = frames >= frame_index
        if not mask.any():
            return 0.0
        err = np.abs(predicted[mask] - recorded[mask])
        return float(frames[mask][int(np.argmax(err))])

    def verify_completion(self) -> bool:
        return self._archive is not None and self._archive.stats.num_elites >= 1
