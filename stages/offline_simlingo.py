from __future__ import annotations
import csv
import gzip
import importlib
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Union, Callable

import cv2
import numpy as np
from omegaconf import DictConfig

from stage import Stage


class OfflineSimlingoEval(Stage):
    """Replay a recorded scenario through the offline SimLingo agent.

    Buffers the recorded frames (RGB image + speed/position/heading/route from
    the per-frame measurements) and runs one ``agent.predict`` per frame, saving
    the controls, predicted trajectories and language output to a CSV.
    """

    def __init__(self, config: DictConfig, stage: str = "offline_simlingo") -> None:
        super().__init__(config, stage)

    def _run(self) -> None:
        self._setup_agent()

        self.frame_start_index = self.config.stage.start_index

        frames = self._load_buffer()
        self.logger.info("Buffered %d frame(s) from %s. Starting index: %d", len(frames), self.config.stage.input_dir, self.frame_start_index)

        out_path = self._output_csv_path()
        rows = []
        for i, frame in enumerate(frames):
            if i < self.frame_start_index:
                continue

            out = self._predict(
                frame["image"],
                speed=frame["speed"],
                position=frame["position"],
                heading=frame["heading"],
                route=frame["route"],
            )
            rows.append(self._row(frame["frame"], out))
            if (i + 1) % 10 == 0 or (i + 1) == len(frames):
                self.logger.info("Predicted %d/%d frame(s)", i + 1, len(frames))

        self._write_csv(rows, out_path)
        self.logger.info("Wrote %d prediction row(s) to %s", len(rows), out_path)

    def _predict(self, image: np.ndarray, speed: float, position: np.ndarray, heading: float, route: np.ndarray) -> dict:
        return self.agent.predict(
            image, speed=speed, position=position, heading=heading, route=route
        )

    def verify_completion(self) -> bool:
        out_path = self._output_csv_path()
        ok = out_path.is_file() and out_path.stat().st_size > 0
        if not ok:
            self.logger.error("Expected output CSV not found or empty: %s", out_path)
        return ok

    def _setup_agent(self) -> None:
        sys.path.append(self.config.stage.pythonpath)
        importlib.invalidate_caches()

        OfflineLingoAgent = self._inject_dependency("team_code.offline_agent", "OfflineLingoAgent")

        self.agent = OfflineLingoAgent(                 # type: ignore
            checkpoint_path=self.config.stage.weights,
            eval_route_as="target_point",
            pretrained_cache_dir="pretrained",
            use_cot=None, # use COT, set False to disable,
            device="cuda"
        )

    def _inject_dependency(self, module_name, class_name: str = "") -> Union[Callable, ModuleType]:
        """Load a module (appended to ``sys.path`` earlier) and return an attribute of it."""
        module = importlib.import_module(module_name)
        return getattr(module, class_name) if class_name else module

    def _input_dir(self) -> Path:
        return Path(self.config.stage.input_dir)

    def _output_csv_path(self) -> Path:
        csv_cfg = str(self.config.stage.output_csv)
        p = Path(csv_cfg)
        return p if p.is_absolute() else Path(self.config.run.output_dir) / p

    def _load_buffer(self) -> list[dict]:
        """Load all frames into memory, paired as image + measurements.

        Frames are identified by the rgb file stems (e.g. ``0001``); the
        matching ``<stem>.json.gz`` measurement must exist. The recorded
        ``route`` is ego-frame (x forward, y left); it is converted to world
        coordinates (with command 4 = keep straight) because ``agent.predict``
        expects world waypoints and re-derives the ego-frame target point.
        """
        rgb_dir = self._input_dir() / self.config.stage.rgb_dir
        meas_dir = self._input_dir() / self.config.stage.measurements_dir
        if not rgb_dir.is_dir():
            raise FileNotFoundError(f"rgb dir not found: {rgb_dir}")
        if not meas_dir.is_dir():
            raise FileNotFoundError(f"measurements dir not found: {meas_dir}")

        images = sorted(rgb_dir.glob("*"), key=lambda p: (p.suffix, p.name))
        images = [p for p in images if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
        if not images:
            raise FileNotFoundError(f"no images found in {rgb_dir}")

        frames = []
        for img_path in images:
            try:
                frame_id = int(img_path.stem)
            except ValueError:
                frame_id = img_path.stem  # non-numeric stem: keep as-is
            meas_path = meas_dir / f"{img_path.stem}.json.gz"
            if not meas_path.is_file():
                meas_path = meas_dir / f"{img_path.stem}.json"
                if not meas_path.is_file():
                    raise FileNotFoundError(f"no measurement for frame {img_path.name}: expected {meas_dir / (img_path.stem + '.json.gz')}")
            meas = self._load_measurement(meas_path)
            image_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise ValueError(f"could not decode image: {img_path}")
            frames.append({
                "frame": frame_id,
                "image": cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
                "speed": float(meas["speed"]),
                "position": np.asarray(meas["pos_global"], dtype=np.float32),
                "heading": float(meas["theta"]),
                "route": self._route_ego_to_world(meas["route"], meas["pos_global"], meas["theta"]),
            })
        return frames

    @staticmethod
    def _load_measurement(path: Path) -> dict:
        if path.name.endswith(".gz"):
            raw = gzip.open(path, "rb").read()
        else:
            raw = path.read_bytes()
        return json.loads(raw)

    @staticmethod
    def _route_ego_to_world(route_ego, position, heading) -> list:
        """Convert the recorded ego-frame route (x forward, y left) to world coords.

        ``agent.predict`` expects world waypoints of the form ``[x, y, command]``
        and derives the ego-frame target point itself (it uses waypoints 1 and 2).
        We mirror the recorded convention (``command`` 4 = keep straight) so the
        agent's inverse conversion recovers exactly the recorded route geometry.
        """
        pos = np.asarray(position, dtype=np.float64)[:2]
        yaw = float(heading)
        rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        pts = np.asarray(route_ego, dtype=np.float64)[:, :2]
        world = (rot @ pts.T).T + pos
        return [[float(x), float(y), 4] for x, y in world]

    @staticmethod
    def _fmt_route(route) -> str:
        """Serialize an (F, 2) waypoint array as ``x y; x y; ...``."""
        arr = np.asarray(route, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] < 2:
            return str(route)
        return "; ".join(f"{x:.4f} {y:.4f}" for x, y in arr[:, :2])

    @classmethod
    def _row(cls, frame, out: dict) -> dict:
        return {
            "frame": frame,
            "throttle": out["throttle"],
            "brake": out["brake"],
            "steering": out["steer"],
            "pred_route": cls._fmt_route(out["pred_route"]),
            "pred_route_world": cls._fmt_route(out["pred_route_world"]),
            "language": out["language"],
        }

    @staticmethod
    def _write_csv(rows: list[dict], out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["frame", "throttle", "brake", "steering", "pred_route", "pred_route_world", "language"]
        with open(out_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
