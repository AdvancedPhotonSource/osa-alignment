#!/usr/bin/env python3
"""Align the OSA by intersecting directed brightness-symmetry lines."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path

import numpy as np

from alignment_io import load_manifest, measure, write_json
from evaluate_alignment import Config, json_safe


@dataclass
class Settings:
    units: str
    bounds: list[list[float]]
    evaluator: dict = field(default_factory=dict)
    valid_axes_required: int = 6
    reverse_step_um: float = 5.
    lateral_step_um: float = 5.
    max_acquisitions: int = 20
    readback_tolerance: float = 0.001
    beta: float = 0.5

    def validate(self):
        bounds = np.asarray(self.bounds, float)
        if self.units != "um":
            raise ValueError("Symmetry alignment uses motor coordinates in um")
        if (bounds.shape != (2, 2) or not np.isfinite(bounds).all()
            or np.any(bounds[:, 0] >= bounds[:, 1])):
            raise ValueError("Require finite increasing x/y motor bounds")
        values = [self.reverse_step_um, self.lateral_step_um, self.readback_tolerance]
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError("Step distances and readback_tolerance must be finite and positive")
        if not 0 < self.beta <= 1:
            raise ValueError("Momentum weight beta must be in (0, 1]")
        if (type(self.valid_axes_required) is not int or type(self.max_acquisitions) is not int
            or not 2 <= self.valid_axes_required <= self.max_acquisitions):
            raise ValueError("Require 2 <= valid_axes_required <= max_acquisitions")
        Config(**self.evaluator).validate()


class Replay:
    """Exact recorded-image replay; continuous targets are never snapped or interpolated."""
    def __init__(self, manifest, start):
        self.units = manifest["units"]
        self.images = {}
        for row in manifest["samples"]:
            self.images.setdefault(tuple(row["position"]), []).append(row["image"])
        self.counts = dict.fromkeys(self.images, 0)
        self.move_absolute(start)

    def read_position(self):
        return self.position.copy()

    def move_absolute(self, position):
        self.position = np.asarray(position, float)

    def acquire_image(self):
        key = tuple(self.position)
        if key not in self.images:
            raise ValueError("Replay requires an image at the exact requested position")
        paths = self.images[key]
        count = self.counts[key]
        self.counts[key] += 1
        return paths[count % len(paths)]


def motor_direction(image_direction):
    """Map image right/down to motor +x/-y and return a unit motor-space vector."""
    direction = np.asarray(image_direction, float) * [1., -1.]
    if direction.shape != (2,) or not np.isfinite(direction).all() or np.linalg.norm(direction) == 0:
        raise ValueError("Expected a finite nonzero image-axis direction")
    return direction / np.linalg.norm(direction)


def symmetry_line(position, direction):
    """Line n dot p = b in motor µm, anchored at the acquisition readback.

    The image-axis intercept is detector geometry, not a motor displacement.
    Only the mapped axis orientation is carried into motor space.
    """
    position, direction = np.asarray(position), np.asarray(direction)
    normal = np.array([-direction[1], direction[0]])
    return {"point": position.tolist(), "direction": direction.tolist(),
            "normal": normal.tolist(), "constant": float(normal @ position)}


def intersect_lines(lines):
    """Unweighted least-squares intersection minimizing squared perpendicular µm distances."""
    normals = np.array([line["normal"] for line in lines])
    positions = np.array([line["point"] for line in lines])
    origin = positions.mean(axis=0)
    offsets = np.einsum("ij,ij->i", normals, positions - origin)
    delta, _, rank, singular = np.linalg.lstsq(normals, offsets, rcond=None)
    if rank < 2:
        raise RuntimeError("Symmetry lines are rank-deficient; no unique intersection")
    point = origin + delta
    residuals = normals @ delta - offsets
    return {"position": point.tolist(), "residuals_um": residuals.tolist(),
            "rms_distance_um": float(np.sqrt(np.mean(residuals**2))),
            "rank": int(rank), "condition_number": float(singular[0] / singular[-1])}


class Controller:
    def __init__(self, settings, backend, log_path, measurement=None):
        settings.validate()
        if backend.units != settings.units:
            raise ValueError("Adapter units must match settings")
        self.s, self.backend = settings, backend
        self.measurement = measurement
        self.evaluator = Config(**settings.evaluator)
        self.first_image_crop = None
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("x"):
            pass
        self.history, self.lines = [], []
        self.momentum = np.zeros(2)
        self.current, self.intersection = None, None
        self.acquisitions = 0
        self.replay = isinstance(backend, Replay)
        self.record("start", settings=asdict(settings), replay=self.replay)

    def record(self, event, **data):
        record = {"time": datetime.now(timezone.utc).isoformat(), "event": event, **data}
        with self.log_path.open("a") as stream:
            stream.write(json.dumps(json_safe(record), allow_nan=False) + "\n")
        if "text" in data:
            print(data["text"], flush=True)

    def position(self):
        p = np.asarray(self.backend.read_position(), float)
        bounds = np.asarray(self.s.bounds)
        if (p.shape != (2,) or not np.isfinite(p).all()
            or np.any(p < bounds[:, 0] - self.s.readback_tolerance)
            or np.any(p > bounds[:, 1] + self.s.readback_tolerance)):
            raise RuntimeError("Readback is nonfinite or outside allowed bounds")
        return p

    def move(self, target, purpose):
        before = self.position()
        proposed = np.asarray(target, float)
        bounds = np.asarray(self.s.bounds)
        if proposed.shape != (2,) or not np.isfinite(proposed).all():
            raise RuntimeError("Proposed move requires two finite motor coordinates")
        target = np.clip(proposed, bounds[:, 0], bounds[:, 1])
        self.record("plan", text=f"{purpose}: move to {target.tolist()} um.",
                    proposed_target=proposed.tolist(),
                    target=target.tolist(), previous_readback=before.tolist(), purpose=purpose)
        if np.linalg.norm(target - before) > 1e-12:
            self.backend.move_absolute(target.tolist())
        actual = self.position()
        if np.linalg.norm(actual - target) > self.s.readback_tolerance:
            raise RuntimeError("Motor readback does not match target")
        self.record("move", previous_readback=before.tolist(), target=target.tolist(),
                    position=actual.tolist(), purpose=purpose)
        return actual

    def accept_axis(self, observation, position, direction):
        """Orient the motor axis so its clockwise normal follows accepted-readback momentum (µm)."""
        lateral = np.array([direction[1], -direction[0]])
        if self.lines:
            delta = position - self.lines[-1]["point"]
            beta = self.s.beta
            self.momentum = (1 - beta) * self.momentum + beta * delta
            projection = lateral @ self.momentum
            # Break an exact tie consistently for either representation of the same line.
            if projection < 0 or (projection == 0 and tuple(lateral) < (0., 0.)):
                direction, lateral = -direction, -lateral
        line = symmetry_line(position, direction)
        line["acquisition"] = self.acquisitions
        self.lines.append(line)
        observation.update(line=line, motor_direction=direction.tolist(),
                           lateral_direction=lateral.tolist(), momentum_um=self.momentum.tolist())

    def reject_axis(self, observation, image_axis):
        """Carry the first rejection's motor-space reverse vector through consecutive rejections."""
        direction = motor_direction(image_axis) if image_axis is not None else None
        if direction is not None:
            observation["motor_direction"] = direction.tolist()
        previous = self.current.get("reverse_direction") if self.current else None
        if previous is not None:
            observation["reverse_direction"] = previous
        elif direction is not None:
            observation["reverse_direction"] = (-direction).tolist()
        else:
            error = "First rejected image has no brightness direction for the reverse step"
            observation["metrics"]["error"] = error
            observation["metrics"]["reasons"].append(error)

    def acquire(self):
        if self.acquisitions >= self.s.max_acquisitions:
            raise RuntimeError("Acquisition budget exhausted")
        before = self.position()
        self.acquisitions += 1
        self.record("acquiring", acquisition=self.acquisitions, position=before.tolist(),
                    text=f"Acquire image {self.acquisitions} at {before.tolist()} um.")
        image = self.backend.acquire_image()
        after = self.position()
        if np.linalg.norm(after - before) > self.s.readback_tolerance:
            raise RuntimeError("Motor position changed during acquisition")
        try:
            if self.measurement is None:
                observation = measure(image, config=self.evaluator, first_image_crop=self.first_image_crop)
                if not self.evaluator.crop_xyxy and self.first_image_crop is None:
                    self.first_image_crop = observation["metrics"]["crop_xyxy"]
            else:
                observation = self.measurement(image)
        except Exception as exc:
            observation = {"image": str(image), "metrics": {
                "valid": False, "fit": None, "error": str(exc), "reasons": [str(exc)],
                "config": asdict(self.evaluator),
                "crop_xyxy": list(self.evaluator.crop_xyxy or self.first_image_crop or ())}}
        observation.update(position=after.tolist(), acquisition=self.acquisitions,
                           stage="search", line=None)
        metrics = observation["metrics"]
        if not metrics.get("error"):
            image_axis = metrics.get("direction_xy")
            if metrics["valid"]:
                if image_axis is None:
                    image_axis = metrics["fit"]["axis_direction_xy"]
                self.accept_axis(observation, after, motor_direction(image_axis))
            else:
                self.reject_axis(observation, image_axis)
        observation["valid_axes_count"] = len(self.lines)
        self.history.append(observation)
        self.current = observation
        state = "failed" if metrics.get("error") else "accepted" if metrics["valid"] else "rejected"
        self.record("observation", observation=observation,
                    text=f"Image {self.acquisitions}: axis {state}; {len(self.lines)}/{self.s.valid_axes_required} valid axes. "
                         + "; ".join(metrics.get("reasons", [])))
        return observation

    def run(self):
        status, reason = "blocked", ""
        try:
            while self.acquisitions < self.s.max_acquisitions:
                observation = self.acquire()
                metrics = observation["metrics"]
                if metrics.get("error"):
                    raise RuntimeError(metrics["error"])
                if len(self.lines) == self.s.valid_axes_required:
                    self.intersection = intersect_lines(self.lines)
                    self.record("intersection", intersection=self.intersection,
                                text=f"Least-squares intersection: {self.intersection['position']} um.")
                    self.move(self.intersection["position"], "Move to symmetry intersection")
                    status, reason = "aligned", "Completed move to the bounded symmetry-intersection target"
                    break
                if self.acquisitions == self.s.max_acquisitions:
                    status, reason = "acquisition_limit", "Acquisition limit reached before collecting enough valid axes"
                    break
                if metrics["valid"]:
                    step = np.array(observation["lateral_direction"]) * self.s.lateral_step_um
                    purpose = "Accepted axis: move along selected perpendicular"
                else:
                    step = np.array(observation["reverse_direction"]) * self.s.reverse_step_um
                    purpose = "Rejected axis: move along the current reverse direction"
                self.move(np.array(observation["position"]) + step, purpose)
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            self.record("fault", text=reason)
        try:
            final_position = self.position().tolist()
        except Exception:
            final_position = None
        result = {"status": status, "reason": reason, "replay": self.replay,
                  "acquisitions": self.acquisitions, "valid_axes_count": len(self.lines),
                  "lines": self.lines, "intersection": self.intersection,
                  "final_observation": self.current, "final_readback": final_position,
                  "log": str(self.log_path)}
        self.record("complete", text=f"Stopped: {status}; {reason}.", result=result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="JSON Settings object")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path, help="Replay recorded positions without interpolation")
    source.add_argument("--adapter", help="Beamline adapter MODULE:FACTORY, e.g. aps2idd_adapter:create_adapter")
    parser.add_argument("--adapter-config", type=Path)
    parser.add_argument("--start", type=float, nargs=2, help="Replay starting position")
    parser.add_argument("--log", type=Path, required=True, help="New JSONL trajectory file")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--webui", action="store_true", help="Serve a read-only live dashboard")
    parser.add_argument("--webui-port", type=int, default=8765, help="Local dashboard port (0 selects a free port)")
    args = parser.parse_args()
    if not 0 <= args.webui_port <= 65535:
        parser.error("--webui-port must be between 0 and 65535")
    settings = Settings(**json.loads(args.config.read_text()))
    settings.validate()
    if args.manifest:
        if args.start is None:
            parser.error("Replay requires --start X Y")
        backend = Replay(load_manifest(args.manifest), args.start)
    else:
        if args.start is not None:
            parser.error("Live mode reads its starting position from the adapter")
        module, factory = args.adapter.split(":", 1)
        options = json.loads(args.adapter_config.read_text()) if args.adapter_config else {}
        backend = getattr(importlib.import_module(module), factory)(options)
    controller = Controller(settings, backend, args.log)
    dashboard = None
    if args.webui:
        from alignment_webui import Dashboard
        dashboard = Dashboard(args.log, port=args.webui_port)
        dashboard.start()
        print(f"OSA web UI: {dashboard.url} (read-only)", flush=True)
    try:
        result = controller.run()
        write_json(args.output, result)
    finally:
        if dashboard is not None:
            dashboard.close()
    if result["status"] != "aligned":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
