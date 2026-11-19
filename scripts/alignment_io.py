"""Image measurements, replay manifests, and JSON output for OSA alignment."""
import json
from pathlib import Path

import numpy as np

from evaluate_alignment import Config, evaluate, json_safe, read_image


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n")


def load_manifest(path):
    """Load positions and image paths; positions use the manifest's explicit units."""
    path = Path(path).resolve()
    data = json.loads(path.read_text())
    if not isinstance(data.get("units"), str) or not data["units"].strip():
        raise ValueError("Manifest must specify motor units")
    if not data.get("samples"):
        raise ValueError("Manifest samples must not be empty")
    for row in data["samples"]:
        p = np.asarray(row["position"], dtype=float)
        if p.shape != (2,) or not np.isfinite(p).all():
            raise ValueError("Each position must contain two finite numbers")
        row["position"] = p.tolist()
        row["image"] = str((path.parent / row["image"]).resolve())
    return data


def measure(path, *, config=None, first_image_crop=None):
    image = read_image(path)
    result = evaluate(image, Config() if config is None else config, first_image_crop=first_image_crop)
    return {"image": str(Path(path).resolve()), "metrics": result}
