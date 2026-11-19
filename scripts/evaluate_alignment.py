#!/usr/bin/env python3
"""Crop an OSA image and measure its directed brightness-reflection axis."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates

from symmetry import fit_symmetry


@dataclass(frozen=True)
class Config:
    crop_xyxy: tuple[int, ...] | None = (276, 73, 616, 413)
    sigma: float = 1.5
    angle_samples: int = 720
    angle_bound_deg: float = 10.
    offset_bound_px: float = 10.
    max_candidates: int = 3
    evaluation_step: int = 2
    mismatch_relative_tolerance: float = 0.10

    def validate(self):
        crop = self.crop_xyxy
        if crop is not None and not isinstance(crop, (list, tuple)):
            raise ValueError("crop_xyxy must be an array of bounds or null")
        if crop and (len(crop) != 4 or any(type(v) is not int for v in crop)
                     or min(crop) < 0 or crop[2] - crop[0] < 16 or crop[3] - crop[1] < 16):
            raise ValueError("crop_xyxy must be nonnegative integer exclusive bounds, at least 16 x 16")
        numbers = [self.sigma, self.angle_bound_deg, self.offset_bound_px,
                   self.mismatch_relative_tolerance]
        if (not np.isfinite(numbers).all() or self.sigma < 0
            or not 0 < self.angle_bound_deg < 90 or self.offset_bound_px <= 0
            or not 0 <= self.mismatch_relative_tolerance < 1):
            raise ValueError("Invalid smoothing, refinement bounds, or mismatch_relative_tolerance")
        if (any(type(v) is not int for v in [self.angle_samples, self.max_candidates, self.evaluation_step])
            or self.angle_samples < 16 or self.max_candidates < 1 or self.evaluation_step < 1):
            raise ValueError("Require >=16 angular samples, >=1 candidate and positive integer evaluation_step")


def grayscale(array: np.ndarray) -> np.ndarray:
    """Preserve intensity units; RGB uses luminance weights without gamma decoding."""
    a = np.asarray(array, dtype=float)
    if a.ndim == 3 and a.shape[-1] == 3:
        a = a @ np.array([0.2126, 0.7152, 0.0722])
    if a.ndim != 2 or min(a.shape) < 16 or not np.isfinite(a).all():
        raise ValueError("Expected a finite 2-D grayscale or RGB image, at least 16 x 16")
    return a


def read_image(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        if getattr(image, "n_frames", 1) != 1:
            raise ValueError("Select a single image from a multi-frame file first")
        return grayscale(np.asarray(image))


def json_safe(value):
    """Use standards-compliant JSON while retaining explicit infinite scores."""
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return "inf" if value > 0 else "-inf" if value < 0 else None
    return value


def auto_crop_bounds(image):
    """Smallest square with side a positive multiple of 128 enclosing >=95% of raw grayscale intensity.

    Pixel centers have integer coordinates. Round each lower bound to the
    nearest integer so the square's center is within half a pixel of the
    full-frame intensity centroid. Outside-detector intensity is zero.
    """
    height, width = image.shape
    total = image.sum(dtype=float)
    if total <= 0 or not np.isfinite(total) or (image < 0).any():
        raise ValueError("Automatic crop requires nonnegative intensity with a finite positive sum")
    cx = image.sum(axis=0, dtype=float) @ np.arange(width) / total
    cy = image.sum(axis=1, dtype=float) @ np.arange(height) / total
    side = 128
    while True:
        x0 = round(float(cx) - (side - 1) / 2)
        y0 = round(float(cy) - (side - 1) / 2)
        x1, y1 = x0 + side, y0 + side
        inside = image[max(y0, 0):min(y1, height), max(x0, 0):min(x1, width)]
        if inside.sum(dtype=float) >= .95 * total:
            return [x0, y0, x1, y1]
        side += 128


def crop_image(image, crop_xyxy, *, pad=False):
    """Extract exclusive bounds, zero-padding out-of-frame automatic windows."""
    x0, y0, x1, y1 = crop_xyxy
    height, width = image.shape
    if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
        if not pad:
            raise ValueError("Configured crop exceeds image dimensions")
        patch = np.zeros((y1 - y0, x1 - x0), dtype=image.dtype)
        left, top, right, bottom = max(0, x0), max(0, y0), min(width, x1), min(height, y1)
        if left < right and top < bottom:
            patch[top-y0:bottom-y0, left-x0:right-x0] = image[top:bottom, left:right]
        return patch
    return image[y0:y1, x0:x1]


def axis_segment(point, direction, shape):
    """Signed distances along a unit axis where it intersects the pixel-center rectangle."""
    low, high = -np.inf, np.inf
    for p, v, extent in zip(point, direction, reversed(shape)):
        if abs(v) < 1e-12:
            if not 0 <= p <= extent - 1:
                raise ValueError("Axis does not intersect the image")
        else:
            ends = sorted((-p / v, (extent - 1 - p) / v))
            low, high = max(low, ends[0]), min(high, ends[1])
    if not np.isfinite([low, high]).all() or high - low < 1:
        raise ValueError("Axis has insufficient image support")
    return low, high


def directed_axis(image, point, direction, *, require_direction=True):
    """Orient a fitted axis by the slope of raw intensity versus signed distance.

    Bilinear samples span its full in-crop segment at <=1-pixel spacing. Ordinary
    least squares fits I(t)=slope*t+intercept; positive slope keeps the supplied
    direction and negative slope reverses it. A numerically flat profile has no
    defined brightness direction. An accepted line can still guide a perpendicular step.
    """
    point, direction = np.asarray(point), np.asarray(direction)
    low, high = axis_segment(point, direction, image.shape)
    t = np.linspace(low, high, int(np.ceil(high - low)) + 1)
    xy = point[:, None] + direction[:, None] * t
    values = map_coordinates(image, [xy[1], xy[0]], order=1, mode="nearest", prefilter=False)
    centered = t - t.mean()
    slope = float(centered @ (values - values.mean()) / (centered @ centered))
    numerical_floor = 64 * np.finfo(float).eps * max(1., float(np.max(np.abs(values))))
    brightness_direction = (direction if slope > 0 else -direction).tolist()
    if abs(slope) * (high - low) <= numerical_floor:
        if require_direction:
            raise ValueError("Along-axis intensity slope is zero; direction is undefined")
        brightness_direction = None
    return {"direction_xy": brightness_direction,
            "profile_slope_intensity_per_px": slope,
            "profile_intercept": float(values.mean() - slope * t.mean()),
            "profile_distance_px": t.tolist(), "profile_intensity": values.tolist()}


def evaluate(image, config=Config(), *, first_image_crop=None):
    """Return symmetry geometry, brightness direction, and mono-axial acceptance.

    Perpendicular mismatch is evaluated through the fitted axis point closest
    to the intensity centroid. Accept when D_perp - D_axis exceeds the threshold,
    where D is the absolute reflection residual divided by twice the smoothed crop intensity sum.
    Configuration errors raise; measurement failures remain in acquisition history.
    """
    config.validate()
    image = grayscale(image)
    result = {"config": asdict(config), "image_shape": list(image.shape),
              "crop_xyxy": list(config.crop_xyxy or ()), "valid": False,
              "error": None, "reasons": [], "fit": None}
    try:
        if not config.crop_xyxy:
            result["crop_xyxy"] = (list(first_image_crop) if first_image_crop is not None
                                   else auto_crop_bounds(image))
        patch = crop_image(image, result["crop_xyxy"], pad=not config.crop_xyxy)
        result["crop_shape"] = list(patch.shape)
        kwargs = {name: getattr(config, name) for name in
                  ("sigma", "angle_samples", "angle_bound_deg", "offset_bound_px", "max_candidates", "evaluation_step")}
        analysis = fit_symmetry(patch, **kwargs)
        fit = analysis["best_axis"]
        result.update(fit=fit, centroid_xy_px=analysis["centroid_xy_px"],
                      fft_score_range=analysis["fft_score_range"], candidates=analysis["candidates"])
        a, b = fit["mismatch"], fit["perpendicular_mismatch"]
        relative = abs(b - a) / max(a, b) if max(a, b) > 0 else 0.
        result.update(reflection_mismatch=a, perpendicular_mismatch=b, relative_mismatch_difference=relative)
        gap = fit["d_perp"] - fit["d_axis"]
        result.update(d_axis=fit["d_axis"], d_perp=fit["d_perp"], intensity_gap=gap)
        valid = gap > config.mismatch_relative_tolerance
        result.update(directed_axis(patch, fit["axis_point_xy_px"], fit["axis_direction_xy"],
                                    require_direction=not valid))
        result["valid"] = valid
        if not result["valid"]:
            result["reasons"].append("Signed intensity gap is at or below the acceptance threshold")
    except (ValueError, RuntimeError) as exc:
        result["error"] = str(exc)
        result["reasons"].append(str(exc))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--config", type=Path, help="Evaluator Config JSON")
    parser.add_argument("--crop", type=int, nargs=4, metavar=("X0", "Y0", "X1", "Y1"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    options = json.loads(args.config.read_text()) if args.config else {}
    if args.crop is not None:
        options["crop_xyxy"] = args.crop
    result = evaluate(read_image(args.image), Config(**options))
    payload = json.dumps(json_safe(result), indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(payload)
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
