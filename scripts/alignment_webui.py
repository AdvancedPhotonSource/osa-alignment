"""Read-only local dashboard backed by the controller's JSONL trajectory."""
from __future__ import annotations

import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
from pathlib import Path
import re
from threading import Thread

import numpy as np
from PIL import Image, ImageDraw

from evaluate_alignment import axis_segment, crop_image, read_image


class Dashboard:
    def __init__(self, log_path, port=8765):
        self.log_path = Path(log_path)
        self.snapshot_path = self.log_path.with_suffix(".html")
        self.template = Path(__file__).with_name("alignment_webui.html").read_text()
        dashboard = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/":
                    body, content_type = dashboard.html().encode(), "text/html; charset=utf-8"
                elif self.path == "/api/status":
                    body = json.dumps(dashboard.state(), allow_nan=False).encode()
                    content_type = "application/json"
                elif match := re.fullmatch(r"/api/images/([1-9][0-9]*)", self.path):
                    acquisition = int(match[1])
                    observation = next((o for o in dashboard.state(include_image=False)["observations"]
                                        if o.get("acquisition") == acquisition), None)
                    if observation is None:
                        self.send_error(404)
                        return
                    body = json.dumps(dashboard.preview(observation), allow_nan=False).encode()
                    content_type = "application/json"
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.url = f"http://127.0.0.1:{self.server.server_port}/"

    def state(self, include_image=True):
        state = {"settings": {}, "observations": [], "moves": [], "intersection": None, "result": None,
                 "message": "Waiting for the first acquisition", "updated": None,
                 "image": None, "image_note": "Waiting for an image"}
        with self.log_path.open() as stream:
            for line in stream:
                # A concurrent append may leave the last record incomplete.
                if not line.endswith("\n"):
                    break
                event = json.loads(line)
                state["updated"] = event["time"]
                if event["event"] == "start":
                    state.update(settings=event["settings"],
                                 replay=event["replay"])
                elif event["event"] == "observation":
                    state["observations"].append(event["observation"])
                elif event["event"] == "move":
                    state["moves"].append(event)
                elif event["event"] == "intersection":
                    state["intersection"] = event["intersection"]
                elif event["event"] == "complete":
                    state["result"] = event["result"]
                if "text" in event:
                    state["message"] = event["text"]
        if include_image and state["observations"]:
            state.update(self.preview(state["observations"][-1]))
        return state

    @staticmethod
    def preview(observation):
        """Show the acquisition crop with its symmetry line and brightness direction."""
        preview_data = {"image": None, "image_note": ""}
        try:
            pixels = read_image(observation["image"])
            metrics = observation["metrics"]
            if metrics["crop_xyxy"]:
                automatic = not metrics.get("config", {}).get("crop_xyxy", metrics["crop_xyxy"])
                pixels = crop_image(pixels, metrics["crop_xyxy"], pad=automatic)
            low, high = float(pixels.min()), float(pixels.max())
            scaled = (pixels - low) / (high - low) if high > low else np.zeros_like(pixels)
            preview = Image.fromarray(np.uint8(np.clip(scaled * 255, 0, 255))).convert("RGB")
            preview.thumbnail((1000, 1000))
            fit = metrics.get("fit")
            overlay_note = "No fitted symmetry axis available"
            if fit:
                draw = ImageDraw.Draw(preview)
                sx, sy = preview.width / pixels.shape[1], preview.height / pixels.shape[0]
                point = np.array(fit["axis_point_xy_px"])
                direction = np.array(metrics.get("direction_xy", fit["axis_direction_xy"]))
                low_t, high_t = axis_segment(point, direction, pixels.shape)
                color = "#00ffff" if metrics.get("valid") else "#ff9900"

                def screen(p):
                    return tuple((np.asarray(p) + .5) * [sx, sy] - .5)

                draw.line([screen(point + low_t * direction), screen(point + high_t * direction)],
                          fill=color, width=2)
                if metrics.get("direction_xy") is not None:
                    middle = point + ((low_t + high_t) / 2) * direction
                    tip = middle + min(.28 * min(pixels.shape), .4 * (high_t - low_t)) * direction
                    draw.line([screen(middle), screen(tip)], fill=color, width=4)
                    tip_screen = np.array(screen(tip))
                    arrow = direction * [sx, sy]
                    arrow /= np.linalg.norm(arrow)
                    normal = np.array([-arrow[1], arrow[0]])
                    draw.polygon([tuple(tip_screen), tuple(tip_screen - 12 * arrow + 6 * normal),
                                  tuple(tip_screen - 12 * arrow - 6 * normal)], fill=color)
                overlay_note = ("Symmetry axis: " + ("accepted (cyan)" if metrics.get("valid") else "rejected/failed (orange)")
                                + "; arrow points toward increasing fitted intensity along the axis.")
            buffer = BytesIO()
            preview.save(buffer, format="PNG")
            preview_data["image"] = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
            preview_data["image_note"] = (f"Display intensity range: {low:.5g}–{high:.5g} (linear). "
                                          + overlay_note)
        except (OSError, ValueError, KeyError) as exc:
            preview_data["image_note"] = f"Image preview unavailable: {exc}"
        return preview_data

    def html(self, state=None):
        if state is not None:
            # Embed previews so saved snapshots can browse TIFF acquisitions offline.
            state = dict(state, observations=[dict(o, preview=self.preview(o))
                                               for o in state["observations"]])
        # Escape '<' so image paths and log text cannot terminate the script element.
        payload = json.dumps(state, allow_nan=False).replace("<", "\\u003c")
        return self.template.replace("/* INITIAL_STATE */ null", payload)

    def start(self):
        self.thread.start()

    def close(self):
        try:
            with self.snapshot_path.open("x") as stream:
                stream.write(self.html(self.state()))
            print(f"OSA dashboard snapshot: {self.snapshot_path.resolve()}", flush=True)
        except (OSError, ValueError) as exc:
            print(f"Could not save dashboard snapshot: {exc}", flush=True)
        finally:
            self.server.shutdown()
            self.server.server_close()
            self.thread.join()
