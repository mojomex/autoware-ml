# Copyright 2026 TIER IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Browser overlay for matched rosbag concat and T4Dataset pointclouds.

Run inside the Autoware-ML container, for example:

    python3 -m autoware_ml.tools.visualization.matched_pointcloud_overlay \
      --mcap /workspace/pa-tools-recordings/b7d25e03-1785756240/b7d25e03-1785756240_0.mcap \
      --data-root /workspace/data/t4dataset \
      --ann-file /workspace/data/t4dataset/info/segdet3d/t4dataset_j6gen2_segdet3d_infos_val.pkl \
      --scene-frame-count 200 \
      --host 0.0.0.0 \
      --port 8766
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np
import numpy.typing as npt
from mcap.reader import make_reader

from autoware_ml.tools.dataset.t4dataset.compare_nebula_order_to_raw import (
    _CdrReader,
    _expand_scene_records,
)
from autoware_ml.tools.dataset.t4dataset.nebula_order_error import _stamp_to_seconds


_CONCAT_TOPIC = "/sensing/lidar/concatenated/pointcloud"


_HTML = r"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Matched Pointcloud Overlay</title>
  <style>
    :root { color-scheme: dark; font-family: system-ui, sans-serif; }
    body { margin: 0; background: #050505; color: #f2f2f2; overflow: hidden; }
    header {
      height: 50px; box-sizing: border-box; display: grid; grid-template-columns: auto auto auto auto 1fr;
      gap: 10px; align-items: center; padding: 8px 12px; background: #171717; border-bottom: 1px solid #333;
    }
    button, input {
      background: #2a2a2a; color: #f2f2f2; border: 1px solid #555; border-radius: 4px; padding: 6px 8px;
    }
    label { display: inline-flex; align-items: center; gap: 6px; font-size: 14px; }
    .stats { display: flex; gap: 14px; flex-wrap: wrap; justify-content: flex-end; color: #ddd; font-size: 13px; }
    #canvas { width: 100vw; height: calc(100vh - 50px); display: block; }
    #legend {
      position: fixed; right: 12px; bottom: 12px; background: rgba(10,10,10,0.78);
      border: 1px solid #444; padding: 10px 12px; display: grid; gap: 8px; font-size: 13px;
    }
    .row { display: grid; grid-template-columns: 14px auto; gap: 8px; align-items: center; }
    .swatch { width: 14px; height: 14px; border-radius: 2px; }
    #help {
      position: fixed; left: 12px; bottom: 12px; color: #bbb; font-size: 12px;
      background: rgba(10,10,10,0.65); border: 1px solid #333; padding: 8px 10px;
    }
  </style>
</head>
<body>
  <header>
    <button id="prev">Prev</button>
    <button id="next">Next</button>
    <label>Frame <input id="frame" type="number" min="0" value="0" style="width:80px"></label>
    <label>Max points <input id="maxPoints" type="number" min="1000" step="10000" value="200000" style="width:95px"></label>
    <div class="stats" id="stats"></div>
  </header>
  <canvas id="canvas"></canvas>
  <div id="legend">
    <label><input id="showRaw" type="checkbox" checked> <span class="row"><span class="swatch" style="background:#ffb000"></span>rosbag concat</span></label>
    <label><input id="showT4" type="checkbox" checked> <span class="row"><span class="swatch" style="background:#00d7ff"></span>T4 unfiltered</span></label>
  </div>
  <div id="help">Left drag: orbit. Right drag: pan. Wheel: zoom. A/D or arrows: frame.</div>
<script>
const canvas = document.getElementById('canvas');
const gl = canvas.getContext('webgl', { antialias: false, preserveDrawingBuffer: false });
if (!gl) document.body.innerHTML = 'WebGL is required.';

let meta = null;
let rawPoints = new Float32Array();
let t4Points = new Float32Array();
let camera = { yaw: -0.7, pitch: 0.5, distance: 95, panX: 0, panY: 0 };
let drag = null;

const vertexShader = `
attribute vec3 position;
uniform mat4 mvp;
uniform vec3 color;
uniform float pointSize;
varying vec3 vColor;
void main() {
  gl_Position = mvp * vec4(position, 1.0);
  gl_PointSize = pointSize;
  vColor = color;
}`;
const fragmentShader = `
precision mediump float;
varying vec3 vColor;
void main() {
  vec2 d = gl_PointCoord - vec2(0.5);
  if (dot(d, d) > 0.25) discard;
  gl_FragColor = vec4(vColor, 0.72);
}`;

function shader(type, source) {
  const s = gl.createShader(type);
  gl.shaderSource(s, source);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
  return s;
}
const program = gl.createProgram();
gl.attachShader(program, shader(gl.VERTEX_SHADER, vertexShader));
gl.attachShader(program, shader(gl.FRAGMENT_SHADER, fragmentShader));
gl.linkProgram(program);
if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
gl.useProgram(program);
const loc = {
  position: gl.getAttribLocation(program, 'position'),
  mvp: gl.getUniformLocation(program, 'mvp'),
  color: gl.getUniformLocation(program, 'color'),
  pointSize: gl.getUniformLocation(program, 'pointSize')
};
const rawBuffer = gl.createBuffer();
const t4Buffer = gl.createBuffer();

function url(path, params) {
  return `${path}?${new URLSearchParams(params).toString()}`;
}

async function loadFrame(index) {
  const maxPoints = document.getElementById('maxPoints').value;
  const response = await fetch(url('/api/frame', { index, max_points: maxPoints }));
  if (!response.ok) throw new Error(await response.text());
  meta = await response.json();
  document.getElementById('frame').value = meta.index;
  document.getElementById('frame').max = Math.max(0, meta.num_frames - 1);
  document.getElementById('stats').innerHTML = [
    `stamp diff ${(meta.stamp_diff_sec * 1000).toFixed(3)} ms`,
    `raw ${meta.raw_points.toLocaleString()} (${meta.raw_sent.toLocaleString()})`,
    `t4 ${meta.t4_points.toLocaleString()} (${meta.t4_sent.toLocaleString()})`,
    `ratio ${(meta.t4_points / Math.max(1, meta.raw_points)).toFixed(2)}x`
  ].map(s => `<span>${s}</span>`).join('');

  rawPoints = new Float32Array(await (await fetch(url('/api/points', { index: meta.index, cloud: 'raw', max_points: maxPoints }))).arrayBuffer());
  t4Points = new Float32Array(await (await fetch(url('/api/points', { index: meta.index, cloud: 't4', max_points: maxPoints }))).arrayBuffer());
  gl.bindBuffer(gl.ARRAY_BUFFER, rawBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, rawPoints, gl.STATIC_DRAW);
  gl.bindBuffer(gl.ARRAY_BUFFER, t4Buffer);
  gl.bufferData(gl.ARRAY_BUFFER, t4Points, gl.STATIC_DRAW);
  draw();
}

function resize() {
  const dpr = window.devicePixelRatio || 1;
  const w = Math.floor(canvas.clientWidth * dpr);
  const h = Math.floor(canvas.clientHeight * dpr);
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w; canvas.height = h;
  }
  gl.viewport(0, 0, canvas.width, canvas.height);
}

function matMul(a, b) {
  const out = new Float32Array(16);
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
    out[c*4+r] = a[0*4+r]*b[c*4+0] + a[1*4+r]*b[c*4+1] + a[2*4+r]*b[c*4+2] + a[3*4+r]*b[c*4+3];
  }
  return out;
}
function perspective(fov, aspect, near, far) {
  const f = 1 / Math.tan(fov / 2), nf = 1 / (near - far);
  return new Float32Array([f/aspect,0,0,0, 0,f,0,0, 0,0,(far+near)*nf,-1, 0,0,2*far*near*nf,0]);
}
function lookAt(eye, target, up) {
  const z = norm(sub(eye, target));
  const x = norm(cross(up, z));
  const y = cross(z, x);
  return new Float32Array([x[0],y[0],z[0],0, x[1],y[1],z[1],0, x[2],y[2],z[2],0, -dot(x,eye),-dot(y,eye),-dot(z,eye),1]);
}
function sub(a,b){ return [a[0]-b[0],a[1]-b[1],a[2]-b[2]]; }
function dot(a,b){ return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }
function cross(a,b){ return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]; }
function norm(a){ const l=Math.hypot(a[0],a[1],a[2]) || 1; return [a[0]/l,a[1]/l,a[2]/l]; }

function mvp() {
  const cp = Math.cos(camera.pitch), sp = Math.sin(camera.pitch);
  const cy = Math.cos(camera.yaw), sy = Math.sin(camera.yaw);
  const eye = [camera.distance * cp * cy + camera.panX, camera.distance * cp * sy + camera.panY, camera.distance * sp];
  const target = [camera.panX, camera.panY, 0];
  return matMul(perspective(60 * Math.PI/180, canvas.width / Math.max(1, canvas.height), 0.1, 700), lookAt(eye, target, [0,0,1]));
}

function drawCloud(buffer, count, color, size) {
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.enableVertexAttribArray(loc.position);
  gl.vertexAttribPointer(loc.position, 3, gl.FLOAT, false, 0, 0);
  gl.uniform3fv(loc.color, color);
  gl.uniform1f(loc.pointSize, size * (window.devicePixelRatio || 1));
  gl.drawArrays(gl.POINTS, 0, count);
}

function draw() {
  resize();
  gl.clearColor(0.01, 0.01, 0.012, 1);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE);
  gl.disable(gl.DEPTH_TEST);
  gl.useProgram(program);
  gl.uniformMatrix4fv(loc.mvp, false, mvp());
  if (document.getElementById('showRaw').checked) drawCloud(rawBuffer, rawPoints.length / 3, [1.0, 0.69, 0.0], 2.0);
  if (document.getElementById('showT4').checked) drawCloud(t4Buffer, t4Points.length / 3, [0.0, 0.84, 1.0], 1.5);
}

canvas.addEventListener('contextmenu', e => e.preventDefault());
canvas.addEventListener('pointerdown', e => { drag = { x: e.clientX, y: e.clientY, button: e.button }; canvas.setPointerCapture(e.pointerId); });
canvas.addEventListener('pointermove', e => {
  if (!drag) return;
  const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
  drag.x = e.clientX; drag.y = e.clientY;
  if (drag.button === 2) {
    camera.panX -= dx * camera.distance / canvas.clientWidth;
    camera.panY += dy * camera.distance / canvas.clientHeight;
  } else {
    camera.yaw -= dx * 0.006;
    camera.pitch = Math.max(-1.45, Math.min(1.45, camera.pitch + dy * 0.006));
  }
  draw();
});
canvas.addEventListener('pointerup', () => { drag = null; });
canvas.addEventListener('wheel', e => {
  e.preventDefault();
  camera.distance = Math.max(2, Math.min(500, camera.distance * Math.exp(e.deltaY * 0.001)));
  draw();
}, { passive: false });
window.addEventListener('resize', draw);

async function step(delta) {
  const index = Math.max(0, Math.min(meta.num_frames - 1, Number(document.getElementById('frame').value) + delta));
  await loadFrame(index);
}
document.getElementById('prev').onclick = () => step(-1);
document.getElementById('next').onclick = () => step(1);
document.getElementById('frame').addEventListener('change', e => loadFrame(Number(e.target.value)));
document.getElementById('maxPoints').addEventListener('change', () => loadFrame(Number(document.getElementById('frame').value)));
document.getElementById('showRaw').addEventListener('change', draw);
document.getElementById('showT4').addEventListener('change', draw);
window.addEventListener('keydown', e => {
  if (e.key === 'ArrowLeft' || e.key === 'a') step(-1);
  if (e.key === 'ArrowRight' || e.key === 'd') step(1);
});

loadFrame(0).catch(err => {
  console.error(err);
  document.getElementById('stats').textContent = String(err);
});
</script>
</body>
</html>
"""


@dataclass(frozen=True)
class ConcatMessage:
    stamp: float
    data: bytes


class OverlayData:
    def __init__(
        self,
        *,
        mcap: Path,
        data_root: Path,
        ann_file: Path,
        max_frames: int,
        start_index: int,
        frame_stride: int,
        scene_frame_count: int,
    ) -> None:
        self.mcap = mcap
        self.data_root = data_root
        self.records = _load_selected_records(
            ann_file=ann_file,
            data_root=data_root,
            max_frames=max_frames,
            start_index=start_index,
            frame_stride=frame_stride,
            scene_frame_count=scene_frame_count,
        )
        self.concat_messages = _load_concat_messages(mcap)
        self.concat_stamps = np.asarray([message.stamp for message in self.concat_messages])

    def frame_meta(self, index: int, max_points: int) -> dict[str, Any]:
        record, raw_points, t4_points, stamp_diff = self._matched_points(index)
        return {
            "index": index,
            "num_frames": len(self.records),
            "token": str(record.get("token")),
            "reference_stamp": _stamp_to_seconds(record["lidar_sources_info"]["stamp"]),
            "concat_stamp": self._nearest_concat(index).stamp,
            "stamp_diff_sec": stamp_diff,
            "raw_points": int(raw_points.shape[0]),
            "t4_points": int(t4_points.shape[0]),
            "raw_sent": int(min(max_points, raw_points.shape[0])),
            "t4_sent": int(min(max_points, t4_points.shape[0])),
        }

    def point_bytes(self, index: int, cloud: str, max_points: int) -> bytes:
        _, raw_points, t4_points, _ = self._matched_points(index)
        points = raw_points if cloud == "raw" else t4_points
        sampled = _sample_points(points, max_points)
        return np.ascontiguousarray(sampled.astype(np.float32, copy=False)).tobytes()

    @lru_cache(maxsize=16)
    def _matched_points(
        self, index: int
    ) -> tuple[dict[str, Any], npt.NDArray[np.float32], npt.NDArray[np.float32], float]:
        if index < 0 or index >= len(self.records):
            raise IndexError(f"Frame index {index} is outside [0, {len(self.records)}).")
        record = self.records[index]
        reference_stamp = _stamp_to_seconds(record["lidar_sources_info"]["stamp"])
        concat = self._nearest_concat(index)
        raw_points = _parse_concat_pointcloud2(concat.data)
        t4_points = _load_t4_points(self.data_root, record)
        return record, raw_points, t4_points, abs(concat.stamp - reference_stamp)

    def _nearest_concat(self, index: int) -> ConcatMessage:
        record = self.records[index]
        reference_stamp = _stamp_to_seconds(record["lidar_sources_info"]["stamp"])
        nearest = int(np.abs(self.concat_stamps - reference_stamp).argmin())
        return self.concat_messages[nearest]


class Handler(BaseHTTPRequestHandler):
    data: OverlayData

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        try:
            if parsed.path == "/":
                self._send_bytes(_HTML.encode(), "text/html; charset=utf-8")
            elif parsed.path == "/api/frame":
                index = _int_param(params, "index", 0)
                max_points = _int_param(params, "max_points", 200_000)
                self._send_json(self.data.frame_meta(index, max_points))
            elif parsed.path == "/api/points":
                index = _int_param(params, "index", 0)
                max_points = _int_param(params, "max_points", 200_000)
                cloud = params.get("cloud", ["raw"])[0]
                if cloud not in {"raw", "t4"}:
                    raise ValueError("cloud must be 'raw' or 't4'.")
                self._send_bytes(
                    self.data.point_bytes(index, cloud, max_points),
                    "application/octet-stream",
                )
            else:
                self.send_error(HTTPStatus.NOT_FOUND, "Not found")
        except Exception as error:  # noqa: BLE001
            self.send_error(HTTPStatus.BAD_REQUEST, str(error))

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        print(f"{self.address_string()} - {format % args}")

    def _send_json(self, value: Any) -> None:
        self._send_bytes(json.dumps(value).encode(), "application/json")

    def _send_bytes(self, data: bytes, content_type: str) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mcap", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("/workspace/data/t4dataset"))
    parser.add_argument(
        "--ann-file",
        type=Path,
        default=Path(
            "/workspace/data/t4dataset/info/segdet3d/t4dataset_j6gen2_segdet3d_infos_val.pkl"
        ),
    )
    parser.add_argument("--max-frames", type=int, default=20)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--scene-frame-count", type=int, default=0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()

    Handler.data = OverlayData(
        mcap=args.mcap,
        data_root=args.data_root,
        ann_file=args.ann_file,
        max_frames=args.max_frames,
        start_index=args.start_index,
        frame_stride=args.frame_stride,
        scene_frame_count=args.scene_frame_count,
    )
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Serving matched pointcloud overlay at http://{args.host}:{args.port}")
    server.serve_forever()


def _load_selected_records(
    *,
    ann_file: Path,
    data_root: Path,
    max_frames: int,
    start_index: int,
    frame_stride: int,
    scene_frame_count: int,
) -> list[dict[str, Any]]:
    with ann_file.open("rb") as file:
        data = pickle.load(file)
    records = list(data["data_list"] if isinstance(data, dict) else data)
    selected = records[start_index::frame_stride][:max_frames]
    if scene_frame_count > 0:
        selected = _expand_scene_records(selected[0], data_root, scene_frame_count)
    return selected


def _load_concat_messages(mcap_path: Path) -> list[ConcatMessage]:
    messages = []
    with mcap_path.open("rb") as file:
        reader = make_reader(file)
        for _, channel, message in reader.iter_messages(topics=[_CONCAT_TOPIC]):
            messages.append(
                ConcatMessage(stamp=_pointcloud2_stamp(message.data), data=message.data)
            )
    if not messages:
        raise ValueError(f"No messages found for topic {_CONCAT_TOPIC}.")
    messages.sort(key=lambda message: message.stamp)
    return messages


def _pointcloud2_stamp(data: bytes) -> float:
    reader = _CdrReader(data)
    sec = reader.i32()
    nanosec = reader.u32()
    return sec + nanosec * 1e-9


def _parse_concat_pointcloud2(data: bytes) -> npt.NDArray[np.float32]:
    reader = _CdrReader(data)
    reader.i32()
    reader.u32()
    reader.string()
    height = reader.u32()
    width = reader.u32()
    fields = {}
    for _ in range(reader.u32()):
        name = reader.string()
        fields[name] = (reader.u32(), reader.u8(), reader.u32())
    reader.bool()
    point_step = reader.u32()
    reader.u32()
    blob = reader.byte_sequence()
    reader.bool()
    count = int(width * height)
    dtype = np.dtype(
        {
            "names": ["x", "y", "z"],
            "formats": ["<f4", "<f4", "<f4"],
            "offsets": [fields["x"][0], fields["y"][0], fields["z"][0]],
            "itemsize": point_step,
        }
    )
    raw = np.frombuffer(blob, dtype=dtype, count=count)
    points = np.column_stack([raw["x"], raw["y"], raw["z"]]).astype(np.float32, copy=False)
    return points[np.isfinite(points).all(axis=1)]


def _load_t4_points(data_root: Path, record: dict[str, Any]) -> npt.NDArray[np.float32]:
    lidar_points = record["lidar_points"]
    point_path = data_root / (lidar_points.get("lidar_path") or lidar_points["filename"])
    load_dim = int(lidar_points.get("num_pts_feats", 7))
    return np.fromfile(point_path, dtype=np.float32).reshape(-1, load_dim)[:, :3].copy()


def _sample_points(points: npt.NDArray[np.float32], max_points: int) -> npt.NDArray[np.float32]:
    if max_points <= 0 or points.shape[0] <= max_points:
        return points
    stride = int(np.ceil(points.shape[0] / max_points))
    return points[::stride][:max_points]


def _int_param(params: dict[str, list[str]], name: str, default: int) -> int:
    return int(params.get(name, [str(default)])[0])


if __name__ == "__main__":
    main()
