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

"""Browser viewer for the Nebula downsample-mask approximation.

Run inside the Autoware-ML container, for example:

    python -m autoware_ml.tools.visualization.nebula_mask_viewer \
      --data-root /workspace/data/t4dataset \
      --ann-file /workspace/data/t4dataset/info/segdet3d/t4dataset_j6gen2_segdet3d_infos_val.pkl
"""

from __future__ import annotations

import argparse
import json
import pickle
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
import numpy.typing as npt

from autoware_ml.transforms.point_cloud.filters import (
    NebulaDownsampleMaskFilter,
    _nebula_azimuth_rad,
    _normalize_lidar_name,
)


_HTML = r"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Nebula Mask Viewer</title>
  <style>
    :root { color-scheme: dark; font-family: system-ui, sans-serif; }
    body { margin: 0; background: #111; color: #eee; }
    header {
      position: sticky; top: 0; z-index: 1; display: grid; gap: 8px;
      grid-template-columns: auto auto auto 1fr auto; align-items: center;
      padding: 10px 12px; background: #1b1b1b; border-bottom: 1px solid #333;
    }
    button, select, input {
      background: #2a2a2a; color: #eee; border: 1px solid #555;
      border-radius: 4px; padding: 6px 8px;
    }
    button { cursor: pointer; }
    .stats {
      display: flex; gap: 14px; flex-wrap: wrap; font-size: 13px; color: #d8d8d8;
      justify-content: flex-end;
    }
    .mode { color: #aaa; font-size: 13px; }
    main { height: calc(100vh - 58px); }
    #grid2d {
      display: grid; grid-template-columns: repeat(2, minmax(0, 1fr));
      grid-template-rows: repeat(2, minmax(0, 1fr)); height: 100%;
    }
    #view3d { display: none; height: 100%; position: relative; }
    .pane { min-width: 0; min-height: 0; border-right: 1px solid #333; border-bottom: 1px solid #333; display: grid; grid-template-rows: auto 1fr; }
    .pane h2 { margin: 0; padding: 8px 10px; font-size: 14px; font-weight: 600; background: #181818; }
    canvas { width: 100%; height: 100%; display: block; background: #050505; image-rendering: pixelated; }
    #cloud { image-rendering: auto; }
    #legend {
      position: absolute; right: 12px; bottom: 12px; display: grid; gap: 6px;
      background: rgba(20,20,20,0.78); border: 1px solid #444; padding: 8px 10px;
      font-size: 13px;
    }
    .cbar {
      width: 180px; height: 12px; border-radius: 2px; border: 1px solid #666;
      background: linear-gradient(90deg, #2a6fdb, #1fc7e0, #99eb57, #ffe45c);
    }
    .cbar-labels { display: flex; justify-content: space-between; color: #ddd; }
    @media (max-width: 900px) {
      header { grid-template-columns: 1fr 1fr; }
      .stats { justify-content: flex-start; }
      main { height: auto; }
      #grid2d { grid-template-columns: 1fr; height: auto; }
      canvas { height: 32vh; }
      #cloud { height: 72vh; }
    }
  </style>
</head>
<body>
  <header>
    <button id="prev">Prev</button>
    <button id="next">Next</button>
    <label>Frame <input id="frame" type="number" min="0" value="0" style="width:80px"></label>
    <label>LiDAR <select id="source"></select></label>
    <span class="mode" id="mode">2D</span>
    <div class="stats" id="stats"></div>
  </header>
  <main>
    <div id="grid2d">
      <section class="pane"><h2>Greyscale Mask</h2><canvas id="mask_gray"></canvas></section>
      <section class="pane"><h2>B/W Mask</h2><canvas id="mask_bw"></canvas></section>
      <section class="pane"><h2>Unfiltered Pseudo Range</h2><canvas id="unfiltered"></canvas></section>
      <section class="pane"><h2>Filtered Pseudo Range</h2><canvas id="filtered"></canvas></section>
    </div>
    <div id="view3d">
      <canvas id="cloud"></canvas>
      <div id="legend">
        <div>undithered mask value</div>
        <div class="cbar"></div>
        <div class="cbar-labels"><span>0</span><span>255</span></div>
      </div>
    </div>
  </main>
  <script>
    const canvases = ['mask_gray', 'mask_bw', 'unfiltered', 'filtered'].map(id => document.getElementById(id));
    const grid2d = document.getElementById('grid2d');
    const view3d = document.getElementById('view3d');
    const cloudCanvas = document.getElementById('cloud');
    const images = new Map();
    let meta = null;
    let mode = '2d';
    let view = { scale: 1, x: 0, y: 0 };
    let dragging = false;
    let last = { x: 0, y: 0 };
    let cloud = null;
    let glState = null;
    let camera = { yaw: -0.7, pitch: 0.45, distance: 90, panX: 0, panY: 0 };
    let cloudDrag = null;

    function url(path, params) {
      const q = new URLSearchParams(params);
      return `${path}?${q.toString()}`;
    }

    async function loadMeta(frame, source) {
      const params = { frame };
      if (source) params.source = source;
      const response = await fetch(url('/api/frame', params));
      if (!response.ok) throw new Error(await response.text());
      meta = await response.json();
      const select = document.getElementById('source');
      const previous = select.value;
      select.innerHTML = '';
      for (const name of meta.sources) {
        const option = document.createElement('option');
        option.value = name; option.textContent = name;
        select.appendChild(option);
      }
      select.value = meta.source || previous || meta.sources[0];
      document.getElementById('frame').max = Math.max(0, meta.num_frames - 1);
      document.getElementById('frame').value = meta.index;
      document.getElementById('stats').innerHTML = [
        `token ${meta.token}`,
        `source ${meta.source}`,
        `raw ${meta.raw_points.toLocaleString()}`,
        `kept ${meta.kept_points.toLocaleString()}`,
        `drop ${(100 * (1 - meta.keep_ratio)).toFixed(1)}%`,
        `range ${meta.range_min.toFixed(1)}-${meta.range_max.toFixed(1)} m`
      ].map(s => `<span>${s}</span>`).join('');
    }

    async function loadImages() {
      images.clear();
      for (const kind of ['mask_gray', 'mask_bw', 'unfiltered', 'filtered']) {
        const img = new Image();
        img.src = url('/api/image', { frame: meta.index, source: meta.source, kind, t: Date.now() });
        await img.decode();
        images.set(kind, img);
      }
      resetView();
      drawAll();
    }

    async function loadCloud() {
      const response = await fetch(url('/api/pointcloud', { frame: meta.index, source: meta.source, t: Date.now() }));
      if (!response.ok) throw new Error(await response.text());
      const buffer = await response.arrayBuffer();
      cloud = new Float32Array(buffer);
      if (!glState) initGl();
      drawCloud();
    }

    function resetView() {
      const img = images.get('mask_gray');
      const canvas = canvases[0];
      const rect = canvas.getBoundingClientRect();
      view.scale = Math.min(rect.width / img.width, rect.height / img.height);
      view.x = (rect.width - img.width * view.scale) / 2;
      view.y = (rect.height - img.height * view.scale) / 2;
    }

    function drawAll() {
      if (mode !== '2d') return;
      for (const canvas of canvases) {
        const kind = canvas.id;
        const img = images.get(kind);
        const rect = canvas.getBoundingClientRect();
        canvas.width = Math.max(1, Math.floor(rect.width * devicePixelRatio));
        canvas.height = Math.max(1, Math.floor(rect.height * devicePixelRatio));
        const ctx = canvas.getContext('2d');
        ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
        ctx.imageSmoothingEnabled = false;
        ctx.clearRect(0, 0, rect.width, rect.height);
        ctx.drawImage(img, view.x, view.y, img.width * view.scale, img.height * view.scale);
      }
    }

    function setMode(next) {
      mode = next;
      grid2d.style.display = mode === '2d' ? 'grid' : 'none';
      view3d.style.display = mode === '3d' ? 'block' : 'none';
      document.getElementById('mode').textContent = mode.toUpperCase();
      if (mode === '2d') drawAll();
      if (mode === '3d') {
        if (!cloud) loadCloud();
        else drawCloud();
      }
    }

    async function refresh(frame = Number(document.getElementById('frame').value), source = document.getElementById('source').value) {
      await loadMeta(frame, source);
      await loadImages();
      cloud = null;
      if (mode === '3d') await loadCloud();
    }

    document.getElementById('prev').onclick = () => refresh(Math.max(0, meta.index - 1), meta.source);
    document.getElementById('next').onclick = () => refresh(Math.min(meta.num_frames - 1, meta.index + 1), meta.source);
    document.getElementById('frame').onchange = e => refresh(Number(e.target.value), meta.source);
    document.getElementById('source').onchange = e => refresh(meta.index, e.target.value);
    document.addEventListener('keydown', e => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
      if (e.key === '2') setMode('2d');
      if (e.key === '3') setMode('3d');
      if (e.key.toLowerCase() === 'v') setMode(mode === '2d' ? '3d' : '2d');
    });

    for (const canvas of canvases) {
      canvas.addEventListener('wheel', e => {
        e.preventDefault();
        const rect = canvas.getBoundingClientRect();
        const mx = e.clientX - rect.left, my = e.clientY - rect.top;
        const old = view.scale;
        view.scale *= Math.exp(-e.deltaY * 0.001);
        view.scale = Math.max(0.05, Math.min(80, view.scale));
        view.x = mx - (mx - view.x) * (view.scale / old);
        view.y = my - (my - view.y) * (view.scale / old);
        drawAll();
      }, { passive: false });
      canvas.addEventListener('pointerdown', e => {
        dragging = true; last = { x: e.clientX, y: e.clientY }; canvas.setPointerCapture(e.pointerId);
      });
      canvas.addEventListener('pointermove', e => {
        if (!dragging) return;
        view.x += e.clientX - last.x; view.y += e.clientY - last.y;
        last = { x: e.clientX, y: e.clientY }; drawAll();
      });
      canvas.addEventListener('pointerup', () => dragging = false);
      canvas.addEventListener('dblclick', resetView);
    }
    function initGl() {
      const gl = cloudCanvas.getContext('webgl', { antialias: false });
      const vs = `
        attribute vec4 a_point;
        uniform mat4 u_mvp;
        varying float v_mask;
        void main() {
          gl_Position = u_mvp * vec4(a_point.xyz, 1.0);
          gl_PointSize = 2.0;
          v_mask = a_point.w;
        }`;
      const fs = `
        precision mediump float;
        varying float v_mask;
        void main() {
          vec3 c0 = vec3(0.16, 0.44, 0.86);
          vec3 c1 = vec3(0.12, 0.78, 0.88);
          vec3 c2 = vec3(0.60, 0.92, 0.34);
          vec3 c3 = vec3(1.00, 0.89, 0.36);
          vec3 color = v_mask < 0.333
            ? mix(c0, c1, v_mask / 0.333)
            : (v_mask < 0.666
              ? mix(c1, c2, (v_mask - 0.333) / 0.333)
              : mix(c2, c3, (v_mask - 0.666) / 0.334));
          gl_FragColor = vec4(color, 1.0);
        }`;
      const program = createProgram(gl, vs, fs);
      glState = {
        gl, program,
        buffer: gl.createBuffer(),
        aPoint: gl.getAttribLocation(program, 'a_point'),
        uMvp: gl.getUniformLocation(program, 'u_mvp'),
      };
    }

    function createProgram(gl, vsSource, fsSource) {
      const vs = compileShader(gl, gl.VERTEX_SHADER, vsSource);
      const fs = compileShader(gl, gl.FRAGMENT_SHADER, fsSource);
      const program = gl.createProgram();
      gl.attachShader(program, vs); gl.attachShader(program, fs); gl.linkProgram(program);
      if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
      return program;
    }

    function compileShader(gl, type, source) {
      const shader = gl.createShader(type);
      gl.shaderSource(shader, source); gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
      return shader;
    }

    function drawCloud() {
      if (!glState || !cloud || mode !== '3d') return;
      const { gl, program, buffer, aPoint, uMvp } = glState;
      const rect = cloudCanvas.getBoundingClientRect();
      cloudCanvas.width = Math.max(1, Math.floor(rect.width * devicePixelRatio));
      cloudCanvas.height = Math.max(1, Math.floor(rect.height * devicePixelRatio));
      gl.viewport(0, 0, cloudCanvas.width, cloudCanvas.height);
      gl.clearColor(0.02, 0.02, 0.02, 1); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      gl.enable(gl.DEPTH_TEST); gl.useProgram(program);
      gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
      gl.bufferData(gl.ARRAY_BUFFER, cloud, gl.STATIC_DRAW);
      gl.enableVertexAttribArray(aPoint);
      gl.vertexAttribPointer(aPoint, 4, gl.FLOAT, false, 16, 0);
      gl.uniformMatrix4fv(uMvp, false, mvp(rect.width / Math.max(1, rect.height)));
      gl.drawArrays(gl.POINTS, 0, cloud.length / 4);
    }

    function mvp(aspect) {
      const proj = perspective(45 * Math.PI / 180, aspect, 0.1, 500);
      const cp = Math.cos(camera.pitch), sp = Math.sin(camera.pitch);
      const cy = Math.cos(camera.yaw), sy = Math.sin(camera.yaw);
      const eye = [
        camera.distance * cp * sy,
        camera.distance * cp * cy,
        camera.distance * sp,
      ];
      const target = [camera.panX, camera.panY, 0];
      return multiply(proj, lookAt(eye, target, [0, 0, 1]));
    }

    function perspective(fovy, aspect, near, far) {
      const f = 1 / Math.tan(fovy / 2), nf = 1 / (near - far);
      return new Float32Array([
        f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, (far + near) * nf, -1,
        0, 0, 2 * far * near * nf, 0
      ]);
    }

    function lookAt(eye, center, up) {
      const z = norm(sub(eye, center));
      const x = norm(cross(up, z));
      const y = cross(z, x);
      return new Float32Array([
        x[0], y[0], z[0], 0, x[1], y[1], z[1], 0, x[2], y[2], z[2], 0,
        -dot(x, eye), -dot(y, eye), -dot(z, eye), 1
      ]);
    }

    function multiply(a, b) {
      const out = new Float32Array(16);
      for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
        out[c * 4 + r] = a[r] * b[c * 4] + a[4 + r] * b[c * 4 + 1] + a[8 + r] * b[c * 4 + 2] + a[12 + r] * b[c * 4 + 3];
      }
      return out;
    }

    function sub(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
    function dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
    function cross(a, b) { return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]; }
    function norm(v) { const l = Math.hypot(v[0], v[1], v[2]) || 1; return [v[0] / l, v[1] / l, v[2] / l]; }

    cloudCanvas.addEventListener('wheel', e => {
      e.preventDefault();
      camera.distance *= Math.exp(e.deltaY * 0.001);
      camera.distance = Math.max(2, Math.min(350, camera.distance));
      drawCloud();
    }, { passive: false });
    cloudCanvas.addEventListener('pointerdown', e => {
      cloudDrag = { x: e.clientX, y: e.clientY, button: e.button, shift: e.shiftKey };
      cloudCanvas.setPointerCapture(e.pointerId);
    });
    cloudCanvas.addEventListener('pointermove', e => {
      if (!cloudDrag) return;
      const dx = e.clientX - cloudDrag.x, dy = e.clientY - cloudDrag.y;
      cloudDrag.x = e.clientX; cloudDrag.y = e.clientY;
      if (cloudDrag.shift) {
        camera.panX -= dx * camera.distance * 0.0015;
        camera.panY += dy * camera.distance * 0.0015;
      } else {
        camera.yaw += dx * 0.008;
        camera.pitch = Math.max(-1.45, Math.min(1.45, camera.pitch + dy * 0.008));
      }
      drawCloud();
    });
    cloudCanvas.addEventListener('pointerup', () => cloudDrag = null);
    cloudCanvas.addEventListener('dblclick', () => {
      camera = { yaw: -0.7, pitch: 0.45, distance: 90, panX: 0, panY: 0 };
      drawCloud();
    });

    window.onresize = () => { drawAll(); drawCloud(); };
    refresh(0, '');
  </script>
</body>
</html>
"""


class ViewerData:
    """Cached access to annotation records and projected range images."""

    def __init__(self, data_root: Path, ann_file: Path, max_frames: int | None) -> None:
        self.data_root = data_root
        self.ann_file = ann_file
        with open(ann_file, "rb") as file:
            data = pickle.load(file)
        frames = data["data_list"]
        self.frames = frames[:max_frames] if max_frames is not None else frames
        self.filter = NebulaDownsampleMaskFilter(return_stats=True)

    def frame_meta(self, index: int, source: str | None) -> dict[str, Any]:
        frame = self._frame(index)
        source = source or self.sources(index)[0]
        projection = self._projection(index, source)
        ranges = projection["range"]
        kept = projection["keep_mask"].sum()
        return {
            "index": index,
            "num_frames": len(self.frames),
            "token": frame.get("token", str(index)),
            "source": source,
            "sources": self.sources(index),
            "raw_points": int(projection["local_points"].shape[0]),
            "kept_points": int(kept),
            "keep_ratio": float(kept / max(1, projection["local_points"].shape[0])),
            "range_min": float(ranges.min()) if ranges.size else 0.0,
            "range_max": float(ranges.max()) if ranges.size else 0.0,
        }

    def sources(self, index: int) -> list[str]:
        frame = self._frame(index)
        sources = [
            name
            for name in frame.get("lidar_sources", {})
            if name != "LIDAR_CONCAT"
            and _normalize_lidar_name(name) in self.filter.lidar_name_to_mask
        ]
        if not sources:
            raise ValueError(f"Frame {index} has no supported lidar_sources.")
        return sources

    def image_png(self, index: int, source: str, kind: str) -> bytes:
        projection = self._projection(index, source)
        if kind == "mask_gray":
            image = projection["mask_gray_image"]
        elif kind == "mask_bw":
            image = projection["mask_bw_image"]
        elif kind == "unfiltered":
            image = self._range_image(
                projection["height"],
                projection["width"],
                projection["rows"],
                projection["cols"],
                projection["range"],
            )
        elif kind == "filtered":
            keep = projection["keep_mask"]
            image = self._range_image(
                projection["height"],
                projection["width"],
                projection["rows"][keep],
                projection["cols"][keep],
                projection["range"][keep],
            )
        else:
            raise ValueError(f"Unknown image kind: {kind}")
        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            raise RuntimeError("Failed to encode PNG.")
        return encoded.tobytes()

    def pointcloud_binary(self, index: int, source: str) -> bytes:
        projection = self._projection(index, source)
        local_points = projection["local_points"].astype(np.float32)
        mask_values = projection["mask_values"].astype(np.float32).reshape(-1, 1)
        points = np.concatenate([local_points, mask_values], axis=1)
        return points.astype(np.float32, copy=False).tobytes()

    def _frame(self, index: int) -> dict[str, Any]:
        if index < 0 or index >= len(self.frames):
            raise IndexError(f"Frame index {index} is outside [0, {len(self.frames)}).")
        return self.frames[index]

    @lru_cache(maxsize=24)
    def _projection(self, index: int, source: str) -> dict[str, Any]:
        frame = self._frame(index)
        lidar_path = self._resolve(frame["lidar_points"]["lidar_path"])
        load_dim = int(frame["lidar_points"].get("num_pts_feats", 7))
        points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, load_dim)
        source_slice = self._source_slice(frame, source)
        source_points = points[source_slice, :3]
        source_meta = frame["lidar_sources"][source]
        translation = np.asarray(source_meta["translation"], dtype=np.float32)
        rotation = np.asarray(source_meta["rotation"], dtype=np.float32)
        local_points = (source_points - translation) @ rotation

        lidar_name = _normalize_lidar_name(source)
        model_name = self.filter.lidar_name_to_model[lidar_name]
        calibration = self.filter._load_calibration(lidar_name)
        mask_bw = self.filter._load_mask(lidar_name, model_name)
        mask_gray = self._load_gray_mask(lidar_name)
        rows = self._nearest_channel(local_points, calibration.elevation_rad)
        azimuth = np.rad2deg(_nebula_azimuth_rad(local_points))
        if self.filter.use_calibration_azimuth_offsets:
            azimuth = azimuth - calibration.azimuth_deg[rows]
        cols = np.floor(((azimuth % 360.0) / 360.0) * mask_bw.shape[1] + 0.5).astype(np.int64)
        valid = (cols >= 0) & (cols < mask_bw.shape[1]) & (rows >= 0) & (rows < mask_bw.shape[0])
        keep_mask = np.zeros(local_points.shape[0], dtype=bool)
        keep_mask[valid] = mask_bw[rows[valid], cols[valid]]
        mask_values = np.zeros(local_points.shape[0], dtype=np.float32)
        mask_values[valid] = mask_gray[rows[valid], cols[valid]].astype(np.float32) / 255.0
        ranges = np.linalg.norm(local_points, axis=1)

        return {
            "local_points": local_points,
            "rows": rows,
            "cols": cols,
            "range": ranges,
            "keep_mask": keep_mask,
            "mask_values": mask_values,
            "height": mask_bw.shape[0],
            "width": mask_bw.shape[1],
            "mask_gray_image": mask_gray,
            "mask_bw_image": (mask_bw.astype(np.uint8) * 255),
        }

    def _load_gray_mask(self, lidar_name: str) -> npt.NDArray[np.uint8]:
        mask_path = self.filter.mask_root / self.filter.lidar_name_to_mask[lidar_name]
        image = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not read Nebula downsample mask: {mask_path}")
        return image

    def _source_slice(self, frame: dict[str, Any], source: str) -> slice:
        token = frame["lidar_sources"][source]["sensor_token"]
        for source_info in frame["lidar_sources_info"]["sources"]:
            if source_info["sensor_token"] == token:
                start = int(source_info["idx_begin"])
                return slice(start, start + int(source_info["length"]))
        raise ValueError(f"No lidar_sources_info entry found for {source}.")

    def _resolve(self, path: str) -> Path:
        resolved = Path(path)
        if resolved.is_absolute():
            return resolved
        return self.data_root / resolved

    @staticmethod
    def _nearest_channel(
        points: npt.NDArray[np.float32], elevation_rad: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.int64]:
        xy_norm = np.linalg.norm(points[:, :2], axis=1)
        point_elevation = np.arctan2(points[:, 2], xy_norm)
        return np.abs(point_elevation[:, None] - elevation_rad[None, :]).argmin(axis=1)

    @staticmethod
    def _range_image(
        height: int,
        width: int,
        rows: npt.NDArray[np.int64],
        cols: npt.NDArray[np.int64],
        ranges: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.uint8]:
        image = np.zeros((height, width), dtype=np.float32)
        valid = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
        if not np.any(valid):
            return image.astype(np.uint8)
        rows = rows[valid]
        cols = cols[valid]
        ranges = ranges[valid]
        order = np.argsort(ranges)[::-1]
        image[rows[order], cols[order]] = ranges[order]
        nonzero = image[image > 0.0]
        if nonzero.size == 0:
            return image.astype(np.uint8)
        low, high = np.percentile(nonzero, [2.0, 98.0])
        if high <= low:
            high = low + 1.0
        normalized = np.clip((image - low) / (high - low), 0.0, 1.0)
        colored = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        colored[image == 0.0] = 0
        return colored


def _make_handler(data: ViewerData) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            try:
                parsed = urlparse(self.path)
                if parsed.path == "/":
                    self._send(HTTPStatus.OK, _HTML.encode(), "text/html; charset=utf-8")
                elif parsed.path == "/api/frame":
                    query = parse_qs(parsed.query)
                    index = int(query.get("frame", ["0"])[0])
                    source = query.get("source", [None])[0] or None
                    payload = json.dumps(data.frame_meta(index, source)).encode()
                    self._send(HTTPStatus.OK, payload, "application/json")
                elif parsed.path == "/api/image":
                    query = parse_qs(parsed.query)
                    index = int(query.get("frame", ["0"])[0])
                    source = query["source"][0]
                    kind = query["kind"][0]
                    self._send(HTTPStatus.OK, data.image_png(index, source, kind), "image/png")
                elif parsed.path == "/api/pointcloud":
                    query = parse_qs(parsed.query)
                    index = int(query.get("frame", ["0"])[0])
                    source = query["source"][0]
                    self._send(
                        HTTPStatus.OK,
                        data.pointcloud_binary(index, source),
                        "application/octet-stream",
                    )
                else:
                    self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")
            except Exception as exc:  # noqa: BLE001 - surface errors in browser during inspection.
                self._send(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    f"{type(exc).__name__}: {exc}".encode(),
                    "text/plain",
                )

        def log_message(self, format: str, *args: Any) -> None:
            print(f"{self.address_string()} - {format % args}")

        def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/workspace/data/t4dataset"))
    parser.add_argument(
        "--ann-file",
        type=Path,
        default=Path(
            "/workspace/data/t4dataset/info/segdet3d/t4dataset_j6gen2_segdet3d_infos_val.pkl"
        ),
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18765)
    parser.add_argument("--max-frames", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = ViewerData(args.data_root, args.ann_file, args.max_frames)
    server = ThreadingHTTPServer((args.host, args.port), _make_handler(data))
    print(f"Serving Nebula mask viewer at http://{args.host}:{args.port}")
    print(f"Annotation file: {args.ann_file}")
    server.serve_forever()


if __name__ == "__main__":
    main()
