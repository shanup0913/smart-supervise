"""
MapVisualizer 2.5: Interactive 2D Surveillance Radar, Floorplan Blueprint & CCTV FOV Coverage Visualizer.

Key Capabilities:
1. Rotatable 360° CCTV Vision Cones (Field of View):
   - Realistic 70° vision wedges projecting from each camera.
   - Dual-zone interaction: click & drag center to reposition, drag outer ring to rotate 360° heading.
2. DORI Range Rings (IEC/EN 62676-4 Standard):
   - Identification Zone (Inner - 35%): sharp facial biometrics (YuNet).
   - Recognition Zone (Mid - 65%): whole-body clothing & color Re-ID.
   - Detection Zone (Outer - 100%): YOLOv8 presence & motion alert.
3. Multi-Camera Overlap Fusion Indicator:
   - Synchronized dual-camera pulsing and shared intersection links when target enters overlap zone.
4. Custom Blueprint / Floorplan Upload (.png, .jpg, .jpeg, .bmp, .webp) with dark-tinted contrast overlay.
5. Persistent Configuration: saves coordinates, angles, and blueprint path to models/map_config.json.
6. Bidirectional Bezier Paths & Sequential Blinking Tracer Loop.
"""

import os
import json
import math
import tkinter as tk
from PIL import Image, ImageTk

DEFAULT_NODE_POSITIONS = [
    (0.22, 0.28),  # Camera 1 (North Entrance / Gate)
    (0.78, 0.28),  # Camera 2 (East Corridor Wing A)
    (0.78, 0.72),  # Camera 3 (Room 204 / South Wing)
    (0.22, 0.72),  # Camera 4 (Cafeteria / West Plaza)
]

DEFAULT_NODE_ANGLES = [
    45.0,   # Camera 1: Facing South-East (45°) towards center
    135.0,  # Camera 2: Facing South-West (135°) towards center
    225.0,  # Camera 3: Facing North-West (225°) towards center
    315.0,  # Camera 4: Facing North-East (315°) towards center
]


class MapVisualizer:
    def __init__(self):
        # Resolve path to models/map_config.json
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.config_dir = os.path.join(base_dir, "models")
        self.config_path = os.path.join(self.config_dir, "map_config.json")

        self.node_positions = list(DEFAULT_NODE_POSITIONS)
        self.node_angles = list(DEFAULT_NODE_ANGLES)
        self.show_fov_cones = True
        self.fov_span_deg = 70.0
        self.custom_blueprint_path = None

        # Image caching to avoid re-decoding image every 60ms
        self._cached_blueprint_tk = None
        self._cached_blueprint_dim = None

        # Interactive Drag, Rotation & Hover State
        self.dragged_node_idx = None
        self.rotating_node_idx = None
        self.hovered_node_idx = None
        self.hovered_part = None  # "body" or "aim"

        # Journey Playback Animation
        self.tracer_progress = 0.0
        self.is_paused = False
        self.speed_multiplier = 1.0
        self.dwell_counter = 0
        self.current_step_info = "Trip Ready"

        # Auto-load saved configuration on startup
        self.load_config()

    def load_config(self):
        """Loads persistent camera positions, headings, FOV toggles, and blueprint path."""
        try:
            if os.path.isfile(self.config_path):
                with open(self.config_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                positions = cfg.get("node_positions")
                if positions and isinstance(positions, list) and len(positions) == 4:
                    self.node_positions = [
                        (float(p[0]), float(p[1])) for p in positions
                    ]
                angles = cfg.get("node_angles")
                if angles and isinstance(angles, list) and len(angles) == 4:
                    self.node_angles = [float(a) % 360.0 for a in angles]
                if "show_fov_cones" in cfg:
                    self.show_fov_cones = bool(cfg["show_fov_cones"])
                bp_path = cfg.get("custom_blueprint_path")
                if bp_path and os.path.isfile(bp_path):
                    self.custom_blueprint_path = bp_path
        except Exception:
            self.node_positions = list(DEFAULT_NODE_POSITIONS)
            self.node_angles = list(DEFAULT_NODE_ANGLES)
            self.show_fov_cones = True
            self.custom_blueprint_path = None

    def save_config(self):
        """Persists current camera positions, angles, and custom blueprint path to models/map_config.json."""
        try:
            os.makedirs(self.config_dir, exist_ok=True)
            cfg = {
                "node_positions": [
                    [round(p[0], 4), round(p[1], 4)] for p in self.node_positions
                ],
                "node_angles": [round(a, 1) for a in self.node_angles],
                "show_fov_cones": self.show_fov_cones,
                "custom_blueprint_path": self.custom_blueprint_path
            }
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)
        except Exception:
            pass

    def set_custom_blueprint(self, image_path):
        """Sets and validates a new custom floorplan image."""
        if not image_path or not os.path.isfile(image_path):
            return False
        try:
            with Image.open(image_path) as test_img:
                test_img.verify()
            self.custom_blueprint_path = image_path
            self._cached_blueprint_tk = None
            self._cached_blueprint_dim = None
            self.save_config()
            return True
        except Exception:
            return False

    def clear_custom_blueprint(self):
        """Clears custom blueprint and restores procedural radar grid."""
        self.custom_blueprint_path = None
        self._cached_blueprint_tk = None
        self._cached_blueprint_dim = None
        self.save_config()

    def reset_to_default(self):
        """Resets all camera pin coordinates and angles to standard facility defaults."""
        self.node_positions = list(DEFAULT_NODE_POSITIONS)
        self.node_angles = list(DEFAULT_NODE_ANGLES)
        self.save_config()

    def set_node_position(self, cam_idx, u, v):
        """Updates normalized position (0.04 - 0.96) for a specific camera."""
        if 0 <= cam_idx < 4:
            clamped_u = max(0.04, min(0.96, float(u)))
            clamped_v = max(0.06, min(0.94, float(v)))
            self.node_positions[cam_idx] = (clamped_u, clamped_v)
            self.save_config()

    def set_node_angle(self, cam_idx, deg):
        """Updates heading angle (0° - 360°) for a specific camera."""
        if 0 <= cam_idx < 4:
            self.node_angles[cam_idx] = float(deg) % 360.0
            self.save_config()

    def toggle_fov_cones(self):
        """Toggles visibility of the CCTV FOV coverage vision cones."""
        self.show_fov_cones = not self.show_fov_cones
        self.save_config()
        return self.show_fov_cones

    def toggle_pause(self):
        """Toggles pause/play for the trip tracer animation."""
        self.is_paused = not self.is_paused
        return self.is_paused

    def step_forward(self, total_hops):
        """Manually advances tracer to the next hop."""
        if total_hops > 0:
            cur = int(self.tracer_progress)
            self.tracer_progress = float((cur + 1) % total_hops)
            self.dwell_counter = 0

    def step_backward(self, total_hops):
        """Manually rewinds tracer to the previous hop."""
        if total_hops > 0:
            cur = int(self.tracer_progress)
            self.tracer_progress = float((cur - 1 + total_hops) % total_hops)
            self.dwell_counter = 0

    def set_speed(self, speed_str):
        """Adjusts tracer speed multiplier."""
        if "Slow" in speed_str or "0.5" in speed_str:
            self.speed_multiplier = 0.5
        elif "Fast" in speed_str or "2" in speed_str:
            self.speed_multiplier = 2.0
        else:
            self.speed_multiplier = 1.0

    def _get_cam_idx(self, cam_info, camera_names=None):
        """Extracts 0-indexed camera number (0..3) from dictionary, int, or string."""
        if isinstance(cam_info, dict):
            if "camera_id" in cam_info and cam_info["camera_id"] is not None:
                return min(3, max(0, int(cam_info["camera_id"])))
            cam_info = cam_info.get("camera", "")
        if isinstance(cam_info, int):
            return min(3, max(0, cam_info))
        if not cam_info:
            return 0
        name_str = str(cam_info).strip()
        if camera_names:
            for idx, cn in enumerate(camera_names):
                if cn and (cn.lower() == name_str.lower() or name_str.lower() in cn.lower() or cn.lower() in name_str.lower()):
                    return idx
        name_lower = name_str.lower()
        for i in range(4):
            if f"camera {i+1}" in name_lower or f"cam {i+1}" in name_lower or f"cam{i+1}" in name_lower or f"c{i+1}" in name_lower:
                return i
        return 0

    def _get_curved_control_point(self, p1, p2, offset=30):
        """Calculates Bezier control point perpendicular to vector (p1 -> p2)."""
        x1, y1 = p1
        x2, y2 = p2
        mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        dx, dy = x2 - x1, y2 - y1
        dist = math.hypot(dx, dy)
        if dist < 1e-4:
            return mx, my
        ux, uy = -dy / dist, dx / dist
        return mx + ux * offset, my + uy * offset

    def _eval_bezier(self, p0, pc, p1, t):
        """Evaluates quadratic Bezier curve at parameter t in [0.0, 1.0]."""
        x0, y0 = p0
        xc, yc = pc
        x1, y1 = p1
        t_inv = 1.0 - t
        x = (t_inv ** 2) * x0 + 2.0 * t_inv * t * xc + (t ** 2) * x1
        y = (t_inv ** 2) * y0 + 2.0 * t_inv * t * yc + (t ** 2) * y1
        return x, y

    def _draw_arc_line(self, canvas, cx, cy, radius, start_rad, end_rad, fill, width=1, dash=None, steps=10):
        """Draws a smooth curved arc segment using interpolated line segments."""
        pts = []
        d_rad = (end_rad - start_rad) / max(1, steps)
        for s in range(steps + 1):
            rad = start_rad + s * d_rad
            px = cx + radius * math.cos(rad)
            py = cy + radius * math.sin(rad)
            pts.extend([px, py])
        if len(pts) >= 4:
            canvas.create_line(*pts, fill=fill, width=width, dash=dash, smooth=True)

    def draw_map(self, canvas, width, height, route=None, camera_names=None, last_known=None, pulse_tick=0, is_mini=False):
        """
        Renders the complete 2D Rapido surveillance map, custom blueprint,
        rotatable CCTV FOV vision cones, DORI zones, and multi-camera overlap highlights.
        """
        canvas.delete("all")
        if width <= 10 or height <= 10:
            return

        route = route or []
        camera_names = camera_names or [f"Camera {i+1}" for i in range(4)]
        last_known = last_known or {}

        # ── 1. Background: Custom Blueprint or Radar Blueprint Grid ──────────
        has_custom_blueprint = (
            self.custom_blueprint_path is not None and
            os.path.isfile(self.custom_blueprint_path)
        )

        if has_custom_blueprint:
            try:
                if self._cached_blueprint_dim != (width, height):
                    raw_img = Image.open(self.custom_blueprint_path).convert("RGBA")
                    resized = raw_img.resize((width, height), Image.Resampling.LANCZOS)
                    overlay = Image.new("RGBA", (width, height), (10, 12, 22, 110))
                    blended = Image.alpha_composite(resized, overlay)
                    self._cached_blueprint_tk = ImageTk.PhotoImage(blended)
                    self._cached_blueprint_dim = (width, height)

                canvas.create_image(width // 2, height // 2, image=self._cached_blueprint_tk)
                canvas._bg_img = self._cached_blueprint_tk
            except Exception:
                has_custom_blueprint = False

        if not has_custom_blueprint:
            bg_color = "#0a0b16" if not is_mini else "#0e0f1e"
            canvas.configure(bg=bg_color)

            grid_spacing = 30 if not is_mini else 20
            grid_color = "#141528"
            for gx in range(0, width, grid_spacing):
                canvas.create_line(gx, 0, gx, height, fill=grid_color, width=1)
            for gy in range(0, height, grid_spacing):
                canvas.create_line(0, gy, width, gy, fill=grid_color, width=1)

        # ── 2. Compute Camera Node Positions ─────────────────────────────────
        nodes_xy = []
        for (nu, nv) in self.node_positions:
            nx = int(nu * width)
            ny = int(nv * height)
            nodes_xy.append((nx, ny))

        # Architectural Zone Room Cards (Only when no custom blueprint in full map mode)
        if not is_mini and not has_custom_blueprint:
            zones = [
                ("ZONE A · NORTH ENTRANCE", nodes_xy[0][0] - 80, nodes_xy[0][1] - 50, nodes_xy[0][0] + 80, nodes_xy[0][1] + 55),
                ("ZONE B · EAST CORRIDOR",  nodes_xy[1][0] - 80, nodes_xy[1][1] - 50, nodes_xy[1][0] + 80, nodes_xy[1][1] + 55),
                ("ZONE C · ROOM 204 / WING",nodes_xy[2][0] - 80, nodes_xy[2][1] - 50, nodes_xy[2][0] + 80, nodes_xy[2][1] + 55),
                ("ZONE D · CAFETERIA PLAZA", nodes_xy[3][0] - 80, nodes_xy[3][1] - 50, nodes_xy[3][0] + 80, nodes_xy[3][1] + 55)
            ]
            for zname, zx1, zy1, zx2, zy2 in zones:
                canvas.create_rectangle(zx1, zy1, zx2, zy2, fill="#121326", outline="#1d2040", width=1)
                canvas.create_text(zx1 + 6, zy1 + 8, text=zname, fill="#40446e", font=("Arial", 7, "bold"), anchor="w")

        # Architectural Corridors (facility background pathways)
        corridor_color = "#1b1d36" if not is_mini else "#191a30"
        if has_custom_blueprint:
            corridor_color = "#202540"
        corridor_w = 12 if not is_mini else 6
        pathway_pairs = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)]
        for p1, p2 in pathway_pairs:
            canvas.create_line(nodes_xy[p1][0], nodes_xy[p1][1], nodes_xy[p2][0], nodes_xy[p2][1],
                               fill=corridor_color, width=corridor_w, dash=(6, 4))

        # ── 3. CCTV Field of View (FOV) Vision Cones & DORI Zones ────────────
        active_cam_ids = last_known.get("active_camera_ids", [])
        curr_cam_id = last_known.get("camera_id")
        if curr_cam_id is not None and curr_cam_id not in active_cam_ids:
            active_cam_ids.append(curr_cam_id)
        is_overlap_mode = last_known.get("is_overlap", False)

        if self.show_fov_cones:
            r_total = 115 if not is_mini else 46
            r_id = int(0.35 * r_total)
            r_recog = int(0.65 * r_total)
            r_detect = r_total
            half_rad = math.radians(self.fov_span_deg / 2.0)

            for i, (nx, ny) in enumerate(nodes_xy):
                deg = self.node_angles[i]
                center_rad = math.radians(deg)
                start_rad = center_rad - half_rad
                end_rad = center_rad + half_rad

                is_cam_active = (i in active_cam_ids)
                is_cam_dragged = (i == self.dragged_node_idx)
                is_cam_rotating = (i == self.rotating_node_idx)
                is_cam_hovered = (i == self.hovered_node_idx)

                # Cone line styling
                if is_cam_active:
                    cone_col = "#e040fb" if is_overlap_mode else "#00e676"
                    cone_w = 2
                else:
                    cone_col = "#253452"
                    cone_w = 1

                # 3A. Boundary Sight Rays (Left, Right, and Center Aim Line)
                lx = nx + int(r_detect * math.cos(start_rad))
                ly = ny + int(r_detect * math.sin(start_rad))
                rx = nx + int(r_detect * math.cos(end_rad))
                ry = ny + int(r_detect * math.sin(end_rad))
                mx = nx + int(r_detect * math.cos(center_rad))
                my = ny + int(r_detect * math.sin(center_rad))

                canvas.create_line(nx, ny, lx, ly, fill=cone_col, width=cone_w)
                canvas.create_line(nx, ny, rx, ry, fill=cone_col, width=cone_w)
                canvas.create_line(nx, ny, mx, my, fill=cone_col, width=1, dash=(3, 3))

                # 3B. DORI Concentric Distance Rings
                # Inner Arc: Identification (Cyan)
                self._draw_arc_line(canvas, nx, ny, r_id, start_rad, end_rad,
                                    fill="#00f5ff" if is_cam_active else "#164555",
                                    width=1, dash=(2, 2))

                # Mid Arc: Recognition (Green)
                self._draw_arc_line(canvas, nx, ny, r_recog, start_rad, end_rad,
                                    fill="#00e676" if is_cam_active else "#144830",
                                    width=1, dash=(3, 3))

                # Outer Arc: Detection (Amber)
                self._draw_arc_line(canvas, nx, ny, r_detect, start_rad, end_rad,
                                    fill="#ffb300" if is_cam_active else "#4a3c18",
                                    width=1)

                # 3C. DORI Zone Labels (Full Map Mode)
                if not is_mini:
                    id_lbl_x = nx + int((r_id * 0.75) * math.cos(center_rad))
                    id_lbl_y = ny + int((r_id * 0.75) * math.sin(center_rad))
                    canvas.create_text(id_lbl_x, id_lbl_y, text="ID", fill="#00f5ff", font=("Arial", 6, "bold"))

                    rec_lbl_x = nx + int(((r_id + r_recog) / 2.0) * math.cos(center_rad))
                    rec_lbl_y = ny + int(((r_id + r_recog) / 2.0) * math.sin(center_rad))
                    canvas.create_text(rec_lbl_x, rec_lbl_y, text="RECOG", fill="#00e676", font=("Arial", 6))

                    det_lbl_x = nx + int(((r_recog + r_detect) / 2.0) * math.cos(center_rad))
                    det_lbl_y = ny + int(((r_recog + r_detect) / 2.0) * math.sin(center_rad))
                    canvas.create_text(det_lbl_x, det_lbl_y, text="DETECT", fill="#ffb300", font=("Arial", 6))

                # 3D. Active Radar Pulse Sweep through the Vision Cone
                if is_cam_active:
                    sweep_r = int(((pulse_tick * 4) % r_detect) + 16)
                    self._draw_arc_line(canvas, nx, ny, sweep_r, start_rad, end_rad,
                                        fill="#e040fb" if is_overlap_mode else "#00e676",
                                        width=2)

                # 3E. Aiming Handle / Heading Indicator Chevron
                handle_r = 30 if not is_mini else 16
                hx = nx + int(handle_r * math.cos(center_rad))
                hy = ny + int(handle_r * math.sin(center_rad))

                # Aiming handle tip
                h_col = "#00f5ff" if (is_cam_rotating or (is_cam_hovered and self.hovered_part == "aim")) else cone_col
                canvas.create_oval(hx - 3, hy - 3, hx + 3, hy + 3, fill=h_col, outline="#ffffff", width=1)

        # 3F. Multi-Camera Overlap Connection Highlight
        if is_overlap_mode and len(active_cam_ids) >= 2:
            p_first = nodes_xy[active_cam_ids[0]]
            p_second = nodes_xy[active_cam_ids[1]]
            canvas.create_line(p_first[0], p_first[1], p_second[0], p_second[1],
                               fill="#e040fb", width=3, dash=(6, 4))
            mid_ox = (p_first[0] + p_second[0]) // 2
            mid_oy = (p_first[1] + p_second[1]) // 2
            if not is_mini:
                canvas.create_rectangle(mid_ox - 72, mid_oy - 9, mid_ox + 72, mid_oy + 9,
                                        fill="#2d0a36", outline="#e040fb", width=1)
                canvas.create_text(mid_ox, mid_oy, text="🤝 DUAL OVERLAP ZONE",
                                   fill="#f5b8ff", font=("Arial", 7, "bold"))

        # ── 4. Bidirectional Rapido Travel Routes (A -> B and B -> A) ────────
        visited_cam_indices = []
        hop_splines = []

        if route and len(route) >= 1:
            route_node_indices = [self._get_cam_idx(r, camera_names) for r in route]
            visited_cam_indices = list(set(route_node_indices))
            pair_seen_counts = {}

            if len(route) >= 2:
                for step_idx in range(len(route) - 1):
                    c1_idx = route_node_indices[step_idx]
                    c2_idx = route_node_indices[step_idx + 1]

                    if c1_idx == c2_idx:
                        continue

                    p1 = nodes_xy[c1_idx]
                    p2 = nodes_xy[c2_idx]

                    pair_key = (min(c1_idx, c2_idx), max(c1_idx, c2_idx))
                    count = pair_seen_counts.get(pair_key, 0)
                    pair_seen_counts[pair_key] = count + 1

                    is_forward = (c1_idx <= c2_idx)
                    base_offset = 24 if not is_mini else 14
                    curve_offset = (base_offset * (count + 1)) if is_forward else (-base_offset * (count + 1))

                    cx, cy = self._get_curved_control_point(p1, p2, offset=curve_offset)
                    hop_num = len(hop_splines) + 1
                    hop_splines.append((p1, (cx, cy), p2, route[step_idx + 1], hop_num))

                    # Outer glowing translucent highway lane
                    glow_color = "#00384d" if is_forward else "#4d2800"
                    glow_w = 10 if not is_mini else 6
                    canvas.create_line(p1[0], p1[1], cx, cy, p2[0], p2[1],
                                       fill=glow_color, width=glow_w, smooth=True, capstyle=tk.ROUND)

                    # Core vibrant neon lane
                    core_color = "#00f5ff" if is_forward else "#ffb300"
                    core_w = 3 if not is_mini else 2
                    canvas.create_line(p1[0], p1[1], cx, cy, p2[0], p2[1],
                                       fill=core_color, width=core_w, smooth=True, capstyle=tk.ROUND)

                    # Directional Motion Chevrons along curve
                    for t_arrow in [0.35, 0.65]:
                        ax, ay = self._eval_bezier(p1, (cx, cy), p2, t_arrow)
                        tx, ty = self._eval_bezier(p1, (cx, cy), p2, min(1.0, t_arrow + 0.05))
                        adx = tx - ax
                        ady = ty - ay
                        adist = math.hypot(adx, ady)
                        if adist > 1e-4:
                            aux, auy = adx / adist, ady / adist
                            ch_sz = 6 if not is_mini else 4
                            perpx, perpy = -auy * ch_sz, aux * ch_sz
                            canvas.create_line(ax - aux * ch_sz + perpx, ay - auy * ch_sz + perpy, ax, ay,
                                               fill="#ffffff", width=2)
                            canvas.create_line(ax - aux * ch_sz - perpx, ay - auy * ch_sz - perpy, ax, ay,
                                               fill="#ffffff", width=2)

                    # Step sequence number badge at arc midpoint
                    if not is_mini:
                        mx, my = self._eval_bezier(p1, (cx, cy), p2, 0.50)
                        vtime = route[step_idx + 1].get("video_time", "")
                        badge_txt = f"Hop {hop_num}: {vtime}"
                        bw = len(badge_txt) * 6 + 10
                        canvas.create_rectangle(mx - bw // 2, my - 8, mx + bw // 2, my + 8,
                                                fill="#080914", outline=core_color, width=1)
                        canvas.create_text(mx, my, text=badge_txt, fill=core_color, font=("Arial", 7, "bold"))

        # ── 5. Sequential Blinking Tracer Loop ───────────────────────────────
        if hop_splines:
            total_hops = len(hop_splines)
            cur_hop_idx = min(total_hops - 1, max(0, int(self.tracer_progress)))
            t_frac = self.tracer_progress - int(self.tracer_progress)

            p1, pc, p2, hop_data, step_num = hop_splines[cur_hop_idx]
            dest_cam_name = hop_data.get("camera", f"Camera {step_num}")
            dest_time = hop_data.get("video_time", "")
            is_final_hop = (cur_hop_idx == total_hops - 1)

            if not self.is_paused:
                base_step = (0.015 if not is_mini else 0.025) * self.speed_multiplier
                if t_frac >= 0.95:
                    max_dwell = 35 if is_final_hop else 18
                    if self.dwell_counter < max_dwell:
                        self.dwell_counter += 1
                        t_frac = 1.0
                    else:
                        self.dwell_counter = 0
                        self.tracer_progress = float((cur_hop_idx + 1) % total_hops)
                        t_frac = 0.0
                else:
                    self.tracer_progress += base_step
                    t_frac = min(1.0, self.tracer_progress - int(self.tracer_progress))

            tracer_x, tracer_y = self._eval_bezier(p1, pc, p2, t_frac)

            if t_frac >= 0.92:
                if is_final_hop:
                    self.current_step_info = f"🏁 ARRIVED AT DESTINATION: {dest_cam_name} @ {dest_time}"
                else:
                    self.current_step_info = f"Station Stop: {dest_cam_name} @ {dest_time}"
            else:
                self.current_step_info = f"Transit: Hop {step_num} of {total_hops} ➔ {dest_cam_name}"

            # Glowing Tracer Comet
            t_radius = 8 if not is_mini else 5
            canvas.create_oval(tracer_x - t_radius - 5, tracer_y - t_radius - 5,
                               tracer_x + t_radius + 5, tracer_y + t_radius + 5,
                               fill="", outline="#ffff00", width=2)
            canvas.create_oval(tracer_x - t_radius, tracer_y - t_radius,
                               tracer_x + t_radius, tracer_y + t_radius,
                               fill="#ffea00", outline="#ffffff", width=1)

            # Station Arrival Radar Ripple
            if t_frac > 0.85:
                pulse_r = (pulse_tick % 10) * 3 + 16
                pulse_col = "#00e676" if is_final_hop else "#00f5ff"
                canvas.create_oval(p2[0] - pulse_r, p2[1] - pulse_r,
                                   p2[0] + pulse_r, p2[1] + pulse_r,
                                   outline=pulse_col, width=2)

            if is_final_hop and t_frac >= 0.88 and not is_mini:
                fin_y = height - 55
                canvas.create_rectangle(width // 2 - 210, fin_y - 14, width // 2 + 210, fin_y + 14,
                                        fill="#052e16", outline="#00e676", width=2)
                canvas.create_text(width // 2, fin_y,
                                   text=f"🏁 TRIP COMPLETE · ARRIVED: {dest_cam_name} @ {dest_time}",
                                   fill="#b9f6ca", font=("Arial", 9, "bold"))

        # ── 6. Camera Nodes (Interactive Pins & Rotation Handles) ────────────
        node_radius = 18 if not is_mini else 10
        for i, (nx, ny) in enumerate(nodes_xy):
            is_visited = i in visited_cam_indices
            is_current = (last_known.get("camera") and self._get_cam_idx(last_known.get("camera")) == i)
            is_dragged = (i == self.dragged_node_idx)
            is_rotating = (i == self.rotating_node_idx)
            is_hovered = (i == self.hovered_node_idx)

            # Outer Interaction / Rotation Ring
            if not is_mini:
                rot_ring_r = 30
                if is_rotating:
                    canvas.create_oval(nx - rot_ring_r, ny - rot_ring_r,
                                       nx + rot_ring_r, ny + rot_ring_r,
                                       outline="#00f5ff", width=2, dash=(2, 2))
                elif is_hovered and self.hovered_part == "aim":
                    canvas.create_oval(nx - rot_ring_r, ny - rot_ring_r,
                                       nx + rot_ring_r, ny + rot_ring_r,
                                       outline="#00f5ff", width=1, dash=(3, 3))

            # Draggable Reposition Halo
            if is_dragged:
                canvas.create_oval(nx - node_radius - 8, ny - node_radius - 8,
                                   nx + node_radius + 8, ny + node_radius + 8,
                                   outline="#ffd700", width=3, dash=(4, 2))
            elif is_hovered and self.hovered_part == "body":
                canvas.create_oval(nx - node_radius - 6, ny - node_radius - 6,
                                   nx + node_radius + 6, ny + node_radius + 6,
                                   outline="#00f5ff", width=2)

            if is_current:
                circle_fill = "#ff1744"
                circle_outline = "#ff80ab"
                outline_w = 2
            elif is_visited:
                circle_fill = "#00c853"
                circle_outline = "#b9f6ca"
                outline_w = 2
            else:
                circle_fill = "#1b1c34"
                circle_outline = "#2e3158"
                outline_w = 1

            canvas.create_oval(nx - node_radius, ny - node_radius, nx + node_radius, ny + node_radius,
                               fill=circle_fill, outline=circle_outline, width=outline_w)

            cam_lbl = f"C{i+1}" if is_mini else f"CAM {i+1}"
            txt_color = "#ffffff" if (is_visited or is_current) else "#7e82aa"
            txt_font = ("Arial", 8 if not is_mini else 6, "bold")
            canvas.create_text(nx, ny, text=cam_lbl, fill=txt_color, font=txt_font)

            if not is_mini:
                clean_name = camera_names[i] if i < len(camera_names) else f"Camera {i+1}"
                canvas.create_text(nx, ny + node_radius + 12, text=clean_name,
                                   fill="#c0c4de" if is_visited else "#5a5e82",
                                   font=("Arial", 9, "bold" if is_visited else "normal"))

                # Contextual Interactive Hint
                if is_rotating:
                    deg = int(self.node_angles[i])
                    canvas.create_text(nx, ny - node_radius - 10,
                                       text=f"🎯 AIMING {deg}°",
                                       fill="#00f5ff", font=("Arial", 7, "bold"))
                elif is_dragged:
                    canvas.create_text(nx, ny - node_radius - 10,
                                       text="📌 MOVING PIN...",
                                       fill="#ffd700", font=("Arial", 7, "bold"))
                elif is_hovered:
                    hint = "✥ DRAG TO MOVE" if self.hovered_part == "body" else "🔄 DRAG TO AIM"
                    canvas.create_text(nx, ny - node_radius - 10,
                                       text=hint, fill="#00f5ff", font=("Arial", 7, "bold"))

        # ── 7. Live Pulsing GPS Target Pin (Last Known Location) ─────────────
        if last_known and (last_known.get("camera") or last_known.get("camera_id") is not None):
            curr_idx = self._get_cam_idx(last_known, camera_names)
            px, py = nodes_xy[curr_idx]

            pulse_col = "#e040fb" if is_overlap_mode else "#ff1744"
            pulse_r = (pulse_tick % 12) * (3 if not is_mini else 2) + node_radius
            canvas.create_oval(px - pulse_r, py - pulse_r, px + pulse_r, py + pulse_r,
                               outline=pulse_col, width=2 if not is_mini else 1)

            if not is_mini:
                target_name = last_known.get("target_name") or "TARGET"
                vt = last_known.get("video_time") or ""
                time_str = f" @ {vt}" if vt else ""
                cam_str = last_known.get("camera") or f"Camera {curr_idx + 1}"

                if is_overlap_mode:
                    overlap_list = ", ".join(last_known.get("overlap_cameras", []))
                    banner_txt = f"📍 {target_name.upper()} (PRIMARY: {cam_str}{time_str}) · 🤝 OVERLAP: {overlap_list}"
                else:
                    banner_txt = f"📍 {target_name.upper()} (CURRENT: {cam_str}{time_str})"

                bx = px
                by = py - node_radius - 22
                bw = max(100, len(banner_txt) * 7 + 16)
                canvas.create_rectangle(bx - bw // 2, by - 10, bx + bw // 2, by + 10,
                                        fill=pulse_col, outline="#ffffff", width=1)
                canvas.create_text(bx, by, text=banner_txt, fill="#ffffff", font=("Arial", 8, "bold"))

        # ── 8. Origin Start Flag ─────────────────────────────────────────────
        if route and len(route) >= 1 and not is_mini:
            orig_idx = self._get_cam_idx(route[0], camera_names)
            ox, oy = nodes_xy[orig_idx]
            orig_txt = f"🚩 START: {route[0].get('video_time', '00:00:00')}"
            obw = len(orig_txt) * 6 + 12
            canvas.create_rectangle(ox - obw // 2, oy - node_radius - 22, ox + obw // 2, oy - node_radius - 6,
                                    fill="#00c853", outline="#b9f6ca", width=1)
            canvas.create_text(ox, oy - node_radius - 14, text=orig_txt, fill="#000000", font=("Arial", 7, "bold"))

        # ── 9. Top Telemetry HUD Bar (Full Map Mode) ─────────────────────────
        if not is_mini:
            hud_y = 20
            canvas.create_rectangle(14, 8, width - 14, 34, fill="#0f1124", outline="#1f2348", width=1)

            hops = max(0, len(route) - 1) if route else 0
            loc_count = len(route) if route else 0
            last_cam = last_known.get("camera") or "Standby"
            status_txt = "SEQUENTIAL TRIP REPLAY ACTIVE ●" if route else "WAITING FOR TARGET SIGHTINGS"
            status_col = "#00e676" if route else "#70759e"

            bp_tag = " [CUSTOM BLUEPRINT]" if has_custom_blueprint else " [STANDARD BLUEPRINT]"
            fov_tag = " | FOV: 70° DORI ON" if self.show_fov_cones else " | FOV: Hidden"
            hud_text = f"SURVEILLANCE RADAR MAP 2.5{bp_tag}{fov_tag}   |   Hops: {hops} ({loc_count} Stations)   |   Last: {last_cam}   |   {status_txt}"
            canvas.create_text(24, hud_y, text=hud_text, fill=status_col, font=("Arial", 9, "bold"), anchor="w")

            comp_x = width - 40
            comp_y = height - 35
            canvas.create_oval(comp_x - 14, comp_y - 14, comp_x + 14, comp_y + 14, outline="#2a2e58", width=1)
            canvas.create_text(comp_x, comp_y - 7, text="N", fill="#ff5252", font=("Arial", 7, "bold"))
            canvas.create_text(comp_x, comp_y + 7, text="S", fill="#70759e", font=("Arial", 6))
