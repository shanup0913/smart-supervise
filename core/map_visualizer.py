import math
import tkinter as tk

class MapVisualizer:
    """
    MapVisualizer: Generates an interactive 2D Rapido/Uber-style GPS surveillance map.
    
    Key Capabilities:
    1. Bidirectional Path Visualization (A -> B and B -> A):
       - Automatically offsets forward and return journeys using curved Bezier arcs.
       - A -> B and B -> A never overlap; both are distinctly visible with directional arrows!
    2. Sequential Blinking Tracer Loop (Start to End Trip Replay):
       - An animated glowing GPS pulse travels sequentially from Origin -> Transits -> Destination.
       - Each visited camera station flashes with an expanding radar ripple as the tracer reaches it!
    3. Live Pulsing GPS Target Pin at Last Known Location.
    4. Architectural facility floorplan with zone outlines and blueprint walkways.
    """
    def __init__(self):
        # Normalized coordinates (u, v) for the 4 camera stations [0.0 - 1.0]
        self.node_positions = [
            (0.22, 0.28),  # Camera 1 (North Entrance / Gate)
            (0.78, 0.28),  # Camera 2 (East Corridor Wing A)
            (0.78, 0.72),  # Camera 3 (Room 204 / South Wing)
            (0.22, 0.72),  # Camera 4 (Cafeteria / West Plaza)
        ]
        self.tracer_progress = 0.0 # Loops from 0 to len(route) for sequential trip playback

    def _get_cam_idx(self, cam_name):
        """Extracts 0-indexed camera number from any camera name string."""
        if not cam_name:
            return 0
        name_lower = str(cam_name).lower()
        for i in range(4):
            if f"camera {i+1}" in name_lower or f"cam {i+1}" in name_lower or f"cam{i+1}" in name_lower:
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
        # Unit normal vector
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

    def draw_map(self, canvas, width, height, route=None, camera_names=None, last_known=None, pulse_tick=0, is_mini=False):
        """
        Renders the complete 2D Rapido-style surveillance map onto a Tkinter Canvas.
        """
        canvas.delete("all")
        if width <= 10 or height <= 10:
            return

        route = route or []
        camera_names = camera_names or [f"Camera {i+1}" for i in range(4)]
        last_known = last_known or {}

        # ── 1. Blueprint Grid Background ─────────────────────────────────────
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

        # Architectural Zone Room Cards (Full map mode only)
        if not is_mini:
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
        corridor_w = 12 if not is_mini else 6
        pathway_pairs = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)]
        for p1, p2 in pathway_pairs:
            canvas.create_line(nodes_xy[p1][0], nodes_xy[p1][1], nodes_xy[p2][0], nodes_xy[p2][1],
                               fill=corridor_color, width=corridor_w, dash=(6, 4))

        # ── 3. Bidirectional Rapido Travel Routes (A -> B and B -> A) ────────
        visited_cam_indices = []
        hop_splines = [] # List of ((x0, y0), (xc, yc), (x1, y1), step_info)

        if route and len(route) >= 1:
            route_node_indices = [self._get_cam_idx(r["camera"]) for r in route]
            visited_cam_indices = list(set(route_node_indices))

            # Track edges between pairs to offset return trips cleanly
            pair_seen_counts = {}

            if len(route) >= 2:
                for step_idx in range(len(route) - 1):
                    c1_idx = route_node_indices[step_idx]
                    c2_idx = route_node_indices[step_idx + 1]
                    p1 = nodes_xy[c1_idx]
                    p2 = nodes_xy[c2_idx]

                    # Canonical edge key
                    pair_key = (min(c1_idx, c2_idx), max(c1_idx, c2_idx))
                    count = pair_seen_counts.get(pair_key, 0)
                    pair_seen_counts[pair_key] = count + 1

                    # If traversing forward: offset +24px; if returning opposite: offset -24px
                    is_forward = (c1_idx <= c2_idx)
                    base_offset = 24 if not is_mini else 14
                    curve_offset = (base_offset * (count + 1)) if is_forward else (-base_offset * (count + 1))

                    cx, cy = self._get_curved_control_point(p1, p2, offset=curve_offset)
                    hop_splines.append((p1, (cx, cy), p2, route[step_idx + 1], step_idx + 1))

                    # 3A. Outer glowing translucent highway lane
                    glow_color = "#00384d" if is_forward else "#4d2800"
                    glow_w = 10 if not is_mini else 6
                    canvas.create_line(p1[0], p1[1], cx, cy, p2[0], p2[1],
                                       fill=glow_color, width=glow_w, smooth=True, capstyle=tk.ROUND)

                    # 3B. Core vibrant neon lane (Cyan for forward, Amber for return)
                    core_color = "#00f5ff" if is_forward else "#ffb300"
                    core_w = 3 if not is_mini else 2
                    canvas.create_line(p1[0], p1[1], cx, cy, p2[0], p2[1],
                                       fill=core_color, width=core_w, smooth=True, capstyle=tk.ROUND)

                    # 3C. Directional Motion Chevrons (> > >) along curve
                    for t_arrow in [0.35, 0.65]:
                        ax, ay = self._eval_bezier(p1, (cx, cy), p2, t_arrow)
                        # Tangent derivative for arrow angle
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

                    # 3D. Step sequence number badge at arc midpoint
                    if not is_mini:
                        mx, my = self._eval_bezier(p1, (cx, cy), p2, 0.50)
                        hop_num = step_idx + 1
                        vtime = route[step_idx + 1].get("video_time", "")
                        badge_txt = f"Hop {hop_num}: {vtime}"
                        bw = len(badge_txt) * 6 + 10
                        canvas.create_rectangle(mx - bw//2, my - 8, mx + bw//2, my + 8,
                                                fill="#080914", outline=core_color, width=1)
                        canvas.create_text(mx, my, text=badge_txt, fill=core_color, font=("Arial", 7, "bold"))

        # ── 4. Sequential Blinking Tracer Loop (Start to End Trip Playback) ─
        # A glowing comet pulse travels along the entire journey path from Start -> End
        if hop_splines:
            total_hops = len(hop_splines)
            # Advance tracer progress
            speed_per_tick = 0.06 if not is_mini else 0.08
            self.tracer_progress = (self.tracer_progress + speed_per_tick) % total_hops
            cur_hop_idx = int(self.tracer_progress)
            t_frac = self.tracer_progress - cur_hop_idx

            p1, pc, p2, _, _ = hop_splines[cur_hop_idx]
            tracer_x, tracer_y = self._eval_bezier(p1, pc, p2, t_frac)

            # Glowing Tracer Comet
            t_radius = 7 if not is_mini else 5
            canvas.create_oval(tracer_x - t_radius - 4, tracer_y - t_radius - 4,
                               tracer_x + t_radius + 4, tracer_y + t_radius + 4,
                               fill="", outline="#ffff00", width=2)
            canvas.create_oval(tracer_x - t_radius, tracer_y - t_radius,
                               tracer_x + t_radius, tracer_y + t_radius,
                               fill="#ffea00", outline="#ffffff", width=1)

            # Sequential Node Flash: when tracer reaches near node endpoint (t > 0.85), pulse that node!
            if t_frac > 0.85:
                pulse_hop_dest = p2
                flash_r = (pulse_tick % 8) * 3 + 18
                canvas.create_oval(pulse_hop_dest[0] - flash_r, pulse_hop_dest[1] - flash_r,
                                   pulse_hop_dest[0] + flash_r, pulse_hop_dest[1] + flash_r,
                                   outline="#00ffcc", width=2)

        # ── 5. Camera Nodes ──────────────────────────────────────────────────
        node_radius = 18 if not is_mini else 10
        for i, (nx, ny) in enumerate(nodes_xy):
            is_visited = i in visited_cam_indices
            is_current = (last_known.get("camera") and self._get_cam_idx(last_known.get("camera")) == i)

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

        # ── 6. Live Pulsing GPS Target Pin (Last Known Location) ─────────────
        if last_known and last_known.get("camera"):
            curr_idx = self._get_cam_idx(last_known["camera"])
            px, py = nodes_xy[curr_idx]

            pulse_r = (pulse_tick % 12) * (3 if not is_mini else 2) + node_radius
            canvas.create_oval(px - pulse_r, py - pulse_r, px + pulse_r, py + pulse_r,
                               outline="#ff1744", width=2 if not is_mini else 1)

            if not is_mini:
                target_name = last_known.get("target_name") or "TARGET"
                vt = last_known.get("video_time") or ""
                time_str = f" @ {vt}" if vt else ""
                banner_txt = f"📍 {target_name.upper()} (CURRENT: {last_known['camera']}{time_str})"

                bx = px
                by = py - node_radius - 22
                bw = max(100, len(banner_txt) * 7 + 16)
                canvas.create_rectangle(bx - bw//2, by - 10, bx + bw//2, by + 10,
                                        fill="#ff1744", outline="#ffffff", width=1)
                canvas.create_text(bx, by, text=banner_txt, fill="#ffffff", font=("Arial", 8, "bold"))

        # ── 7. Origin Start Flag ─────────────────────────────────────────────
        if route and len(route) >= 1 and not is_mini:
            orig_idx = self._get_cam_idx(route[0]["camera"])
            ox, oy = nodes_xy[orig_idx]
            orig_txt = f"🚩 START: {route[0].get('video_time', '00:00:00')}"
            obw = len(orig_txt) * 6 + 12
            canvas.create_rectangle(ox - obw//2, oy - node_radius - 22, ox + obw//2, oy - node_radius - 6,
                                    fill="#00c853", outline="#b9f6ca", width=1)
            canvas.create_text(ox, oy - node_radius - 14, text=orig_txt, fill="#000000", font=("Arial", 7, "bold"))

        # ── 8. Top Telemetry HUD Bar (Full Map Mode) ─────────────────────────
        if not is_mini:
            hud_y = 20
            canvas.create_rectangle(14, 8, width - 14, 34, fill="#0f1124", outline="#1f2348", width=1)

            hops = max(0, len(route) - 1) if route else 0
            loc_count = len(route) if route else 0
            last_cam = last_known.get("camera") or "Standby"
            status_txt = "SEQUENTIAL TRIP REPLAY ACTIVE ●" if route else "WAITING FOR TARGET SIGHTINGS"
            status_col = "#00e676" if route else "#70759e"

            hud_text = f"RAPIDO LIVE SURVEILLANCE RADAR   |   Hops: {hops} ({loc_count} Stations)   |   Last Seen: {last_cam}   |   {status_txt}"
            canvas.create_text(24, hud_y, text=hud_text, fill=status_col, font=("Arial", 9, "bold"), anchor="w")

            comp_x = width - 40
            comp_y = height - 35
            canvas.create_oval(comp_x - 14, comp_y - 14, comp_x + 14, comp_y + 14, outline="#2a2e58", width=1)
            canvas.create_text(comp_x, comp_y - 7, text="N", fill="#ff5252", font=("Arial", 7, "bold"))
            canvas.create_text(comp_x, comp_y + 7, text="S", fill="#70759e", font=("Arial", 6))
