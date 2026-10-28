import math
import tkinter as tk

class MapVisualizer:
    """
    MapVisualizer: Generates an interactive 2D Rapido/Uber-style GPS surveillance map.
    
    Visualizes:
    - Architectural floorplan blueprint with 4 facility zones
    - Physical camera placement nodes
    - Dynamic glowing neon travel route with directional motion chevrons
    - Live pulsing radar GPS target pin at the Last Known Location
    - Origin start flag and transit time badges
    """
    def __init__(self):
        # Normalized (u, v) coordinates [0.0 - 1.0] for the 4 camera stations
        # Designed as a connected campus / facility perimeter loop
        self.node_positions = [
            (0.22, 0.28),  # Camera 1 (North Gate / Entry)
            (0.78, 0.28),  # Camera 2 (East Corridor Wing A)
            (0.78, 0.72),  # Camera 3 (Room 204 / South Wing)
            (0.22, 0.72),  # Camera 4 (Cafeteria / West Plaza)
        ]

    def _get_cam_idx(self, cam_name):
        """Extracts 0-indexed camera number from any camera name string."""
        if not cam_name:
            return 0
        name_lower = str(cam_name).lower()
        for i in range(4):
            if f"camera {i+1}" in name_lower or f"cam {i+1}" in name_lower or f"cam{i+1}" in name_lower:
                return i
        return 0

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

        # ── 2. Facility Zone Outlines & Walkways ─────────────────────────────
        # Compute absolute (x, y) coordinates for all 4 camera nodes
        nodes_xy = []
        for (nu, nv) in self.node_positions:
            nx = int(nu * width)
            ny = int(nv * height)
            nodes_xy.append((nx, ny))

        if not is_mini:
            # Draw architectural zone room cards
            zones = [
                ("ZONE A · NORTH ENTRANCE", nodes_xy[0][0] - 80, nodes_xy[0][1] - 50, nodes_xy[0][0] + 80, nodes_xy[0][1] + 55),
                ("ZONE B · EAST CORRIDOR",  nodes_xy[1][0] - 80, nodes_xy[1][1] - 50, nodes_xy[1][0] + 80, nodes_xy[1][1] + 55),
                ("ZONE C · ROOM 204 / WING",nodes_xy[2][0] - 80, nodes_xy[2][1] - 50, nodes_xy[2][0] + 80, nodes_xy[2][1] + 55),
                ("ZONE D · CAFETERIA PLAZA", nodes_xy[3][0] - 80, nodes_xy[3][1] - 50, nodes_xy[3][0] + 80, nodes_xy[3][1] + 55)
            ]
            for zname, zx1, zy1, zx2, zy2 in zones:
                canvas.create_rectangle(zx1, zy1, zx2, zy2, fill="#121326", outline="#1d2040", width=1)
                canvas.create_text(zx1 + 6, zy1 + 8, text=zname, fill="#40446e", font=("Arial", 7, "bold"), anchor="w")

        # Architectural Corridors (connecting pathways)
        corridor_color = "#1b1d36" if not is_mini else "#191a30"
        corridor_w = 12 if not is_mini else 6
        # Perimeter pathways: 0->1->2->3->0
        pathway_pairs = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)]
        for p1, p2 in pathway_pairs:
            canvas.create_line(nodes_xy[p1][0], nodes_xy[p1][1], nodes_xy[p2][0], nodes_xy[p2][1],
                               fill=corridor_color, width=corridor_w, dash=(6, 4))

        # ── 3. Live Rapido / Uber Travel Route Path ─────────────────────────
        # Trace chronological movement from origin through transits to destination
        visited_cam_indices = []
        if route and len(route) >= 1:
            route_node_indices = [self._get_cam_idx(r["camera"]) for r in route]
            visited_cam_indices = list(set(route_node_indices))

            if len(route) >= 2:
                for step_idx in range(len(route) - 1):
                    c1_idx = route_node_indices[step_idx]
                    c2_idx = route_node_indices[step_idx + 1]
                    x1, y1 = nodes_xy[c1_idx]
                    x2, y2 = nodes_xy[c2_idx]

                    # Layer A: Glowing wide cyan shadow
                    glow_w = 10 if not is_mini else 6
                    canvas.create_line(x1, y1, x2, y2, fill="#00384d", width=glow_w, capstyle=tk.ROUND)

                    # Layer B: Vibrant neon cyan transit line
                    core_w = 3 if not is_mini else 2
                    canvas.create_line(x1, y1, x2, y2, fill="#00e5ff", width=core_w, capstyle=tk.ROUND)

                    # Layer C: Directional Motion Chevrons (> > >) along vector
                    dx = x2 - x1
                    dy = y2 - y1
                    dist = math.hypot(dx, dy)
                    if dist > 30:
                        ux = dx / dist
                        uy = dy / dist
                        num_chevrons = max(2, int(dist / (45 if not is_mini else 30)))
                        for ch_i in range(1, num_chevrons):
                            cx = x1 + ux * (dist * ch_i / num_chevrons)
                            cy = y1 + uy * (dist * ch_i / num_chevrons)
                            # Draw clean arrowhead pointing in travel direction
                            ch_size = 5 if not is_mini else 3
                            px = -uy * ch_size
                            py = ux * ch_size
                            canvas.create_line(cx - ux * ch_size + px, cy - uy * ch_size + py, cx, cy, fill="#ffdd00", width=2)
                            canvas.create_line(cx - ux * ch_size - px, cy - uy * ch_size - py, cx, cy, fill="#ffdd00", width=2)

                    # Layer D: Transit Time badge near line midpoint
                    if not is_mini and "video_time" in route[step_idx + 1]:
                        mx = int((x1 + x2) / 2)
                        my = int((y1 + y2) / 2) - 10
                        hop_time = route[step_idx + 1].get("video_time", "")
                        badge_txt = f"⏱ {hop_time}"
                        canvas.create_rectangle(mx - 28, my - 8, mx + 28, my + 8, fill="#090a18", outline="#00e5ff", width=1)
                        canvas.create_text(mx, my, text=badge_txt, fill="#00e5ff", font=("Arial", 7, "bold"))

        # ── 4. Camera Nodes ──────────────────────────────────────────────────
        node_radius = 18 if not is_mini else 10
        for i, (nx, ny) in enumerate(nodes_xy):
            is_visited = i in visited_cam_indices
            is_current = (last_known.get("camera") and self._get_cam_idx(last_known.get("camera")) == i)

            # Node Base Circle
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

            # Camera ID / Icon inside circle
            cam_lbl = f"C{i+1}" if is_mini else f"CAM {i+1}"
            txt_color = "#ffffff" if (is_visited or is_current) else "#7e82aa"
            txt_font = ("Arial", 8 if not is_mini else 6, "bold")
            canvas.create_text(nx, ny, text=cam_lbl, fill=txt_color, font=txt_font)

            # Camera Location Text Label
            if not is_mini:
                clean_name = camera_names[i] if i < len(camera_names) else f"Camera {i+1}"
                canvas.create_text(nx, ny + node_radius + 12, text=clean_name,
                                   fill="#c0c4de" if is_visited else "#5a5e82",
                                   font=("Arial", 9, "bold" if is_visited else "normal"))

        # ── 5. Live Pulsing GPS Target Pin (Rapido / Uber Pin) ───────────────
        if last_known and last_known.get("camera"):
            curr_idx = self._get_cam_idx(last_known["camera"])
            px, py = nodes_xy[curr_idx]

            # Pulsing Concentric Radar Rings
            pulse_r = (pulse_tick % 12) * (3 if not is_mini else 2) + node_radius
            canvas.create_oval(px - pulse_r, py - pulse_r, px + pulse_r, py + pulse_r,
                               outline="#ff1744", width=2 if not is_mini else 1)

            # Live Pin Banner Card above the current location
            if not is_mini:
                target_name = last_known.get("target_name") or "TARGET"
                vt = last_known.get("video_time") or ""
                time_str = f" @ {vt}" if vt else ""
                banner_txt = f"📍 {target_name.upper()} (LIVE: {last_known['camera']}{time_str})"

                bx = px
                by = py - node_radius - 22
                bw = max(100, len(banner_txt) * 7 + 16)
                canvas.create_rectangle(bx - bw//2, by - 10, bx + bw//2, by + 10,
                                        fill="#ff1744", outline="#ffffff", width=1)
                canvas.create_text(bx, by, text=banner_txt, fill="#ffffff", font=("Arial", 8, "bold"))

        # ── 6. Origin Start Flag ─────────────────────────────────────────────
        if route and len(route) >= 1 and not is_mini:
            orig_idx = self._get_cam_idx(route[0]["camera"])
            ox, oy = nodes_xy[orig_idx]
            orig_txt = f"🚩 ORIGIN ({route[0].get('video_time', '00:00:00')})"
            obw = len(orig_txt) * 6 + 12
            canvas.create_rectangle(ox - obw//2, oy - node_radius - 22, ox + obw//2, oy - node_radius - 6,
                                    fill="#00c853", outline="#b9f6ca", width=1)
            canvas.create_text(ox, oy - node_radius - 14, text=orig_txt, fill="#000000", font=("Arial", 7, "bold"))

        # ── 7. Top Telemetry HUD Bar (Full Map Mode Only) ─────────────────────
        if not is_mini:
            hud_y = 20
            canvas.create_rectangle(14, 8, width - 14, 34, fill="#0f1124", outline="#1f2348", width=1)

            hops = max(0, len(route) - 1) if route else 0
            loc_count = len(route) if route else 0
            last_cam = last_known.get("camera") or "Standby"
            status_txt = "ACTIVE TRACKING ●" if route else "WAITING FOR TARGET SIGHTINGS"
            status_col = "#00e676" if route else "#70759e"

            hud_text = f"RAPIDO LIVE SURVEILLANCE RADAR   |   Hops: {hops} ({loc_count} Locations)   |   Last Seen: {last_cam}   |   {status_txt}"
            canvas.create_text(24, hud_y, text=hud_text, fill=status_col, font=("Arial", 9, "bold"), anchor="w")

            # Compass in top right
            comp_x = width - 40
            comp_y = height - 35
            canvas.create_oval(comp_x - 14, comp_y - 14, comp_x + 14, comp_y + 14, outline="#2a2e58", width=1)
            canvas.create_text(comp_x, comp_y - 7, text="N", fill="#ff5252", font=("Arial", 7, "bold"))
            canvas.create_text(comp_x, comp_y + 7, text="S", fill="#70759e", font=("Arial", 6))
