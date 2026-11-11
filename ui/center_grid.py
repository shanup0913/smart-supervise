"""
CenterGrid: 2x2 CCTV Matrix, focused camera zoom, Spotlight mode,
interactive 2D Rapido GPS Radar Map 2.0 with custom blueprint upload,
draggable camera pins, and zero-lag dirty-checked telemetry.
"""

import os
import math
import cv2
from PIL import Image
import tkinter as tk
from tkinter import filedialog
import customtkinter as ctk

from ui.theme import (
    DARK_BG, PANEL_BG, CARD_BG, ACCENT_BLUE,
    ACCENT_GREEN, ACCENT_RED, ACCENT_AMBER,
    TEXT_LIGHT, TEXT_DIM
)


class CenterGrid(ctk.CTkFrame):
    """Center surveillance viewport supporting 2x2 grid, spotlight, single cam, and full radar map 2.0."""
    def __init__(self, parent, app):
        super().__init__(parent, fg_color=PANEL_BG, corner_radius=12)
        self.app = app

        self.display_mode = ctk.StringVar(value="grid")
        self._drag_node_idx = None

        self._build_ui()

    def _build_ui(self):
        # ── Top Control & Responsive Telemetry Bar ──
        ctrl_box = ctk.CTkFrame(self, fg_color="transparent")
        ctrl_box.pack(fill="x", padx=6, pady=(6, 2))

        # Row 1: View Selection Buttons
        row1 = ctk.CTkFrame(ctrl_box, fg_color="transparent", height=28)
        row1.pack(fill="x", pady=(0, 3))

        ctk.CTkLabel(
            row1, text="DISPLAY VIEW:",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=TEXT_DIM
        ).pack(side="left", padx=(2, 6))

        self.view_buttons = ctk.CTkSegmentedButton(
            row1,
            values=["⊞ 4-Grid", "Cam 1", "Cam 2", "Cam 3", "Cam 4", "Spotlight", "🗺️ Map"],
            command=self._on_display_mode_change,
            selected_color=ACCENT_BLUE,
            height=26,
            font=ctk.CTkFont(size=10)
        )
        self.view_buttons.set("⊞ 4-Grid")
        self.view_buttons.pack(side="left", fill="x", expand=True, padx=(0, 6))

        self.map_quick_btn = ctk.CTkButton(
            row1, text="🗺️ Full Map View",
            command=self._toggle_map_view,
            fg_color="#1a2838", hover_color="#204060",
            text_color=ACCENT_BLUE,
            border_width=1, border_color=ACCENT_BLUE,
            corner_radius=6, width=115, height=26,
            font=ctk.CTkFont(size=10, weight="bold")
        )
        self.map_quick_btn.pack(side="left")

        # Row 2: Dedicated Full-Width Hardware Telemetry Strip (Dirty-checked for zero lag)
        telem_strip = ctk.CTkFrame(ctrl_box, fg_color="#121220", corner_radius=6, height=24)
        telem_strip.pack(fill="x", pady=(1, 2))

        self.fps_label = ctk.CTkLabel(
            telem_strip, text="⚡ Speed Telemetry: Standby",
            font=ctk.CTkFont(family="Courier New", size=10, weight="bold"),
            text_color=ACCENT_AMBER
        )
        self.fps_label.pack(side="left", padx=8, pady=2)

        # ── View A: 2x2 Fixed-Size Grid Container ──
        self.grid_container = ctk.CTkFrame(self, fg_color="transparent")
        self.grid_container.pack(fill="both", expand=True, padx=6, pady=(2, 6))

        self.grid_container.grid_rowconfigure(0, weight=1, uniform="row")
        self.grid_container.grid_rowconfigure(1, weight=1, uniform="row")
        self.grid_container.grid_columnconfigure(0, weight=1, uniform="col")
        self.grid_container.grid_columnconfigure(1, weight=1, uniform="col")

        self.tiles = []
        for i in range(4):
            r = i // 2
            c = i % 2
            tile_card = ctk.CTkFrame(
                self.grid_container, fg_color=DARK_BG,
                corner_radius=8, border_width=1, border_color="#22223a"
            )
            tile_card.grid(row=r, column=c, padx=4, pady=4, sticky="nsew")

            hdr = ctk.CTkLabel(
                tile_card, text=f"CAM {i+1} — Standby",
                font=ctk.CTkFont(size=10, weight="bold"),
                fg_color=CARD_BG, height=22, corner_radius=6,
                text_color=TEXT_LIGHT
            )
            hdr.pack(fill="x", padx=2, pady=(2, 0))

            # Per-camera individual progress footer
            footer = ctk.CTkFrame(tile_card, fg_color="#121220", height=20, corner_radius=4)
            footer.pack(fill="x", side="bottom", padx=2, pady=(0, 2))

            tile_pbar = ctk.CTkProgressBar(
                footer, height=5, corner_radius=3,
                progress_color=ACCENT_GREEN, fg_color="#202036"
            )
            tile_pbar.set(0.0)
            tile_pbar.pack(side="left", fill="x", expand=True, padx=(6, 6), pady=3)

            tile_plabel = ctk.CTkLabel(
                footer, text="0.0%",
                font=ctk.CTkFont(family="Courier New", size=9, weight="bold"),
                text_color=TEXT_DIM
            )
            tile_plabel.pack(side="right", padx=(0, 6))

            vlabel = ctk.CTkLabel(
                tile_card, text=f"Camera {i+1}\n(Standby / Disabled)",
                font=ctk.CTkFont(size=11), text_color=TEXT_DIM
            )
            vlabel.pack(fill="both", expand=True, padx=2, pady=2)

            self.tiles.append({
                "frame": tile_card,
                "header": hdr,
                "label": vlabel,
                "footer": footer,
                "pbar": tile_pbar,
                "plabel": tile_plabel,
                "last_rendered_seq": -1
            })

        # ── View B: Single / Focused Camera Container ──
        self.single_container = ctk.CTkFrame(self, fg_color="transparent")
        self.single_header = ctk.CTkLabel(
            self.single_container, text="Focused Camera",
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color=CARD_BG, height=24, corner_radius=6,
            text_color=TEXT_LIGHT
        )
        self.single_header.pack(fill="x", padx=2, pady=(2, 4))

        self.single_footer = ctk.CTkFrame(self.single_container, fg_color="#121220", height=24, corner_radius=6)
        self.single_footer.pack(fill="x", side="bottom", padx=2, pady=(0, 4))

        self.single_pbar = ctk.CTkProgressBar(
            self.single_footer, height=7, corner_radius=4,
            progress_color=ACCENT_GREEN, fg_color="#202036"
        )
        self.single_pbar.set(0.0)
        self.single_pbar.pack(side="left", fill="x", expand=True, padx=(8, 8), pady=5)

        self.single_plabel = ctk.CTkLabel(
            self.single_footer, text="0.0%",
            font=ctk.CTkFont(family="Courier New", size=10, weight="bold"),
            text_color=TEXT_DIM
        )
        self.single_plabel.pack(side="right", padx=(0, 8))

        self.single_label = ctk.CTkLabel(
            self.single_container, text="No Feed Selected",
            font=ctk.CTkFont(size=14), text_color=TEXT_DIM,
            fg_color=DARK_BG, corner_radius=8
        )
        self.single_label.pack(fill="both", expand=True, padx=2, pady=2)
        self.last_single_rendered_seq = -1
        self.last_single_rendered_idx = -1

        # ── View C: Interactive 2D Rapido Live Map 2.0 Container ──
        self.map_container = ctk.CTkFrame(self, fg_color="transparent")

        # Top Map Header
        map_hdr = ctk.CTkFrame(self.map_container, fg_color=CARD_BG, corner_radius=6, height=28)
        map_hdr.pack(fill="x", padx=2, pady=(2, 2))

        ctk.CTkLabel(
            map_hdr, text="🗺️ SURVEILLANCE RADAR & BLUEPRINT MAP 2.0",
            font=ctk.CTkFont(size=11, weight="bold"), text_color=ACCENT_AMBER
        ).pack(side="left", padx=8)

        ctk.CTkButton(
            map_hdr, text="📹 Return to 4-Grid Video Matrix",
            command=self.switch_to_grid,
            fg_color=ACCENT_GREEN, hover_color="#04a57a",
            text_color="#000000",
            corner_radius=6, height=22,
            font=ctk.CTkFont(size=10, weight="bold")
        ).pack(side="right", padx=6, pady=2)

        # Map 2.0 Calibration Toolbar (Upload Blueprint, Clear, Reset, Drag Hint)
        self.map_tools_bar = ctk.CTkFrame(self.map_container, fg_color="#121326", corner_radius=6, height=30)
        self.map_tools_bar.pack(fill="x", padx=2, pady=(0, 3))

        ctk.CTkButton(
            self.map_tools_bar, text="🖼️ Upload Blueprint",
            command=self._on_upload_blueprint,
            fg_color="#1c2844", hover_color="#283a66",
            text_color=ACCENT_BLUE,
            border_width=1, border_color=ACCENT_BLUE,
            corner_radius=6, height=22, width=130,
            font=ctk.CTkFont(size=10, weight="bold")
        ).pack(side="left", padx=(6, 4), pady=3)

        self.clear_bp_btn = ctk.CTkButton(
            self.map_tools_bar, text="🗑️ Clear Blueprint",
            command=self._on_clear_blueprint,
            fg_color=DARK_BG, hover_color="#3a1c1c",
            text_color="#d05050",
            border_width=1, border_color="#552222",
            corner_radius=6, height=22, width=110,
            font=ctk.CTkFont(size=10)
        )
        self.clear_bp_btn.pack(side="left", padx=2, pady=3)

        ctk.CTkButton(
            self.map_tools_bar, text="🔄 Reset Pin Locations",
            command=self._on_reset_pins,
            fg_color=DARK_BG, hover_color=PANEL_BG,
            text_color=TEXT_LIGHT,
            border_width=1, border_color="#333355",
            corner_radius=6, height=22, width=130,
            font=ctk.CTkFont(size=10)
        ).pack(side="left", padx=2, pady=3)

        self.fov_toggle_btn = ctk.CTkButton(
            self.map_tools_bar, text="👁️ Hide FOV Cones",
            command=self._on_toggle_fov,
            fg_color="#182436", hover_color="#20344d",
            text_color=ACCENT_AMBER,
            border_width=1, border_color=ACCENT_AMBER,
            corner_radius=6, height=22, width=125,
            font=ctk.CTkFont(size=10, weight="bold")
        )
        self.fov_toggle_btn.pack(side="left", padx=2, pady=3)

        ctk.CTkLabel(
            self.map_tools_bar, text="💡 Drag pin center to move  |  Drag outer ring to rotate 360° heading",
            font=ctk.CTkFont(size=9, weight="bold"),
            text_color=ACCENT_AMBER
        ).pack(side="right", padx=8, pady=3)

        # ── Map Journey Playback Controller Bar ──
        self.map_ctrl_bar = ctk.CTkFrame(self.map_container, fg_color=CARD_BG, corner_radius=8, height=36)
        self.map_ctrl_bar.pack(fill="x", padx=2, pady=(0, 4), side="bottom")

        # Play / Pause
        self.map_play_btn = ctk.CTkButton(
            self.map_ctrl_bar, text="⏸ Pause", width=68, height=26,
            fg_color=DARK_BG, hover_color=PANEL_BG, font=ctk.CTkFont(size=10, weight="bold"),
            command=self._toggle_map_playback
        )
        self.map_play_btn.pack(side="left", padx=(8, 4), pady=4)

        # Step Back
        ctk.CTkButton(
            self.map_ctrl_bar, text="⏮ Prev Hop", width=74, height=26,
            fg_color=DARK_BG, hover_color=PANEL_BG, font=ctk.CTkFont(size=10),
            command=self._step_map_backward
        ).pack(side="left", padx=2, pady=4)

        # Step Forward
        ctk.CTkButton(
            self.map_ctrl_bar, text="Next Hop ⏭", width=74, height=26,
            fg_color=DARK_BG, hover_color=PANEL_BG, font=ctk.CTkFont(size=10),
            command=self._step_map_forward
        ).pack(side="left", padx=2, pady=4)

        # Tracer Speed Selector
        ctk.CTkLabel(self.map_ctrl_bar, text="Speed:", font=ctk.CTkFont(size=10, weight="bold"), text_color=TEXT_DIM).pack(side="left", padx=(10, 4))
        self.map_tracer_speed = ctk.CTkSegmentedButton(
            self.map_ctrl_bar, values=["Slow (0.5x)", "Normal (1x)", "Fast (2x)"],
            height=22, font=ctk.CTkFont(size=9), selected_color=ACCENT_BLUE,
            command=self._on_map_speed_change
        )
        self.map_tracer_speed.set("Normal (1x)")
        self.map_tracer_speed.pack(side="left", padx=2, pady=4)

        # Journey Step Telemetry Pill
        self.map_step_label = ctk.CTkLabel(
            self.map_ctrl_bar, text="Trip Status: Standby",
            font=ctk.CTkFont(size=10, weight="bold"), text_color=ACCENT_GREEN
        )
        self.map_step_label.pack(side="right", padx=12, pady=4)

        # Main Map Canvas with Drag-and-Drop Pin Calibration bindings
        self.full_map_canvas = tk.Canvas(self.map_container, bg="#0a0b16", highlightthickness=0)
        self.full_map_canvas.pack(fill="both", expand=True, padx=2, pady=2)

        self.full_map_canvas.bind("<ButtonPress-1>", self._on_map_press)
        self.full_map_canvas.bind("<B1-Motion>", self._on_map_drag)
        self.full_map_canvas.bind("<ButtonRelease-1>", self._on_map_release)
        self.full_map_canvas.bind("<Motion>", self._on_map_motion)
        self.full_map_canvas.bind("<Leave>", self._on_map_leave)

        # ── LAST KNOWN LOCATION Banner ──
        self.last_seen_banner = ctk.CTkFrame(self, fg_color=CARD_BG, corner_radius=8, height=42)
        self.last_seen_banner.pack(fill="x", padx=6, pady=(0, 6))
        self.last_seen_banner.pack_propagate(False)

        self.last_seen_label = ctk.CTkLabel(
            self.last_seen_banner,
            text="Last Known Location: Ready to search",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=TEXT_DIM
        )
        self.last_seen_label.pack(expand=True)

    # ── Map 2.0 User Action Handlers ──
    def _on_upload_blueprint(self):
        """Allows user to select a floorplan image to render as the surveillance map background."""
        path = filedialog.askopenfilename(
            title="Select Facility Floorplan / Blueprint Image",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.webp")]
        )
        if not path:
            return
        success = self.app.map_visualizer.set_custom_blueprint(path)
        if success:
            self.app.set_status(f"Custom Blueprint loaded: {os.path.basename(path)}", color=ACCENT_GREEN)
            self.render_full_map()
        else:
            self.app.set_status("Failed to load blueprint image file!", color=ACCENT_RED)

    def _on_clear_blueprint(self):
        """Restores map background to standard high-tech radar blueprint grid."""
        self.app.map_visualizer.clear_custom_blueprint()
        self.app.set_status("Blueprint restored to procedural radar grid", color=TEXT_DIM)
        self.render_full_map()

    def _on_reset_pins(self):
        """Resets all 4 camera pin positions to default corner layout."""
        self.app.map_visualizer.reset_to_default()
        self.app.set_status("Camera pin positions reset to default", color=ACCENT_GREEN)
        self.render_full_map()

    def _on_toggle_fov(self):
        """Toggles visibility of the CCTV Field-of-View vision cones on the map."""
        is_shown = self.app.map_visualizer.toggle_fov_cones()
        self.fov_toggle_btn.configure(
            text="👁️ Hide FOV Cones" if is_shown else "👁️ Show FOV Cones",
            text_color=ACCENT_AMBER if is_shown else TEXT_DIM
        )
        self.render_full_map()

    def _get_cardinal(self, deg):
        """Converts angle in degrees to 8-point compass cardinal direction."""
        dirs = ["East", "SE", "South", "SW", "West", "NW", "North", "NE"]
        idx = int(((deg + 22.5) % 360.0) / 45.0) % 8
        return dirs[idx]

    def _on_map_press(self, event):
        """
        Interactive Dual-Zone Pin Controller:
        - Inner circle (d <= 18px): Drag to reposition camera on floorplan.
        - Outer ring (18px < d <= 42px): Rotate camera heading 360°.
        """
        fw = self.full_map_canvas.winfo_width() or 640
        fh = self.full_map_canvas.winfo_height() or 460
        self._drag_node_idx = None
        self._rotate_node_idx = None
        for idx, (nu, nv) in enumerate(self.app.map_visualizer.node_positions):
            nx = int(nu * fw)
            ny = int(nv * fh)
            d = math.hypot(event.x - nx, event.y - ny)
            if d <= 18:
                self._drag_node_idx = idx
                self.app.map_visualizer.dragged_node_idx = idx
                self.full_map_canvas.configure(cursor="fleur")
                self.render_full_map()
                break
            elif d <= 42:
                self._rotate_node_idx = idx
                self.app.map_visualizer.rotating_node_idx = idx
                self.full_map_canvas.configure(cursor="crosshair")
                self.render_full_map()
                break

    def _on_map_drag(self, event):
        """Handles real-time pin repositioning or 360° heading rotation."""
        fw = self.full_map_canvas.winfo_width() or 640
        fh = self.full_map_canvas.winfo_height() or 460
        if getattr(self, "_drag_node_idx", None) is not None:
            nu = max(0.04, min(0.96, event.x / fw))
            nv = max(0.06, min(0.94, event.y / fh))
            self.app.map_visualizer.node_positions[self._drag_node_idx] = (nu, nv)
            if hasattr(self, "map_step_label"):
                cam_num = self._drag_node_idx + 1
                self.map_step_label.configure(
                    text=f"📍 Relocating CAM {cam_num} to ({int(nu*100)}%, {int(nv*100)}%)"
                )
            self.render_full_map()
        elif getattr(self, "_rotate_node_idx", None) is not None:
            idx = self._rotate_node_idx
            nu, nv = self.app.map_visualizer.node_positions[idx]
            nx = int(nu * fw)
            ny = int(nv * fh)
            deg = (math.degrees(math.atan2(event.y - ny, event.x - nx))) % 360.0
            self.app.map_visualizer.node_angles[idx] = round(deg, 1)
            if hasattr(self, "map_step_label"):
                cardinal = self._get_cardinal(deg)
                self.map_step_label.configure(
                    text=f"🎯 Aiming CAM {idx+1} at {int(deg)}° ({cardinal})"
                )
            self.render_full_map()

    def _on_map_release(self, event):
        """Finalizes pin movement or rotation and saves to models/map_config.json."""
        saved_something = False
        if getattr(self, "_drag_node_idx", None) is not None:
            idx = self._drag_node_idx
            self._drag_node_idx = None
            self.app.map_visualizer.dragged_node_idx = None
            if hasattr(self, "map_step_label"):
                self.map_step_label.configure(text=f"💾 Saved Camera {idx + 1} position!")
            saved_something = True
        if getattr(self, "_rotate_node_idx", None) is not None:
            idx = self._rotate_node_idx
            self._rotate_node_idx = None
            self.app.map_visualizer.rotating_node_idx = None
            if hasattr(self, "map_step_label"):
                deg = int(self.app.map_visualizer.node_angles[idx])
                cardinal = self._get_cardinal(deg)
                self.map_step_label.configure(text=f"💾 Saved Camera {idx + 1} heading ({deg}° {cardinal})")
            saved_something = True
        if saved_something:
            self.full_map_canvas.configure(cursor="")
            self.app.map_visualizer.save_config()
            self.render_full_map()

    def _on_map_motion(self, event):
        """Updates cursor dynamically based on whether hovering pin body (move) or rotation ring (aim)."""
        if getattr(self, "_drag_node_idx", None) is not None or getattr(self, "_rotate_node_idx", None) is not None:
            return
        fw = self.full_map_canvas.winfo_width() or 640
        fh = self.full_map_canvas.winfo_height() or 460
        hover_idx = None
        hover_part = None
        for idx, (nu, nv) in enumerate(self.app.map_visualizer.node_positions):
            nx = int(nu * fw)
            ny = int(nv * fh)
            d = math.hypot(event.x - nx, event.y - ny)
            if d <= 18:
                hover_idx = idx
                hover_part = "body"
                break
            elif d <= 42:
                hover_idx = idx
                hover_part = "aim"
                break
        if hover_idx != self.app.map_visualizer.hovered_node_idx or hover_part != getattr(self.app.map_visualizer, "hovered_part", None):
            self.app.map_visualizer.hovered_node_idx = hover_idx
            self.app.map_visualizer.hovered_part = hover_part
            cursor_type = "fleur" if hover_part == "body" else ("crosshair" if hover_part == "aim" else "")
            self.full_map_canvas.configure(cursor=cursor_type)
            self.render_full_map()

    def _on_map_leave(self, event):
        """Clears hover states and resets cursor when mouse leaves the canvas."""
        if self.app.map_visualizer.hovered_node_idx is not None or self.app.map_visualizer.hovered_part is not None:
            self.app.map_visualizer.hovered_node_idx = None
            self.app.map_visualizer.hovered_part = None
            self.full_map_canvas.configure(cursor="")
            self.render_full_map()
            self.render_full_map()

    def switch_to_map(self):
        """Switches central viewport to the interactive GPS journey map."""
        self.view_buttons.set("🗺️ Map")
        self._on_display_mode_change("🗺️ Map")

    def switch_to_grid(self):
        """Switches central viewport back to the 4-camera video matrix."""
        self.view_buttons.set("⊞ 4-Grid")
        self._on_display_mode_change("⊞ 4-Grid")

    def _toggle_map_view(self):
        """One-click toggle between 4-Grid video feeds and Live Journey Map."""
        if self.display_mode.get() == "map":
            self.switch_to_grid()
        else:
            self.switch_to_map()

    def _on_display_mode_change(self, mode):
        self.grid_container.pack_forget()
        self.single_container.pack_forget()
        self.map_container.pack_forget()

        if "Map" in mode:
            self.map_container.pack(fill="both", expand=True, padx=6, pady=(2, 6))
            self.display_mode.set("map")
            if hasattr(self, "map_quick_btn"):
                self.map_quick_btn.configure(
                    text="📹 Back to Videos",
                    fg_color=ACCENT_GREEN,
                    hover_color="#04a57a",
                    text_color="#000000"
                )
            self.render_full_map()
        else:
            if hasattr(self, "map_quick_btn"):
                self.map_quick_btn.configure(
                    text="🗺️ Full Map View",
                    fg_color="#1a2838",
                    hover_color="#204060",
                    text_color=ACCENT_BLUE
                )
            if mode == "⊞ 4-Grid":
                self.grid_container.pack(fill="both", expand=True, padx=6, pady=(2, 6))
                self.display_mode.set("grid")
            elif mode == "Spotlight":
                self.single_container.pack(fill="both", expand=True, padx=6, pady=(2, 6))
                self.display_mode.set("spotlight")
            else:
                idx = int(mode.replace("Cam ", "")) - 1
                self.single_container.pack(fill="both", expand=True, padx=6, pady=(2, 6))
                self.display_mode.set(f"cam{idx}")

    def _toggle_map_playback(self):
        is_paused = self.app.map_visualizer.toggle_pause()
        self.map_play_btn.configure(text="▶ Play" if is_paused else "⏸ Pause")

    def _step_map_backward(self):
        route = self.app.tracker.get_journey_route()
        total_hops = max(1, len(route) - 1)
        self.app.map_visualizer.step_backward(total_hops)
        self.render_full_map()

    def _step_map_forward(self):
        route = self.app.tracker.get_journey_route()
        total_hops = max(1, len(route) - 1)
        self.app.map_visualizer.step_forward(total_hops)
        self.render_full_map()

    def _on_map_speed_change(self, value):
        self.app.map_visualizer.set_speed(value)

    def render_full_map(self):
        fw = self.full_map_canvas.winfo_width() or 640
        fh = self.full_map_canvas.winfo_height() or 460
        route = self.app.tracker.get_journey_route()
        cam_names = [s["location"] for s in self.app.video_engine.slots]
        self.app.map_visualizer.draw_map(
            self.full_map_canvas, fw, fh,
            route=route, camera_names=cam_names,
            last_known=self.app.tracker.last_known,
            pulse_tick=self.app.pulse_tick, is_mini=False
        )
        if hasattr(self, "map_step_label") and self.app.map_visualizer.current_step_info:
            self.map_step_label.configure(text=self.app.map_visualizer.current_step_info)

    def update_view(self, video_engine):
        """
        High-performance view update with strict dirty checking.
        Eliminates 300+ redundant Tcl widget reconfigurations per second, freeing
        the Tkinter event loop so mousewheel scrolling remains 100% fluid.
        """
        current_display = self.display_mode.get()

        # Update Actual Measured Speed & Telemetry (Dirty-checked)
        telem = video_engine.get_speed_telemetry()
        spd = self.app.left_panel.global_speed_var.get()
        telemetry_text = (
            f"⚡ Actual: {telem['actual_speed_str']} (Target: {spd}) | "
            f"{telem['steps_per_sec']} steps/s | "
            f"ETA: {telem['eta_str']} | "
            f"{telem['device']}"
        )
        if getattr(self, "_last_telemetry_text", None) != telemetry_text:
            self._last_telemetry_text = telemetry_text
            self.fps_label.configure(text=telemetry_text)

        # ── Mode A: 2x2 Fixed Grid ──
        if current_display == "grid":
            for i in range(4):
                tile = self.tiles[i]
                state = video_engine.get_slot_state(i)
                if state and state["enabled"]:
                    tile["_last_disabled"] = False
                    pct = state.get("progress_pct", 0.0)
                    pstr = state.get("progress_str", "0.0%")
                    spd_val = state.get("speed", "10x")
                    ended = state.get("ended", False)

                    # Determine target footer values
                    if ended:
                        target_pct = 1.0
                        target_pbar_col = "#04a57a"
                        target_plbl_text = f"🏁 100% DONE ({pstr})"
                        target_plbl_col = "#04a57a"
                    elif state.get("is_file", False):
                        target_pct = pct / 100.0
                        target_pbar_col = ACCENT_GREEN
                        target_plbl_text = f"⚡{spd_val} · {pstr}"
                        target_plbl_col = ACCENT_GREEN
                    else:
                        target_pct = 1.0
                        target_pbar_col = ACCENT_BLUE
                        target_plbl_text = "🔴 LIVE FEED"
                        target_plbl_col = ACCENT_BLUE

                    # Dirty check for progress bar and label
                    if abs(tile.get("_last_pbar_val", -1.0) - target_pct) > 0.005 or tile.get("_last_pbar_col") != target_pbar_col:
                        tile["_last_pbar_val"] = target_pct
                        tile["_last_pbar_col"] = target_pbar_col
                        tile["pbar"].set(target_pct)
                        tile["pbar"].configure(progress_color=target_pbar_col)

                    if tile.get("_last_plbl_text") != target_plbl_text or tile.get("_last_plbl_col") != target_plbl_col:
                        tile["_last_plbl_text"] = target_plbl_text
                        tile["_last_plbl_col"] = target_plbl_col
                        tile["plabel"].configure(text=target_plbl_text, text_color=target_plbl_col)

                    if state["frame"] is not None:
                        frame = state["frame"]
                        loc = state["location"]
                        spotted = state.get("spotted", False)

                        if spotted:
                            target_hdr_text = f"🚨 CAM {i+1}: {loc} [TARGET FOUND!]"
                            target_hdr_fg = "#801010"
                            target_hdr_col = "#ffffff"
                            target_border_col = ACCENT_GREEN
                            target_border_w = 2
                        else:
                            target_hdr_text = f"● CAM {i+1}: {loc} (Scanning)"
                            target_hdr_fg = CARD_BG
                            target_hdr_col = TEXT_LIGHT
                            target_border_col = "#22223a"
                            target_border_w = 1

                        if tile.get("_last_hdr_text") != target_hdr_text or tile.get("_last_hdr_fg") != target_hdr_fg:
                            tile["_last_hdr_text"] = target_hdr_text
                            tile["_last_hdr_fg"] = target_hdr_fg
                            tile["header"].configure(text=target_hdr_text, fg_color=target_hdr_fg, text_color=target_hdr_col)

                        if tile.get("_last_border_col") != target_border_col or tile.get("_last_border_w") != target_border_w:
                            tile["_last_border_col"] = target_border_col
                            tile["_last_border_w"] = target_border_w
                            tile["frame"].configure(border_color=target_border_col, border_width=target_border_w)

                        # Only re-render CTkImage if a new frame sequence has arrived
                        frame_seq = state.get("frame_seq", 0)
                        if frame_seq != tile.get("last_rendered_seq", -1):
                            tile["last_rendered_seq"] = frame_seq
                            tw = max(10, tile["label"].winfo_width() or 310)
                            th = max(10, tile["label"].winfo_height() or 210)
                            fh, fw = frame.shape[:2]
                            scale = min(tw / fw, th / fh)
                            nw, nh = max(1, int(fw * scale)), max(1, int(fh * scale))
                            resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
                            frame_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                            pimg = Image.fromarray(frame_rgb)
                            cimg = ctk.CTkImage(pimg, size=(nw, nh))
                            tile["label"].configure(image=cimg, text="")
                            tile["label"].image = cimg

                elif state and not state["enabled"]:
                    if not tile.get("_last_disabled", False):
                        self._clear_tile_widgets(i, tile, label_text=f"Camera {i+1}\n(Disabled / Removed)", plabel_text="Standby")

        # ── Mode B: Interactive Full Radar Map ──
        elif current_display == "map":
            self.render_full_map()

        # ── Mode C: Single Camera / Spotlight ──
        else:
            if current_display == "spotlight":
                idx = video_engine.last_spotlight_idx
            else:
                idx = int(current_display.replace("cam", ""))

            state = video_engine.get_slot_state(idx)
            if state and not state["enabled"]:
                self._clear_single_view(idx)
            elif state:
                loc = state["location"]
                pct = state.get("progress_pct", 0.0)
                pstr = state.get("progress_str", "0.0%")
                spd = state.get("speed", "10x")
                ended = state.get("ended", False)

                if ended:
                    target_pct = 1.0
                    target_col = "#04a57a"
                    target_txt = f"CAM {idx+1} 🏁 FINISHED 100% ({pstr})"
                elif state.get("is_file", False):
                    target_pct = pct / 100.0
                    target_col = ACCENT_GREEN
                    target_txt = f"CAM {idx+1} [⚡{spd}] · {pstr}"
                else:
                    target_pct = 1.0
                    target_col = ACCENT_BLUE
                    target_txt = f"CAM {idx+1} 🔴 LIVE STREAM"

                if abs(getattr(self, "_last_single_pbar", -1.0) - target_pct) > 0.005 or getattr(self, "_last_single_col", None) != target_col:
                    self._last_single_pbar = target_pct
                    self._last_single_col = target_col
                    self.single_pbar.set(target_pct)
                    self.single_pbar.configure(progress_color=target_col)

                if getattr(self, "_last_single_txt", None) != target_txt:
                    self._last_single_txt = target_txt
                    self.single_plabel.configure(text=target_txt, text_color=target_col)

                spotted = state.get("spotted", False)
                if spotted:
                    hdr_text = f"🚨 SPOTLIGHT: CAM {idx+1} — {loc} [TARGET SPOTTED!]"
                    hdr_fg = "#801010"
                else:
                    hdr_text = f"FOCUSED: CAM {idx+1} — {loc}"
                    hdr_fg = CARD_BG

                if getattr(self, "_last_single_hdr_txt", None) != hdr_text or getattr(self, "_last_single_hdr_fg", None) != hdr_fg:
                    self._last_single_hdr_txt = hdr_text
                    self._last_single_hdr_fg = hdr_fg
                    self.single_header.configure(text=hdr_text, fg_color=hdr_fg)

                if state["frame"] is not None:
                    frame = state["frame"]
                    frame_seq = state.get("frame_seq", 0)
                    if frame_seq != self.last_single_rendered_seq or idx != self.last_single_rendered_idx:
                        self.last_single_rendered_seq = frame_seq
                        self.last_single_rendered_idx = idx
                        sw = max(10, self.single_label.winfo_width() or 640)
                        sh = max(10, self.single_label.winfo_height() or 460)
                        fh, fw = frame.shape[:2]
                        scale = min(sw / fw, sh / fh)
                        nw, nh = max(1, int(fw * scale)), max(1, int(fh * scale))
                        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
                        frame_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                        pimg = Image.fromarray(frame_rgb)
                        cimg = ctk.CTkImage(pimg, size=(nw, nh))
                        self.single_label.configure(image=cimg, text="")
                        self.single_label.image = cimg

        self.update_last_seen_banner()

    def update_last_seen_banner(self):
        """Dirty-checked banner update with Multi-Camera Overlap Fusion styling."""
        last_known = self.app.tracker.last_known
        if last_known.get("camera"):
            vt = last_known.get("video_time", "")
            vt_text = f"  ·  ⏱ Time: {vt}" if vt else ""
            is_overlap = last_known.get("is_overlap", False)
            overlap_cams = last_known.get("overlap_cameras", [])

            if is_overlap and overlap_cams:
                cams_str = ", ".join(overlap_cams)
                loc_text = (
                    f"📍 Primary: {last_known['camera']}{vt_text}  ·  "
                    f"🤝 Overlap Coverage: {cams_str}  ·  Confidence: {int(last_known['confidence']*100)}%"
                )
                color = "#f5b8ff"
                bg_col = "#380d44"
            else:
                loc_text = (
                    f"📍 Current / Last Known Location: {last_known['camera']}"
                    f"{vt_text}  ·  Confidence: {int(last_known['confidence']*100)}%"
                )
                is_lost = self.app.tracker.is_target_lost() if self.app.video_engine.is_running else False
                color = ACCENT_RED if is_lost else ACCENT_GREEN
                bg_col = "#2e0a0a" if is_lost else "#0a2e1a"
        else:
            loc_text = "Last Known Location: Ready to search"
            color = TEXT_DIM
            bg_col = CARD_BG

        if getattr(self, "_last_banner_text", None) != loc_text or getattr(self, "_last_banner_bg", None) != bg_col:
            self._last_banner_text = loc_text
            self._last_banner_bg = bg_col
            self.last_seen_banner.configure(fg_color=bg_col)
            self.last_seen_label.configure(text=loc_text, text_color=color)

    def _clear_tile_widgets(self, idx, tile, label_text=None, plabel_text="Standby"):
        tile["_last_disabled"] = True
        tile["_last_pbar_val"] = None
        tile["_last_pbar_col"] = None
        tile["_last_plbl_text"] = None
        tile["_last_hdr_text"] = None
        tile["_last_border_col"] = None
        tile["header"].configure(text=f"CAM {idx+1} — Disabled", fg_color="#181828", text_color=TEXT_DIM)
        tile["frame"].configure(border_color="#181828", border_width=1)
        tile["label"].configure(image=None, text=label_text or f"Camera {idx+1}\n(Disabled in Settings)")
        tile["label"].image = None
        tile["pbar"].set(0.0)
        tile["plabel"].configure(text=plabel_text, text_color=TEXT_DIM)
        tile["last_rendered_seq"] = -1

    def _clear_single_view(self, idx=None):
        if getattr(self, "_last_single_cleared_idx", None) == idx and self.last_single_rendered_seq == -1:
            return
        self._last_single_cleared_idx = idx
        self._last_single_hdr_txt = None
        self._last_single_hdr_fg = None
        self._last_single_txt = None
        self._last_single_pbar = None
        self._last_single_col = None
        title = f"CAM {idx+1} — Disabled / Removed" if idx is not None else "Focused Camera"
        self.single_header.configure(text=title, fg_color=CARD_BG)
        self.single_label.configure(image=None, text="No Feed Selected")
        self.single_label.image = None
        self.single_pbar.set(0.0)
        self.single_plabel.configure(text="Standby", text_color=TEXT_DIM)
        self.last_single_rendered_seq = -1
        self.last_single_rendered_idx = -1

    def reset_slot_tile(self, idx):
        """Resets a single camera tile cleanly when its video/source is removed."""
        if 0 <= idx < len(self.tiles):
            tile = self.tiles[idx]
            self._clear_tile_widgets(idx, tile, label_text=f"Camera {idx+1}\n(Removed / No Source)", plabel_text="Removed")
            if self.display_mode.get() == f"cam{idx}":
                self._clear_single_view(idx)

    def reset_tiles(self):
        for i, tile in enumerate(self.tiles):
            tile["_last_disabled"] = None
            tile["_last_pbar_val"] = None
            tile["_last_pbar_col"] = None
            tile["_last_plbl_text"] = None
            tile["_last_hdr_text"] = None
            tile["_last_border_col"] = None
            tile["header"].configure(text="Standby", fg_color=CARD_BG)
            tile["frame"].configure(border_color="#22223a")
            tile["label"].configure(image=None, text="Camera Standby")
            tile["label"].image = None
            tile["pbar"].set(0.0)
            tile["plabel"].configure(text="0.0%", text_color=TEXT_DIM)
            tile["last_rendered_seq"] = -1
        self._last_single_cleared_idx = None
        self._clear_single_view(None)
