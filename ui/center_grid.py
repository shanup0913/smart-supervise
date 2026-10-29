"""
CenterGrid: 2x2 CCTV Matrix, focused camera zoom, Spotlight mode,
interactive 2D Rapido GPS Radar Map, and Last Known Location telemetry.
"""

import cv2
from PIL import Image
import tkinter as tk
import customtkinter as ctk

from ui.theme import (
    DARK_BG, PANEL_BG, CARD_BG, ACCENT_BLUE,
    ACCENT_GREEN, ACCENT_RED, ACCENT_AMBER,
    TEXT_LIGHT, TEXT_DIM
)


class CenterGrid(ctk.CTkFrame):
    """Center surveillance viewport supporting 2x2 grid, spotlight, single cam, and full radar map."""
    def __init__(self, parent, app):
        super().__init__(parent, fg_color=PANEL_BG, corner_radius=12)
        self.app = app

        self.display_mode = ctk.StringVar(value="grid")
        self._build_ui()

    def _build_ui(self):
        # Top Control Bar
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=10, pady=(8, 4))

        ctk.CTkLabel(
            header, text="DISPLAY VIEW:",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=TEXT_DIM
        ).pack(side="left")

        self.view_buttons = ctk.CTkSegmentedButton(
            header,
            values=["⊞ 4-Grid", "Cam 1", "Cam 2", "Cam 3", "Cam 4", "Spotlight", "🗺️ Rapido Map"],
            command=self._on_display_mode_change,
            selected_color=ACCENT_BLUE,
            height=26,
            font=ctk.CTkFont(size=10)
        )
        self.view_buttons.set("⊞ 4-Grid")
        self.view_buttons.pack(side="left", padx=8)

        self.fps_label = ctk.CTkLabel(
            header, text="FPS: —",
            font=ctk.CTkFont(size=11), text_color=TEXT_DIM
        )
        self.fps_label.pack(side="right")

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

            vlabel = ctk.CTkLabel(
                tile_card, text=f"Camera {i+1}\n(Standby / Disabled)",
                font=ctk.CTkFont(size=11), text_color=TEXT_DIM
            )
            vlabel.pack(fill="both", expand=True, padx=2, pady=2)

            self.tiles.append({
                "frame": tile_card,
                "header": hdr,
                "label": vlabel
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
        self.single_label = ctk.CTkLabel(
            self.single_container, text="No Feed Selected",
            font=ctk.CTkFont(size=14), text_color=TEXT_DIM,
            fg_color=DARK_BG, corner_radius=8
        )
        self.single_label.pack(fill="both", expand=True, padx=2, pady=2)

        # ── View C: Interactive 2D Rapido Live Map Container ──
        self.map_container = ctk.CTkFrame(self, fg_color="transparent")
        map_hdr = ctk.CTkFrame(self.map_container, fg_color=CARD_BG, corner_radius=6, height=28)
        map_hdr.pack(fill="x", padx=2, pady=(2, 4))
        ctk.CTkLabel(
            map_hdr, text="🗺️ RAPIDO LIVE SURVEILLANCE RADAR & TRANSIT ROUTE MAP",
            font=ctk.CTkFont(size=11, weight="bold"), text_color=ACCENT_AMBER
        ).pack(side="left", padx=8)

        self.full_map_canvas = tk.Canvas(self.map_container, bg="#0a0b16", highlightthickness=0)
        self.full_map_canvas.pack(fill="both", expand=True, padx=2, pady=2)

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

    def switch_to_map(self):
        self.view_buttons.set("🗺️ Rapido Map")
        self._on_display_mode_change("🗺️ Rapido Map")

    def _on_display_mode_change(self, mode):
        self.grid_container.pack_forget()
        self.single_container.pack_forget()
        self.map_container.pack_forget()

        if mode == "⊞ 4-Grid":
            self.grid_container.pack(fill="both", expand=True, padx=6, pady=(2, 6))
            self.display_mode.set("grid")
        elif mode == "Spotlight":
            self.single_container.pack(fill="both", expand=True, padx=6, pady=(2, 6))
            self.display_mode.set("spotlight")
        elif "Map" in mode:
            self.map_container.pack(fill="both", expand=True, padx=6, pady=(2, 6))
            self.display_mode.set("map")
            self.render_full_map()
        else:
            idx = int(mode.replace("Cam ", "")) - 1
            self.single_container.pack(fill="both", expand=True, padx=6, pady=(2, 6))
            self.display_mode.set(f"cam{idx}")

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

    def update_view(self, video_engine):
        current_display = self.display_mode.get()

        # Update FPS label
        fps = video_engine.current_fps
        spd = self.app.left_panel.global_speed_var.get()
        self.fps_label.configure(text=f"Speed: {spd} | FPS: {fps:.1f}")

        # ── Mode A: 2x2 Fixed Grid ──
        if current_display == "grid":
            for i in range(4):
                tile = self.tiles[i]
                state = video_engine.get_slot_state(i)
                if state and state["enabled"] and state["frame"] is not None:
                    frame = state["frame"]
                    loc = state["location"]
                    if state["spotted"]:
                        tile["header"].configure(
                            text=f"🚨 CAM {i+1}: {loc} [TARGET FOUND!]",
                            fg_color="#801010", text_color="#ffffff"
                        )
                        tile["frame"].configure(border_color=ACCENT_GREEN, border_width=2)
                    else:
                        tile["header"].configure(
                            text=f"● CAM {i+1}: {loc} (Scanning)",
                            fg_color=CARD_BG, text_color=TEXT_LIGHT
                        )
                        tile["frame"].configure(border_color="#22223a", border_width=1)

                    # Ultra-fast proportional OpenCV C++ resize
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
                    tile["header"].configure(text=f"CAM {i+1} — Disabled", fg_color="#181828", text_color=TEXT_DIM)
                    tile["frame"].configure(border_color="#181828", border_width=1)
                    tile["label"].configure(image=None, text=f"Camera {i+1}\n(Disabled in Settings)")

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
            if state:
                loc = state["location"]
                if state["spotted"]:
                    self.single_header.configure(text=f"🚨 SPOTLIGHT: CAM {idx+1} — {loc} [TARGET SPOTTED!]", fg_color="#801010")
                else:
                    self.single_header.configure(text=f"FOCUSED: CAM {idx+1} — {loc}", fg_color=CARD_BG)

                if state["frame"] is not None:
                    frame = state["frame"]
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
        last_known = self.app.tracker.last_known
        if last_known.get("camera"):
            vt = last_known.get("video_time", "")
            vt_text = f"  ·  ⏱ Time: {vt}" if vt else ""
            loc_text = (
                f"📍 Current / Last Known Location: {last_known['camera']}"
                f"{vt_text}  ·  Confidence: {int(last_known['confidence']*100)}%"
            )
            is_lost = self.app.tracker.is_target_lost() if self.app.video_engine.is_running else False
            color = ACCENT_RED if is_lost else ACCENT_GREEN
            self.last_seen_banner.configure(fg_color="#0a2e1a" if not is_lost else "#2e0a0a")
            self.last_seen_label.configure(text=loc_text, text_color=color)
        else:
            self.last_seen_label.configure(
                text="Last Known Location: Scanning synchronized feeds...",
                text_color=TEXT_DIM
            )
            self.last_seen_banner.configure(fg_color=CARD_BG)

    def reset_tiles(self):
        for tile in self.tiles:
            tile["header"].configure(text="Standby", fg_color=CARD_BG)
            tile["frame"].configure(border_color="#22223a")
            tile["label"].configure(image=None, text="Camera Standby")
        self.single_header.configure(text="Focused Camera", fg_color=CARD_BG)
        self.single_label.configure(image=None, text="No Feed Selected")
