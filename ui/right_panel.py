"""
RightPanel: Live Rapido mini-radar map canvas, transit route cards,
ASCII journey flowchart modal, chronological sightings timeline, and report exports.
"""

import os
import tkinter as tk
import customtkinter as ctk
from tkinter import messagebox

from ui.theme import (
    DARK_BG, PANEL_BG, CARD_BG, ACCENT_BLUE,
    ACCENT_GREEN, ACCENT_RED, ACCENT_AMBER,
    TEXT_LIGHT, TEXT_DIM
)


class RightPanel(ctk.CTkFrame):
    """Right telemetry sidebar: live radar mini-map, journey route, timeline log, and exports."""
    def __init__(self, parent, app):
        super().__init__(parent, fg_color=PANEL_BG, corner_radius=12)
        self.app = app

        # Dirty checking flags to prevent UI lag
        self._last_trail_text = ""
        self._last_route_text = ""

        self._build_ui()

    def _build_ui(self):
        # ── 1. Action Buttons & Stats Pinned to BOTTOM First ──
        bottom_box = ctk.CTkFrame(self, fg_color="transparent")
        bottom_box.pack(side="bottom", fill="x", padx=8, pady=(0, 8))

        ctk.CTkButton(
            bottom_box, text="Clear Timeline",
            command=self._clear_timeline,
            fg_color=CARD_BG, hover_color=DARK_BG,
            border_width=1, border_color=ACCENT_RED,
            corner_radius=8, height=28,
            font=ctk.CTkFont(size=11)
        ).pack(side="bottom", fill="x", pady=(2, 0))

        ctk.CTkButton(
            bottom_box, text="Export Movement Report (CSV+JSON)",
            command=self._export_timeline,
            fg_color=CARD_BG, hover_color=DARK_BG,
            border_width=1, border_color=ACCENT_AMBER,
            corner_radius=8, height=30,
            font=ctk.CTkFont(size=11)
        ).pack(side="bottom", fill="x", pady=(4, 0))

        # Stats strip
        stats = ctk.CTkFrame(bottom_box, fg_color=CARD_BG, corner_radius=8)
        stats.pack(side="bottom", fill="x", pady=(4, 0))

        self.sightings_label = ctk.CTkLabel(
            stats, text="Total Sightings: 0",
            font=ctk.CTkFont(size=11), text_color=TEXT_DIM
        )
        self.sightings_label.pack(pady=3, padx=8, anchor="w")

        self.time_since_label = ctk.CTkLabel(
            stats, text="Last seen: —",
            font=ctk.CTkFont(size=11), text_color=TEXT_DIM
        )
        self.time_since_label.pack(pady=(0, 3), padx=8, anchor="w")

        # ── 2. Top Section: Journey Route Trail ──
        ctk.CTkLabel(
            self, text="MOVEMENT JOURNEY PATH",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=ACCENT_AMBER
        ).pack(side="top", pady=(8, 4), padx=12, anchor="w")

        journey_card = ctk.CTkFrame(self, fg_color=CARD_BG, corner_radius=8)
        journey_card.pack(side="top", fill="x", padx=8, pady=(0, 6))

        ctk.CTkLabel(
            journey_card, text="📍 RAPIDO LIVE ROUTE RADAR (Click for Full Map):",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=ACCENT_GREEN
        ).pack(pady=(4, 1), padx=8, anchor="w")

        # Mini Live Rapido Radar Map Canvas
        self.mini_map_canvas = tk.Canvas(journey_card, height=130, bg="#0e0f1e", highlightthickness=0, cursor="hand2")
        self.mini_map_canvas.pack(fill="x", padx=6, pady=(1, 4))
        self.mini_map_canvas.bind("<Button-1>", lambda e: self.app.center_grid.switch_to_map())

        self.journey_route_label = ctk.CTkLabel(
            journey_card, text="Waiting for sightings...",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=TEXT_LIGHT,
            wraplength=280, justify="left"
        )
        self.journey_route_label.pack(padx=8, pady=(0, 4), anchor="w")

        btn_box = ctk.CTkFrame(journey_card, fg_color="transparent")
        btn_box.pack(fill="x", padx=6, pady=(0, 6))

        ctk.CTkButton(
            btn_box, text="🗺️ Open Full Map",
            command=self.app.center_grid.switch_to_map,
            fg_color=DARK_BG, hover_color=PANEL_BG,
            border_width=1, border_color=ACCENT_BLUE,
            corner_radius=6, height=24,
            font=ctk.CTkFont(size=9, weight="bold")
        ).pack(side="left", fill="x", expand=True, padx=(0, 2))

        ctk.CTkButton(
            btn_box, text="📊 Flowchart",
            command=self._show_journey_modal,
            fg_color=DARK_BG, hover_color=PANEL_BG,
            border_width=1, border_color=ACCENT_AMBER,
            corner_radius=6, height=24,
            font=ctk.CTkFont(size=9, weight="bold")
        ).pack(side="left", fill="x", expand=True, padx=(2, 0))

        ctk.CTkLabel(
            self, text="DETAILED TIMELINE LOG",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=TEXT_DIM
        ).pack(side="top", pady=(2, 2), padx=12, anchor="w")

        # ── 3. Middle Area: Timeline Textbox ──
        self.timeline_box = ctk.CTkTextbox(
            self, fg_color=DARK_BG, corner_radius=8,
            font=ctk.CTkFont(family="Courier New", size=10),
            text_color=TEXT_LIGHT,
            wrap="word", state="disabled"
        )
        self.timeline_box.pack(side="top", fill="both", expand=True, padx=8, pady=(0, 6))

    def update_view(self):
        # Update Mini Live Rapido Map
        mw = self.mini_map_canvas.winfo_width() or 290
        mh = self.mini_map_canvas.winfo_height() or 130
        route = self.app.tracker.get_journey_route()
        cam_names = [s["location"] for s in self.app.video_engine.slots]
        self.app.map_visualizer.draw_map(
            self.mini_map_canvas, mw, mh,
            route=route, camera_names=cam_names,
            last_known=self.app.tracker.last_known,
            pulse_tick=self.app.pulse_tick, is_mini=True
        )

        # Update stats
        summary = self.app.tracker.get_timeline_summary()
        self.sightings_label.configure(text=f"Total Sightings: {summary['total_sightings']}")
        if summary["time_since_last_seen"]:
            self.time_since_label.configure(text=f"Last seen: {summary['time_since_last_seen']} ago")
        else:
            self.time_since_label.configure(text="Last seen: —")

        # Dirty-check route text
        route_arrow = self.app.tracker.get_journey_arrow_str()
        if route_arrow != self._last_route_text:
            self.journey_route_label.configure(text=route_arrow)
            self._last_route_text = route_arrow

        # Dirty-check timeline text
        trail_str = self.app.tracker.get_movement_trail()
        if trail_str != self._last_trail_text:
            self.timeline_box.configure(state="normal")
            self.timeline_box.delete("1.0", "end")
            self.timeline_box.insert("1.0", trail_str)
            self.timeline_box.see("end")
            self.timeline_box.configure(state="disabled")
            self._last_trail_text = trail_str

    def _show_journey_modal(self):
        journey_text = self.app.tracker.get_formatted_journey()
        modal = ctk.CTkToplevel(self)
        modal.title("Complete Movement Journey Map")
        modal.geometry("640x540")
        modal.configure(fg_color=DARK_BG)
        modal.grab_set()

        ctk.CTkLabel(
            modal, text="🗺 COMPLETE JOURNEY FLOWCHART",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=ACCENT_GREEN
        ).pack(pady=(12, 6))

        tbox = ctk.CTkTextbox(
            modal, fg_color=PANEL_BG, corner_radius=10,
            font=ctk.CTkFont(family="Courier New", size=11),
            text_color=TEXT_LIGHT, wrap="none"
        )
        tbox.pack(fill="both", expand=True, padx=14, pady=(0, 10))
        tbox.insert("1.0", journey_text)
        tbox.configure(state="disabled")

        ctk.CTkButton(
            modal, text="Close",
            command=modal.destroy,
            fg_color=ACCENT_BLUE, width=100
        ).pack(pady=(0, 12))

    def _export_timeline(self):
        if not self.app.tracker.timeline:
            messagebox.showinfo("Export", "No sightings recorded yet to export.")
            return

        csv_path, json_path = self.app.tracker.export_to_csv_and_json()
        messagebox.showinfo(
            "Export Successful",
            f"Movement journey exported successfully!\n\nCSV:  {os.path.basename(csv_path)}\nJSON: {os.path.basename(json_path)}\n\nSaved in reports/ folder."
        )

    def _clear_timeline(self):
        name = self.app.left_panel.target_name_var.get().strip() or "Target Person"
        self.app.tracker.set_target(name)
        self._last_trail_text = ""
        self._last_route_text = ""
        self.timeline_box.configure(state="normal")
        self.timeline_box.delete("1.0", "end")
        self.timeline_box.configure(state="disabled")
        self.journey_route_label.configure(text="Waiting for sightings...")
        self.sightings_label.configure(text="Total Sightings: 0")
        self.time_since_label.configure(text="Last seen: —")
        self.app.center_grid.update_last_seen_banner()
