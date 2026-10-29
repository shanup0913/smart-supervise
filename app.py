"""
Smart Supervise — Main Application Bootstrap & UI Orchestrator
Modular architecture: UI panels in ui/, background video streaming in core/video_engine.py
"""

import sys
import os
import gc
import cv2
import threading
import customtkinter as ctk
from PIL import Image

# Optimize OpenCV for multi-core: prevent internal thread pools from thrashing CPU
cv2.setNumThreads(1)

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from core.cctv_scanner import CCTVScanner
from core.location_tracker import LocationTracker
from core.map_visualizer import MapVisualizer
from core.video_engine import VideoEngine

from ui.theme import (
    DARK_BG, PANEL_BG, ACCENT_BLUE,
    ACCENT_GREEN, ACCENT_AMBER, TEXT_DIM
)
from ui.left_panel import LeftPanel
from ui.center_grid import CenterGrid
from ui.right_panel import RightPanel


class SmartSuperviseApp(ctk.CTk):
    """
    SmartSuperviseApp: Main Application Window & Orchestrator.
    Decoupled modular architecture for zero lag and silky-smooth rendering.
    """
    def __init__(self):
        super().__init__()

        # Window Setup
        self.title("Smart Supervise — Time-Synchronized Multi-Camera CCTV Intelligence")
        self.geometry("1360x780")
        self.minsize(1050, 620)
        self.configure(fg_color=DARK_BG)

        # Core Engines
        self.scanner = CCTVScanner()
        self.tracker = LocationTracker(lost_timeout_seconds=30)
        self.map_visualizer = MapVisualizer()
        self.video_engine = VideoEngine(self.scanner, self.tracker)
        self.pulse_tick = 0

        # Build Modular GUI
        self._build_titlebar()
        self._build_panels()

        # Update Last Seen Banner initial state
        self.center_grid.update_last_seen_banner()

        # Start periodic GUI render loop (60ms ~ 16 FPS: keeps left panel scrolling silky smooth)
        self.after(60, self._render_loop)

        # Window close handler
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_titlebar(self):
        bar = ctk.CTkFrame(self, fg_color=PANEL_BG, height=50, corner_radius=0)
        bar.pack(fill="x", side="top")
        bar.pack_propagate(False)

        ctk.CTkLabel(
            bar, text="⬡  SMART SUPERVISE",
            font=ctk.CTkFont(family="Segoe UI", size=18, weight="bold"),
            text_color=ACCENT_BLUE
        ).pack(side="left", padx=16, pady=10)

        ctk.CTkLabel(
            bar, text="Time-Synchronized Multi-Camera Search & Journey Path Engine",
            font=ctk.CTkFont(size=12),
            text_color=TEXT_DIM
        ).pack(side="left", padx=0, pady=10)

        self.status_dot = ctk.CTkLabel(
            bar, text="●  IDLE",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=TEXT_DIM
        )
        self.status_dot.pack(side="right", padx=14)

        self.global_clock_label = ctk.CTkLabel(
            bar, text="⏱  CLOCK: 00:00:00",
            font=ctk.CTkFont(family="Courier New", size=13, weight="bold"),
            text_color=ACCENT_AMBER
        )
        self.global_clock_label.pack(side="right", padx=14)

        self.active_feeds_label = ctk.CTkLabel(
            bar, text="📹  FEEDS: Standby",
            font=ctk.CTkFont(family="Courier New", size=12, weight="bold"),
            text_color=ACCENT_BLUE
        )
        self.active_feeds_label.pack(side="right", padx=14)

    def _build_panels(self):
        main = ctk.CTkFrame(self, fg_color=DARK_BG)
        main.pack(fill="both", expand=True, padx=8, pady=(4, 8))

        main.grid_columnconfigure(0, weight=0, minsize=290)  # Left panel
        main.grid_columnconfigure(1, weight=1)               # Center video grid
        main.grid_columnconfigure(2, weight=0, minsize=320)  # Right panel
        main.grid_rowconfigure(0, weight=1)

        self.left_panel = LeftPanel(main, self)
        self.left_panel.grid(row=0, column=0, padx=(0, 6), pady=0, sticky="nsew")

        self.center_grid = CenterGrid(main, self)
        self.center_grid.grid(row=0, column=1, padx=4, pady=0, sticky="nsew")

        self.right_panel = RightPanel(main, self)
        self.right_panel.grid(row=0, column=2, padx=(4, 0), pady=0, sticky="nsew")

    def _render_loop(self):
        """Throttled UI refresh loop: decouples rendering from video decoding for zero lag and no GDI leaks."""
        try:
            self.pulse_tick += 1

            # Garbage collect periodically to ensure OS GDI handles never hit 10,000 threshold
            if self.pulse_tick % 100 == 0:
                gc.collect()

            if self.video_engine.is_running:
                # Update clock in titlebar
                clock_str, _ = self.video_engine.get_clock_display()
                self.global_clock_label.configure(text=f"⏱  CLOCK: {clock_str}")

                # Update active feeds telemetry in titlebar
                active_slots = [s for s in self.video_engine.slots if s["enabled"]]
                remaining_slots = [s for s in active_slots if not s["ended"]]
                self.active_feeds_label.configure(
                    text=f"📹  FEEDS: {len(remaining_slots)}/{len(active_slots)} Active"
                )

                # Update Center Video Matrix, Left Panel Cards, and Right Telemetry
                self.center_grid.update_view(self.video_engine)
                self.left_panel.update_camera_progress(self.video_engine)
                self.right_panel.update_view()
            elif self.center_grid.display_mode.get() == "map":
                # Keep map animations alive even when stopped
                self.center_grid.render_full_map()
                self.right_panel.update_view()
        except Exception:
            pass
        finally:
            self.after(60, self._render_loop)

    def set_status(self, text, color=TEXT_DIM):
        """Updates the status pill in the top titlebar."""
        self.status_dot.configure(text=f"●  {text}", text_color=color)

    def on_scan_finished(self):
        """Callback invoked when all recorded video feeds reach their conclusion."""
        def _update_ui():
            self.set_status("SEARCH FINISHED — ALL FEEDS COMPLETE", ACCENT_GREEN)
            self.active_feeds_label.configure(text="📹  FEEDS: All Completed")
            self.left_panel.start_btn.configure(state="normal")
            self.left_panel.stop_btn.configure(state="disabled")
        self.after(0, _update_ui)

    def _on_close(self):
        """Safely stops video engine and shuts down window without access violations."""
        try:
            self.video_engine.stop()
        except Exception:
            pass
        self.destroy()


# ─── Application Entry Point ─────────────────────────────────────────────────
if __name__ == "__main__":
    app = SmartSuperviseApp()
    app.mainloop()
