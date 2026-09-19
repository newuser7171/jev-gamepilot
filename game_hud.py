"""
Cyber Desktop GUI for Jev-GamePilot.
Built with CustomTkinter to deliver a high-tech gaming HUD with live computer vision feed,
Jev System One brain telemetry, window snapping, and emergency controls.
"""

import os
import subprocess
import threading
import time
import webbrowser
import cv2
import customtkinter as ctk
import numpy as np
from PIL import Image, ImageTk
from pilot_core import PilotCore

# Cyber aesthetic styling
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class GamePilotHUD(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Jev-GamePilot // Autonomous AI Gaming Agent")
        self.geometry("1100x720")
        self.minsize(980, 650)
        self.configure(fg_color="#0a0b10")

        self.core = PilotCore()
        self.core.set_telemetry_callback(self._on_telemetry_update)

        # State tracking
        self.current_img_tk: ImageTk.PhotoImage | None = None
        self.is_monitoring = False
        self.is_armed = False

        self._build_header()
        self._build_main_layout()

        # Keyboard shortcuts
        self.bind("<Escape>", lambda e: self.emergency_stop())
        self.bind("<F6>", lambda e: self.toggle_pilot(armed=True))

        # Initial snap attempt
        self.after(500, self._auto_detect_game_window)

    def _build_header(self):
        header_frame = ctk.CTkFrame(self, fg_color="#12131c", corner_radius=0, height=60)
        header_frame.pack(fill="x", side="top", padx=0, pady=0)

        # Title & Subtitle
        title_box = ctk.CTkFrame(header_frame, fg_color="transparent")
        title_box.pack(side="left", padx=20, pady=10)

        title_lbl = ctk.CTkLabel(
            title_box,
            text="⚡ JEV-GAMEPILOT",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color="#00ffcc",
        )
        title_lbl.pack(side="left")

        sub_lbl = ctk.CTkLabel(
            title_box,
            text=" // TYPE-SAFE SYSTEM ONE GAMING AI",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#70758a",
        )
        sub_lbl.pack(side="left", padx=5)

        # Status Badge
        self.status_badge = ctk.CTkLabel(
            header_frame,
            text="● SYSTEM STANDBY",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#70758a",
            fg_color="#1a1c26",
            corner_radius=12,
            padx=15,
            pady=4,
        )
        self.status_badge.pack(side="right", padx=20)

    def _build_main_layout(self):
        # Container
        container = ctk.CTkFrame(self, fg_color="transparent")
        container.pack(fill="both", expand=True, padx=15, pady=15)

        # Left Column: Controls & Window Calibration (width: 360)
        left_col = ctk.CTkFrame(container, fg_color="#12131c", corner_radius=10, width=360)
        left_col.pack(side="left", fill="y", padx=(0, 10), pady=0)
        left_col.pack_propagate(False)

        # Right Column: Viewport & Jev Brain Telemetry
        right_col = ctk.CTkFrame(container, fg_color="transparent")
        right_col.pack(side="right", fill="both", expand=True, padx=0, pady=0)

        self._build_left_controls(left_col)
        self._build_right_viewport(right_col)

    def _build_left_controls(self, parent):
        pad_x = 15

        # 1. Game Mode Selector
        sec1_lbl = ctk.CTkLabel(
            parent,
            text="GAME ADAPTER",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#00ffcc",
        )
        sec1_lbl.pack(anchor="w", padx=pad_x, pady=(15, 5))

        self.mode_selector = ctk.CTkOptionMenu(
            parent,
            values=["Chrome Dino / Edge Surf", "Subway Surfers (3-Lane)", "Screen Chess Bot (FEN)"],
            command=self._on_mode_change,
            fg_color="#1a1c26",
            button_color="#2b2f42",
            button_hover_color="#00ffcc",
            text_color="#ffffff",
            dropdown_fg_color="#1a1c26",
        )
        self.mode_selector.pack(fill="x", padx=pad_x, pady=(0, 10))

        # Launch Dino Button
        launch_dino_btn = ctk.CTkButton(
            parent,
            text="🌐 Open Offline Dino Game",
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#202230",
            hover_color="#2e3146",
            text_color="#00ffcc",
            command=self.open_offline_dino,
            height=32,
        )
        launch_dino_btn.pack(fill="x", padx=pad_x, pady=(0, 15))

        # Separator
        ctk.CTkFrame(parent, height=1, fg_color="#1f2230").pack(fill="x", padx=pad_x, pady=5)

        # 2. Window Calibration & Snapping
        sec2_lbl = ctk.CTkLabel(
            parent,
            text="VIEWPORT CALIBRATION",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#00ffcc",
        )
        sec2_lbl.pack(anchor="w", padx=pad_x, pady=(10, 5))

        win_row = ctk.CTkFrame(parent, fg_color="transparent")
        win_row.pack(fill="x", padx=pad_x, pady=(0, 5))

        self.window_dropdown = ctk.CTkOptionMenu(
            win_row,
            values=["(Auto-Detect Game Window)"],
            fg_color="#1a1c26",
            button_color="#2b2f42",
            dropdown_fg_color="#1a1c26",
            width=230,
        )
        self.window_dropdown.pack(side="left", fill="x", expand=True, padx=(0, 5))

        refresh_win_btn = ctk.CTkButton(
            win_row,
            text="🔄",
            width=36,
            height=28,
            fg_color="#1a1c26",
            hover_color="#2b2f42",
            command=self._refresh_windows,
        )
        refresh_win_btn.pack(side="right")

        snap_btn = ctk.CTkButton(
            parent,
            text="🎯 Snap to Selected Window",
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#1a1c26",
            hover_color="#2b2f42",
            text_color="#e0e0e0",
            command=self._snap_to_selected_window,
            height=28,
        )
        snap_btn.pack(fill="x", padx=pad_x, pady=(0, 10))

        # Coordinates info
        self.coords_lbl = ctk.CTkLabel(
            parent,
            text="Region: Auto Center (800x400)",
            font=ctk.CTkFont(size=11),
            text_color="#70758a",
        )
        self.coords_lbl.pack(anchor="w", padx=pad_x, pady=(0, 15))

        # Separator
        ctk.CTkFrame(parent, height=1, fg_color="#1f2230").pack(fill="x", padx=pad_x, pady=5)

        # 3. Action Buttons (Launch & Emergency Stop)
        sec3_lbl = ctk.CTkLabel(
            parent,
            text="PILOT ENGAGEMENT",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#00ffcc",
        )
        sec3_lbl.pack(anchor="w", padx=pad_x, pady=(10, 8))

        # Big Green Start Button
        self.arm_btn = ctk.CTkButton(
            parent,
            text="🚀 ENGAGE GAMEPILOT [F6]\n(Live Automated Controls)",
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#00d26a",
            hover_color="#00b058",
            text_color="#000000",
            height=50,
            command=lambda: self.toggle_pilot(armed=True),
        )
        self.arm_btn.pack(fill="x", padx=pad_x, pady=(0, 8))

        # Monitor Only Button
        self.monitor_btn = ctk.CTkButton(
            parent,
            text="👁 Monitor Only (No Inputs)",
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#1f2230",
            hover_color="#2e3146",
            text_color="#ffffff",
            height=36,
            command=lambda: self.toggle_pilot(armed=False),
        )
        self.monitor_btn.pack(fill="x", padx=pad_x, pady=(0, 10))

        # Big Red Emergency Stop Button
        self.stop_btn = ctk.CTkButton(
            parent,
            text="🛑 EMERGENCY KILL [ESC]",
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#ff2a55",
            hover_color="#d61f43",
            text_color="#ffffff",
            height=44,
            command=self.emergency_stop,
        )
        self.stop_btn.pack(fill="x", padx=pad_x, pady=(0, 10))

        # Failsafe Notice
        notice_lbl = ctk.CTkLabel(
            parent,
            text="⚠️ Fail-Safe: Press ESC or flick mouse cursor\nto top-left screen corner to instantly halt.",
            font=ctk.CTkFont(size=10),
            text_color="#70758a",
            justify="center",
        )
        notice_lbl.pack(fill="x", padx=pad_x, pady=(5, 10))

    def _build_right_viewport(self, parent):
        # 1. Live Video Viewport Card
        viewport_card = ctk.CTkFrame(parent, fg_color="#12131c", corner_radius=10)
        viewport_card.pack(fill="both", expand=True, padx=0, pady=(0, 10))

        # Top Bar of Viewport
        vp_top = ctk.CTkFrame(viewport_card, fg_color="transparent", height=35)
        vp_top.pack(fill="x", padx=15, pady=(10, 5))

        vp_title = ctk.CTkLabel(
            vp_top,
            text="LIVE VISION HUD & CONTOUR TRACKING",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#00ffcc",
        )
        vp_title.pack(side="left")

        self.fps_lbl = ctk.CTkLabel(
            vp_top,
            text="0.0 FPS // 0ms Latency",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#70758a",
        )
        self.fps_lbl.pack(side="right")

        # Video Canvas / Label
        self.video_container = ctk.CTkFrame(viewport_card, fg_color="#08090d", corner_radius=6)
        self.video_container.pack(fill="both", expand=True, padx=15, pady=(0, 15))

        self.video_lbl = ctk.CTkLabel(
            self.video_container,
            text="[ Viewport Offline // Click 'Engage GamePilot' or 'Monitor' to stream ]",
            font=ctk.CTkFont(size=13),
            text_color="#53586d",
        )
        self.video_lbl.pack(fill="both", expand=True)

        # 2. Bottom Jev Brain Telemetry Deck
        telemetry_deck = ctk.CTkFrame(parent, fg_color="#12131c", corner_radius=10, height=180)
        telemetry_deck.pack(fill="x", padx=0, pady=0)
        telemetry_deck.pack_propagate(False)

        # Grid layout inside telemetry deck
        deck_grid = ctk.CTkFrame(telemetry_deck, fg_color="transparent")
        deck_grid.pack(fill="both", expand=True, padx=15, pady=12)

        # Col 1: Action Badge & Urgency Gauge
        col1 = ctk.CTkFrame(deck_grid, fg_color="#1a1c26", corner_radius=8, width=220)
        col1.pack(side="left", fill="y", padx=(0, 10))
        col1.pack_propagate(False)

        act_title = ctk.CTkLabel(
            col1,
            text="RECOMMENDED ACTION",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color="#70758a",
        )
        act_title.pack(anchor="w", padx=12, pady=(10, 2))

        self.action_badge = ctk.CTkLabel(
            col1,
            text="RUN NORMAL",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="#00ffcc",
            fg_color="#12131c",
            corner_radius=6,
            height=38,
        )
        self.action_badge.pack(fill="x", padx=12, pady=(0, 8))

        urg_title = ctk.CTkLabel(
            col1,
            text="THREAT URGENCY",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color="#70758a",
        )
        urg_title.pack(anchor="w", padx=12, pady=(0, 2))

        self.urgency_bar = ctk.CTkProgressBar(col1, height=10, fg_color="#12131c", progress_color="#00ffcc")
        self.urgency_bar.set(0.0)
        self.urgency_bar.pack(fill="x", padx=12, pady=(0, 5))

        self.urgency_pct = ctk.CTkLabel(
            col1,
            text="0% (Clear Horizon)",
            font=ctk.CTkFont(size=10),
            text_color="#70758a",
        )
        self.urgency_pct.pack(anchor="w", padx=12)

        # Col 2: Tactical Telemetry Grid
        col2 = ctk.CTkFrame(deck_grid, fg_color="#1a1c26", corner_radius=8)
        col2.pack(side="left", fill="both", expand=True, padx=(0, 10))

        grid_inner = ctk.CTkFrame(col2, fg_color="transparent")
        grid_inner.pack(fill="both", expand=True, padx=12, pady=10)

        # Row 1: Obstacles & Speed
        self.obs_lbl = ctk.CTkLabel(
            grid_inner,
            text="Obstacle: None Detected",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#e0e0e0",
        )
        self.obs_lbl.pack(anchor="w", pady=2)

        self.dist_lbl = ctk.CTkLabel(
            grid_inner,
            text="Distance / Impact: -- px (-- ms)",
            font=ctk.CTkFont(size=11),
            text_color="#70758a",
        )
        self.dist_lbl.pack(anchor="w", pady=2)

        self.speed_lbl = ctk.CTkLabel(
            grid_inner,
            text="Game Velocity: 300.0 px/sec",
            font=ctk.CTkFont(size=11),
            text_color="#70758a",
        )
        self.speed_lbl.pack(anchor="w", pady=2)

        # Col 3: Jev Brain Reasoning & Counters
        col3 = ctk.CTkFrame(deck_grid, fg_color="#1a1c26", corner_radius=8, width=240)
        col3.pack(side="right", fill="y", padx=0)
        col3.pack_propagate(False)

        c3_title = ctk.CTkLabel(
            col3,
            text="JEV SYSTEM ONE INTUITION",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color="#70758a",
        )
        c3_title.pack(anchor="w", padx=12, pady=(10, 4))

        self.brain_source_lbl = ctk.CTkLabel(
            col3,
            text="Brain Model: jev-latest",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#00ffcc",
        )
        self.brain_source_lbl.pack(anchor="w", padx=12, pady=2)

        self.counters_lbl = ctk.CTkLabel(
            col3,
            text="Total Maneuvers:\nJumps: 0  |  Ducks: 0",
            font=ctk.CTkFont(size=11),
            text_color="#e0e0e0",
            justify="left",
        )
        self.counters_lbl.pack(anchor="w", padx=12, pady=4)

        self.fast_fall_lbl = ctk.CTkLabel(
            col3,
            text="Fast-Fall Ready: YES",
            font=ctk.CTkFont(size=10),
            text_color="#00d26a",
        )
        self.fast_fall_lbl.pack(anchor="w", padx=12)

    def _refresh_windows(self):
        titles = self.core.vision.list_windows()
        filtered = [
            t
            for t in titles
            if any(k in t.lower() for k in ["dino", "edge", "chrome", "surf", "subway", "bluestack"])
        ]
        if not filtered:
            filtered = titles[:8] if titles else ["No active windows found"]

        self.window_dropdown.configure(values=filtered)
        if filtered:
            self.window_dropdown.set(filtered[0])

    def _auto_detect_game_window(self):
        self._refresh_windows()
        # Look for Chrome Dino or Edge
        for keyword in ["dino", "t-rex", "edge", "chrome"]:
            if self.core.vision.snap_to_window(keyword):
                reg = self.core.vision.region
                self.coords_lbl.configure(
                    text=f"Snapped to '{keyword}' ({reg['width']}x{reg['height']})"
                )
                break

    def _snap_to_selected_window(self):
        selected = self.window_dropdown.get()
        if selected and "(" not in selected:
            if self.core.vision.snap_to_window(selected):
                reg = self.core.vision.region
                self.coords_lbl.configure(
                    text=f"Snapped: {reg['left']},{reg['top']} ({reg['width']}x{reg['height']})"
                )
            else:
                self.coords_lbl.configure(text="Could not lock to window dimensions.")

    def _on_mode_change(self, mode: str):
        if "Dino" in mode:
            self.core.current_game_mode = "dino"
        elif "Subway" in mode:
            self.core.current_game_mode = "runner"

    def open_offline_dino(self):
        """Launches the built-in offline Dino HTML game in the default browser."""
        dino_path = os.path.abspath("dino_game.html")
        if os.path.exists(dino_path):
            webbrowser.open(f"file:///{dino_path}")
            # Schedule auto window snap after 1.5s
            self.after(1500, lambda: self.core.vision.snap_to_window("Dino"))

    def toggle_pilot(self, armed: bool):
        if self.core.is_running:
            self.core.stop()
            self._update_status_idle()
        else:
            self.core.start(arm_inputs=armed)
            if armed:
                self.status_badge.configure(
                    text="● LIVE AUTOMATION ARMED",
                    text_color="#000000",
                    fg_color="#00d26a",
                )
                self.arm_btn.configure(text="🛑 DISENGAGE [F6]", fg_color="#ff2a55")
            else:
                self.status_badge.configure(
                    text="● MONITORING ONLY",
                    text_color="#ffffff",
                    fg_color="#0090ff",
                )
                self.monitor_btn.configure(text="🛑 Stop Monitoring", fg_color="#ff2a55")

    def emergency_stop(self):
        self.core.stop()
        self._update_status_idle()

    def _update_status_idle(self):
        self.status_badge.configure(
            text="● SYSTEM STANDBY",
            text_color="#70758a",
            fg_color="#1a1c26",
        )
        self.arm_btn.configure(
            text="🚀 ENGAGE GAMEPILOT [F6]\n(Live Automated Controls)",
            fg_color="#00d26a",
        )
        self.monitor_btn.configure(
            text="👁 Monitor Only (No Inputs)",
            fg_color="#1f2230",
        )
        self.action_badge.configure(
            text="STANDBY", text_color="#70758a", fg_color="#12131c"
        )
        self.urgency_bar.set(0.0)

    def _on_telemetry_update(self, data: dict):
        """Called from PilotCore loop thread on every frame."""
        self.after(0, self._render_telemetry_ui, data)

    def _render_telemetry_ui(self, data: dict):
        # Update FPS
        self.fps_lbl.configure(text=f"{data['fps']} FPS // Real-Time Vision Loop")

        # Update Frame
        frame = data.get("frame")
        if frame is not None:
            # Resize frame to fit container
            container_w = max(400, self.video_container.winfo_width())
            container_h = max(200, self.video_container.winfo_height())

            h, w, _ = frame.shape
            scale = min(container_w / w, container_h / h)
            new_w = max(50, int(w * scale))
            new_h = max(50, int(h * scale))

            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            img_tk = ImageTk.PhotoImage(image=pil_img)
            self.video_lbl.configure(image=img_tk, text="")
            self.current_img_tk = img_tk

        # Update Telemetry Stats
        state = data.get("state")
        decision = data.get("decision", {})

        if state:
            act = decision.get("action", "run_normal").upper()
            self.action_badge.configure(text=act)

            # Colors for action badge
            if act == "JUMP":
                self.action_badge.configure(text_color="#ffffff", fg_color="#ff2a55")
            elif act == "DUCK":
                self.action_badge.configure(text_color="#ffffff", fg_color="#ff9900")
            elif act == "RESTART":
                self.action_badge.configure(text_color="#000000", fg_color="#00ffcc")
            else:
                self.action_badge.configure(text_color="#00ffcc", fg_color="#12131c")

            # Urgency Bar
            urg = float(decision.get("threat_score", state.action_urgency))
            self.urgency_bar.set(urg)
            if urg > 0.7:
                self.urgency_bar.configure(progress_color="#ff2a55")
            elif urg > 0.3:
                self.urgency_bar.configure(progress_color="#ff9900")
            else:
                self.urgency_bar.configure(progress_color="#00ffcc")

            self.urgency_pct.configure(text=f"{int(urg * 100)}% ({'CRITICAL IMPACT' if urg > 0.75 else 'Approaching' if urg > 0.2 else 'Clear'})")

            # Obstacles info
            if state.nearest_obstacle:
                obs = state.nearest_obstacle
                self.obs_lbl.configure(text=f"Obstacle: {obs.obstacle_type.upper()}")
                self.dist_lbl.configure(
                    text=f"Distance: {obs.distance_from_dino}px | Impact: {int(obs.time_to_impact_ms)}ms"
                )
            else:
                self.obs_lbl.configure(text="Obstacle: Horizon Clear")
                self.dist_lbl.configure(text="Distance / Impact: -- px (-- ms)")

            self.speed_lbl.configure(text=f"Game Velocity: {round(state.game_speed_px_sec, 1)} px/sec")

        # Counters & Brain Source
        jumps = data.get("jumps", 0)
        ducks = data.get("ducks", 0)
        self.counters_lbl.configure(
            text=f"Total Maneuvers:\nJumps: {jumps}  |  Ducks: {ducks}"
        )

        source = decision.get("source", "init")
        latency = decision.get("latency_ms", 0.0)
        conf = decision.get("confidence", 0.95)
        self.brain_source_lbl.configure(
            text=f"Mode: {source.upper()} ({latency}ms, {int(conf * 100)}% conf)"
        )


if __name__ == "__main__":
    app = GamePilotHUD()
    app.mainloop()
