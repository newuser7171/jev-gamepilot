"""
Cyber Desktop GUI for Jev-GamePilot.
Built with CustomTkinter to deliver a high-tech gaming HUD with:
1. Native Embedded 60 FPS Dino Arena (play manually or turn on Jev Autopilot with zero window occlusion).
2. External Screen Grabber (tracks Chrome/Edge/Emulators live via mss and OpenCV).
3. TypeSafe Jev System One Brain Telemetry Deck.
"""

import ctypes
import os
import subprocess
import threading
import time
import webbrowser
import cv2
import customtkinter as ctk
import numpy as np
from PIL import Image, ImageTk
from embedded_dino import EmbeddedDinoArena
from pilot_core import PilotCore

# Cyber aesthetic styling
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class GamePilotHUD(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Jev-GamePilot // Autonomous AI Gaming Agent")
        self.geometry("1140x740")
        self.minsize(1020, 680)
        self.configure(fg_color="#0a0b10")

        self.core = PilotCore()
        self.core.set_telemetry_callback(self._on_telemetry_update)

        # State tracking
        self.current_img_tk: ImageTk.PhotoImage | None = None
        self.is_monitoring = False
        self.is_armed = False
        self.active_arena_mode = "native"  # "native" or "external"
        self.countdown_timer = None
        self.countdown_val = 0
        self._active_bound_key = None

        self._build_header()
        self._build_main_layout()

        # Keyboard shortcuts
        self.bind("<Escape>", lambda e: self.emergency_stop())

        # Auto-detect game window on load
        self.after(500, self._auto_detect_game_window)

    def _build_header(self):
        header_frame = ctk.CTkFrame(
            self, fg_color="#12131c", corner_radius=0, height=60
        )
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
        container = ctk.CTkFrame(self, fg_color="transparent")
        container.pack(fill="both", expand=True, padx=15, pady=15)

        # Left Column: Controls & Window Calibration (width: 340)
        left_col = ctk.CTkFrame(
            container, fg_color="#12131c", corner_radius=10, width=340
        )
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
            values=[
                "Chrome Dino / Edge Surf",
                "Subway Surfers (3-Lane)",
                "Screen Chess Bot (FEN)",
            ],
            command=self._on_mode_change,
            fg_color="#1a1c26",
            button_color="#2b2f42",
            button_hover_color="#00ffcc",
            text_color="#ffffff",
            dropdown_fg_color="#1a1c26",
        )
        self.mode_selector.pack(fill="x", padx=pad_x, pady=(0, 8))

        # Launch Offline Dino Browser Button
        launch_dino_btn = ctk.CTkButton(
            parent,
            text="🌐 Open Dino in Browser",
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#202230",
            hover_color="#2e3146",
            text_color="#00ffcc",
            command=self.open_offline_dino,
            height=30,
        )
        launch_dino_btn.pack(fill="x", padx=pad_x, pady=(0, 12))

        # Separator
        ctk.CTkFrame(parent, height=1, fg_color="#1f2230").pack(
            fill="x", padx=pad_x, pady=4
        )

        # 2. Window Calibration & Snapping
        sec2_lbl = ctk.CTkLabel(
            parent,
            text="EXTERNAL SCREEN CAPTURE",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#00ffcc",
        )
        sec2_lbl.pack(anchor="w", padx=pad_x, pady=(8, 4))

        win_row = ctk.CTkFrame(parent, fg_color="transparent")
        win_row.pack(fill="x", padx=pad_x, pady=(0, 5))

        self.window_dropdown = ctk.CTkOptionMenu(
            win_row,
            values=["(Auto-Detect Game Window)"],
            fg_color="#1a1c26",
            button_color="#2b2f42",
            dropdown_fg_color="#1a1c26",
            width=210,
        )
        self.window_dropdown.pack(
            side="left", fill="x", expand=True, padx=(0, 5)
        )

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
            text="🎯 Focus & Snap Window",
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#1a1c26",
            hover_color="#2b2f42",
            text_color="#e0e0e0",
            command=self._snap_to_selected_window,
            height=28,
        )
        snap_btn.pack(fill="x", padx=pad_x, pady=(0, 6))

        self.coords_lbl = ctk.CTkLabel(
            parent,
            text="Region: Auto Center (800x400)",
            font=ctk.CTkFont(size=10),
            text_color="#70758a",
        )
        self.coords_lbl.pack(anchor="w", padx=pad_x, pady=(0, 10))

        # Separator
        ctk.CTkFrame(parent, height=1, fg_color="#1f2230").pack(
            fill="x", padx=pad_x, pady=4
        )

        # 3. Action Buttons & Start Configuration
        sec3_lbl = ctk.CTkLabel(
            parent,
            text="PILOT ENGAGEMENT",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#00ffcc",
        )
        sec3_lbl.pack(anchor="w", padx=pad_x, pady=(8, 4))

        opt_frame = ctk.CTkFrame(parent, fg_color="transparent")
        opt_frame.pack(fill="x", padx=pad_x, pady=(0, 8))

        self.delay_switch = ctk.CTkSwitch(
            opt_frame,
            text="3s Focus Delay",
            font=ctk.CTkFont(size=11),
            text_color="#c0c0c0",
            progress_color="#00ffcc",
        )
        self.delay_switch.deselect()  # Default off for instant native play!
        self.delay_switch.pack(side="left")

        self.hotkey_selector = ctk.CTkOptionMenu(
            opt_frame,
            values=["No Hotkey (Click Only)", "Enter", "F2", "Tab", "F8"],
            font=ctk.CTkFont(size=10),
            width=130,
            height=24,
            fg_color="#1a1c26",
            button_color="#2b2f42",
            dropdown_fg_color="#1a1c26",
            command=self._on_hotkey_change,
        )
        self.hotkey_selector.pack(side="right")

        # Big Green Start Button
        self.arm_btn = ctk.CTkButton(
            parent,
            text="🚀 START AUTOPILOT\n(Jev System One Play)",
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#00d26a",
            hover_color="#00b058",
            text_color="#000000",
            height=50,
            command=lambda: self.toggle_pilot(armed=True),
        )
        self.arm_btn.pack(fill="x", padx=pad_x, pady=(0, 8))

        # Reset Game Button
        self.reset_btn = ctk.CTkButton(
            parent,
            text="🔄 Reset Game Arena",
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#1f2230",
            hover_color="#2e3146",
            text_color="#ffffff",
            height=32,
            command=self._on_reset_game,
        )
        self.reset_btn.pack(fill="x", padx=pad_x, pady=(0, 8))

        # Big Red Emergency Stop Button
        self.stop_btn = ctk.CTkButton(
            parent,
            text="🛑 STOP / HALT [ESC]",
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#ff2a55",
            hover_color="#d61f43",
            text_color="#ffffff",
            height=42,
            command=self.emergency_stop,
        )
        self.stop_btn.pack(fill="x", padx=pad_x, pady=(0, 10))

        # Failsafe Notice
        notice_lbl = ctk.CTkLabel(
            parent,
            text="💡 Tip: Click inside the game arena to play\nmanually anytime (SPACE=Jump, DOWN=Duck)!",
            font=ctk.CTkFont(size=10),
            text_color="#70758a",
            justify="center",
        )
        notice_lbl.pack(fill="x", padx=pad_x, pady=(2, 6))

    def _build_right_viewport(self, parent):
        # 1. Viewport Card
        viewport_card = ctk.CTkFrame(
            parent, fg_color="#12131c", corner_radius=10
        )
        viewport_card.pack(fill="both", expand=True, padx=0, pady=(0, 10))

        # Top Bar of Viewport
        vp_top = ctk.CTkFrame(viewport_card, fg_color="transparent", height=38)
        vp_top.pack(fill="x", padx=15, pady=(10, 5))

        # Segmented Button to switch between Native Arena and External Screen Grabber
        self.arena_tab = ctk.CTkSegmentedButton(
            vp_top,
            values=[
                "🎮 Native Dino Arena (Instant Play)",
                "🖥️ External Screen Grabber (Chrome/Edge)",
            ],
            command=self._on_arena_tab_changed,
            fg_color="#1a1c26",
            selected_color="#00d26a",
            selected_hover_color="#00b058",
            unselected_color="#1a1c26",
            unselected_hover_color="#2b2f42",
            text_color="#ffffff",
            font=ctk.CTkFont(size=11, weight="bold"),
        )
        self.arena_tab.set("🎮 Native Dino Arena (Instant Play)")
        self.arena_tab.pack(side="left")

        self.fps_lbl = ctk.CTkLabel(
            vp_top,
            text="60.0 FPS // Active Arena",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#70758a",
        )
        self.fps_lbl.pack(side="right")

        # Viewport Content Container
        self.content_container = ctk.CTkFrame(
            viewport_card, fg_color="#08090d", corner_radius=6
        )
        self.content_container.pack(
            fill="both", expand=True, padx=15, pady=(0, 15)
        )

        # Tab 1: Embedded Native Dino Arena
        self.embedded_arena = EmbeddedDinoArena(
            self.content_container, width=740, height=270
        )
        self.embedded_arena.pack(fill="both", expand=True, padx=10, pady=10)
        self.embedded_arena.start()

        # Tab 2: External Screen Video Label (initially hidden)
        self.video_lbl = ctk.CTkLabel(
            self.content_container,
            text="[ External Screen Viewport Offline // Click 'Snap Window' then 'Start Autopilot' ]",
            font=ctk.CTkFont(size=13),
            text_color="#53586d",
        )

        # 2. Bottom Jev Brain Telemetry Deck
        telemetry_deck = ctk.CTkFrame(
            parent, fg_color="#12131c", corner_radius=10, height=180
        )
        telemetry_deck.pack(fill="x", padx=0, pady=0)
        telemetry_deck.pack_propagate(False)

        deck_grid = ctk.CTkFrame(telemetry_deck, fg_color="transparent")
        deck_grid.pack(fill="both", expand=True, padx=15, pady=12)

        # Col 1: Action Badge & Urgency Gauge
        col1 = ctk.CTkFrame(
            deck_grid, fg_color="#1a1c26", corner_radius=8, width=220
        )
        col1.pack(side="left", fill="y", padx=(0, 10))
        col1.pack_propagate(False)

        act_title = ctk.CTkLabel(
            col1,
            text="JEV DECISION",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color="#70758a",
        )
        act_title.pack(anchor="w", padx=12, pady=(8, 2))

        self.action_badge = ctk.CTkLabel(
            col1,
            text="RUN NORMAL",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="#00ffcc",
            fg_color="#12131c",
            corner_radius=6,
            height=38,
        )
        self.action_badge.pack(fill="x", padx=12, pady=(0, 6))

        urg_title = ctk.CTkLabel(
            col1,
            text="THREAT URGENCY",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color="#70758a",
        )
        urg_title.pack(anchor="w", padx=12, pady=(0, 2))

        self.urgency_bar = ctk.CTkProgressBar(
            col1, height=10, fg_color="#12131c", progress_color="#00ffcc"
        )
        self.urgency_bar.set(0.0)
        self.urgency_bar.pack(fill="x", padx=12, pady=(0, 4))

        self.urgency_pct = ctk.CTkLabel(
            col1,
            text="0% (Horizon Clear)",
            font=ctk.CTkFont(size=10),
            text_color="#70758a",
        )
        self.urgency_pct.pack(anchor="w", padx=12)

        # Col 2: Tactical Telemetry Grid
        col2 = ctk.CTkFrame(deck_grid, fg_color="#1a1c26", corner_radius=8)
        col2.pack(side="left", fill="both", expand=True, padx=(0, 10))

        grid_inner = ctk.CTkFrame(col2, fg_color="transparent")
        grid_inner.pack(fill="both", expand=True, padx=12, pady=10)

        self.score_lbl = ctk.CTkLabel(
            grid_inner,
            text="Score: 00000 | High: 00000",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#00ffcc",
        )
        self.score_lbl.pack(anchor="w", pady=1)

        self.obs_lbl = ctk.CTkLabel(
            grid_inner,
            text="Obstacle: None Detected",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#e0e0e0",
        )
        self.obs_lbl.pack(anchor="w", pady=1)

        self.dist_lbl = ctk.CTkLabel(
            grid_inner,
            text="Distance / Impact: -- px (-- ms)",
            font=ctk.CTkFont(size=11),
            text_color="#70758a",
        )
        self.dist_lbl.pack(anchor="w", pady=1)

        self.speed_lbl = ctk.CTkLabel(
            grid_inner,
            text="Game Velocity: 360.0 px/sec",
            font=ctk.CTkFont(size=11),
            text_color="#70758a",
        )
        self.speed_lbl.pack(anchor="w", pady=1)

        # Col 3: Jev Brain Reasoning & Counters
        col3 = ctk.CTkFrame(
            deck_grid, fg_color="#1a1c26", corner_radius=8, width=230
        )
        col3.pack(side="right", fill="y", padx=0)
        col3.pack_propagate(False)

        c3_title = ctk.CTkLabel(
            col3,
            text="JEV SYSTEM ONE INTUITION",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color="#70758a",
        )
        c3_title.pack(anchor="w", padx=12, pady=(8, 2))

        self.brain_source_lbl = ctk.CTkLabel(
            col3,
            text="Model: jev-latest (System One)",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#00ffcc",
        )
        self.brain_source_lbl.pack(anchor="w", padx=12, pady=1)

        self.counters_lbl = ctk.CTkLabel(
            col3,
            text="Total Maneuvers:\nJumps: 0  |  Ducks: 0",
            font=ctk.CTkFont(size=11),
            text_color="#e0e0e0",
            justify="left",
        )
        self.counters_lbl.pack(anchor="w", padx=12, pady=2)

        self.fast_fall_lbl = ctk.CTkLabel(
            col3,
            text="Fast-Fall Ready: YES",
            font=ctk.CTkFont(size=10),
            text_color="#00d26a",
        )
        self.fast_fall_lbl.pack(anchor="w", padx=12)

    def _on_arena_tab_changed(self, choice: str):
        if "Native" in choice:
            self.active_arena_mode = "native"
            self.video_lbl.pack_forget()
            self.embedded_arena.pack(fill="both", expand=True, padx=10, pady=10)
            self.fps_lbl.configure(text="60.0 FPS // Native Arena")
        else:
            self.active_arena_mode = "external"
            self.embedded_arena.pack_forget()
            self.video_lbl.pack(fill="both", expand=True)
            self.fps_lbl.configure(text="0.0 FPS // External Capture")

    def _on_reset_game(self):
        self.embedded_arena.reset_game()

    def _on_hotkey_change(self, choice: str):
        if self._active_bound_key:
            try:
                self.unbind(self._active_bound_key)
            except Exception:
                pass
            self._active_bound_key = None

        key_map = {
            "Enter": "<Return>",
            "F2": "<F2>",
            "Tab": "<Tab>",
            "F8": "<F8>",
        }
        if choice in key_map:
            seq = key_map[choice]
            self.bind(seq, lambda e: self.toggle_pilot(armed=True))
            self._active_bound_key = seq

    def _refresh_windows(self):
        titles = self.core.vision.list_windows()
        filtered = [
            t
            for t in titles
            if any(
                k in t.lower()
                for k in [
                    "dino",
                    "edge",
                    "chrome",
                    "surf",
                    "subway",
                    "bluestack",
                ]
            )
        ]
        if not filtered:
            filtered = titles[:8] if titles else ["No active windows found"]

        self.window_dropdown.configure(values=filtered)
        if filtered:
            self.window_dropdown.set(filtered[0])

    def _auto_detect_game_window(self):
        self._refresh_windows()
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
                # Automatically switch tab to External Screen Grabber
                self.arena_tab.set("🖥️ External Screen Grabber (Chrome/Edge)")
                self._on_arena_tab_changed("External")

                # Bring target window to foreground
                try:
                    import pygetwindow as gw

                    wins = [
                        w
                        for w in gw.getAllWindows()
                        if selected.lower() in w.title.lower()
                    ]
                    if wins:
                        w = wins[0]
                        ctypes.windll.user32.ShowWindow(w._hWnd, 9)  # SW_RESTORE
                        ctypes.windll.user32.SetForegroundWindow(w._hWnd)
                except Exception:
                    pass
            else:
                self.coords_lbl.configure(
                    text="Could not lock to window dimensions."
                )

    def _on_mode_change(self, mode: str):
        if "Dino" in mode:
            self.core.current_game_mode = "dino"
        elif "Subway" in mode:
            self.core.current_game_mode = "runner"

    def open_offline_dino(self):
        """Launches the built-in offline Dino HTML game in browser."""
        dino_path = os.path.abspath("dino_game.html")
        if os.path.exists(dino_path):
            webbrowser.open(f"file:///{dino_path}")
            self.after(1500, self._auto_detect_game_window)

    def toggle_pilot(self, armed: bool):
        if self.countdown_timer:
            self.after_cancel(self.countdown_timer)
            self.countdown_timer = None
            self._update_status_idle()
            return

        if self.active_arena_mode == "native":
            # Native Embedded Arena Mode
            if self.embedded_arena.autopilot_enabled:
                self.embedded_arena.set_autopilot(False)
                self._update_status_idle()
            else:
                if self.delay_switch.get() == 1:
                    self._start_countdown(3, mode="native")
                else:
                    self._engage_native()
        else:
            # External Screen Grabber Mode
            if self.core.is_running:
                self.core.stop()
                self._update_status_idle()
            else:
                if self.delay_switch.get() == 1:
                    self._start_countdown(3, mode="external")
                else:
                    self._engage_external()

    def _start_countdown(self, seconds_left: int, mode: str):
        if seconds_left > 0:
            self.arm_btn.configure(
                text=f"⏳ Starting in {seconds_left}s...\nGet ready!",
                fg_color="#ff9900",
            )
            self.status_badge.configure(
                text=f"● COUNTDOWN {seconds_left}s",
                text_color="#000000",
                fg_color="#ff9900",
            )
            self.countdown_timer = self.after(
                1000, lambda: self._start_countdown(seconds_left - 1, mode)
            )
        else:
            self.countdown_timer = None
            if mode == "native":
                self._engage_native()
            else:
                self._engage_external()

    def _engage_native(self):
        self.embedded_arena.set_autopilot(
            True, self.core.brain, self._on_embedded_telemetry
        )
        self.status_badge.configure(
            text="● JEV AUTOPILOT PLAYING",
            text_color="#000000",
            fg_color="#00d26a",
        )
        self.arm_btn.configure(
            text="🛑 STOP AUTOPILOT\n(Click or press ESC)", fg_color="#ff2a55"
        )

    def _engage_external(self):
        self.core.start(arm_inputs=True)
        self.status_badge.configure(
            text="● EXTERNAL PILOT ARMED",
            text_color="#000000",
            fg_color="#00d26a",
        )
        self.arm_btn.configure(
            text="🛑 STOP EXTERNAL PILOT\n(Click or press ESC)",
            fg_color="#ff2a55",
        )

    def emergency_stop(self):
        if self.countdown_timer:
            self.after_cancel(self.countdown_timer)
            self.countdown_timer = None

        self.embedded_arena.set_autopilot(False)
        self.core.stop()
        self._update_status_idle()

    def _update_status_idle(self):
        if self.countdown_timer:
            self.after_cancel(self.countdown_timer)
            self.countdown_timer = None
        self.status_badge.configure(
            text="● SYSTEM STANDBY",
            text_color="#70758a",
            fg_color="#1a1c26",
        )
        self.arm_btn.configure(
            text="🚀 START AUTOPILOT\n(Jev System One Play)",
            fg_color="#00d26a",
        )
        self.action_badge.configure(
            text="STANDBY", text_color="#70758a", fg_color="#12131c"
        )
        self.urgency_bar.set(0.0)

    def _on_embedded_telemetry(self, data: dict):
        """Called directly from EmbeddedDinoArena loop."""
        state = data.get("state")
        decision = data.get("decision", {})
        score = data.get("score", 0)
        hi_score = data.get("hi_score", 0)

        self.score_lbl.configure(
            text=f"Score: {str(score).zfill(5)} | High: {str(hi_score).zfill(5)}"
        )

        act = decision.get("action", "run_normal").upper()
        self.action_badge.configure(text=act)

        if act == "JUMP":
            self.action_badge.configure(
                text_color="#ffffff", fg_color="#ff2a55"
            )
        elif act == "DUCK":
            self.action_badge.configure(
                text_color="#ffffff", fg_color="#ff9900"
            )
        elif act == "RESTART":
            self.action_badge.configure(
                text_color="#000000", fg_color="#00ffcc"
            )
        else:
            self.action_badge.configure(
                text_color="#00ffcc", fg_color="#12131c"
            )

        urg = float(decision.get("threat_score", 0.0))
        self.urgency_bar.set(urg)
        if urg > 0.7:
            self.urgency_bar.configure(progress_color="#ff2a55")
        elif urg > 0.3:
            self.urgency_bar.configure(progress_color="#ff9900")
        else:
            self.urgency_bar.configure(progress_color="#00ffcc")

        self.urgency_pct.configure(
            text=f"{int(urg * 100)}% ({'CRITICAL THREAT' if urg > 0.75 else 'Approaching' if urg > 0.2 else 'Clear'})"
        )

        if state and state.nearest_obstacle:
            obs = state.nearest_obstacle
            self.obs_lbl.configure(text=f"Obstacle: {obs.obstacle_type.upper()}")
            self.dist_lbl.configure(
                text=f"Distance: {obs.distance_from_dino}px | Impact: {int(obs.time_to_impact_ms)}ms"
            )
        else:
            self.obs_lbl.configure(text="Obstacle: Horizon Clear")
            self.dist_lbl.configure(text="Distance / Impact: -- px (-- ms)")

        if state:
            self.speed_lbl.configure(
                text=f"Game Velocity: {round(state.game_speed_px_sec, 1)} px/sec"
            )

        jumps = data.get("jumps", 0)
        ducks = data.get("ducks", 0)
        self.counters_lbl.configure(
            text=f"Total Maneuvers:\nJumps: {jumps}  |  Ducks: {ducks}"
        )

        self.brain_source_lbl.configure(
            text="Model: JEV-LATEST (Reflex Active)"
        )

    def _on_telemetry_update(self, data: dict):
        """Called from PilotCore loop for external screen capture."""
        self.after(0, self._render_telemetry_ui, data)

    def _render_telemetry_ui(self, data: dict):
        self.fps_lbl.configure(
            text=f"{data['fps']} FPS // External Screen Capture"
        )

        frame = data.get("frame")
        if frame is not None and self.active_arena_mode == "external":
            container_w = max(400, self.content_container.winfo_width())
            container_h = max(200, self.content_container.winfo_height())

            h, w, _ = frame.shape
            scale = min(container_w / w, container_h / h)
            new_w = max(50, int(w * scale))
            new_h = max(50, int(h * scale))

            resized = cv2.resize(
                frame, (new_w, new_h), interpolation=cv2.INTER_AREA
            )
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            img_tk = ImageTk.PhotoImage(image=pil_img)
            self.video_lbl.configure(image=img_tk, text="")
            self.current_img_tk = img_tk

        state = data.get("state")
        decision = data.get("decision", {})

        if state:
            act = decision.get("action", "run_normal").upper()
            self.action_badge.configure(text=act)

            if act == "JUMP":
                self.action_badge.configure(
                    text_color="#ffffff", fg_color="#ff2a55"
                )
            elif act == "DUCK":
                self.action_badge.configure(
                    text_color="#ffffff", fg_color="#ff9900"
                )
            elif act == "RESTART":
                self.action_badge.configure(
                    text_color="#000000", fg_color="#00ffcc"
                )
            else:
                self.action_badge.configure(
                    text_color="#00ffcc", fg_color="#12131c"
                )

            urg = float(decision.get("threat_score", state.action_urgency))
            self.urgency_bar.set(urg)
            if urg > 0.7:
                self.urgency_bar.configure(progress_color="#ff2a55")
            elif urg > 0.3:
                self.urgency_bar.configure(progress_color="#ff9900")
            else:
                self.urgency_bar.configure(progress_color="#00ffcc")

            self.urgency_pct.configure(
                text=f"{int(urg * 100)}% ({'CRITICAL THREAT' if urg > 0.75 else 'Approaching' if urg > 0.2 else 'Clear'})"
            )

            if state.nearest_obstacle:
                obs = state.nearest_obstacle
                self.obs_lbl.configure(
                    text=f"Obstacle: {obs.obstacle_type.upper()}"
                )
                self.dist_lbl.configure(
                    text=f"Distance: {obs.distance_from_dino}px | Impact: {int(obs.time_to_impact_ms)}ms"
                )
            else:
                self.obs_lbl.configure(text="Obstacle: Horizon Clear")
                self.dist_lbl.configure(text="Distance / Impact: -- px (-- ms)")

            self.speed_lbl.configure(
                text=f"Game Velocity: {round(state.game_speed_px_sec, 1)} px/sec"
            )

        jumps = data.get("jumps", 0)
        ducks = data.get("ducks", 0)
        self.counters_lbl.configure(
            text=f"Total Maneuvers:\nJumps: {jumps}  |  Ducks: {ducks}"
        )


if __name__ == "__main__":
    app = GamePilotHUD()
    app.mainloop()
