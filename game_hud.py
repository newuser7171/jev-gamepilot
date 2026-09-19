"""
Universal Cyber Desktop GUI for Jev-GamePilot.
Built with CustomTkinter to deliver a universal autonomous gaming AI:
1. Dynamic Game Profile System (Dino Runner, Subway Surfers, Flappy Bird, Aim Trainer, 2D Platformer, + Custom).
2. Native 60 FPS Dino Arena + Universal External Screen Grabber for ANY game.
3. Player Avatar Calibration & Threat Vector Tracking.
4. Floating Mini-Bar Mode for playing games full-screen.
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
from custom_profile_dialog import CustomProfileDialog
from embedded_dino import EmbeddedDinoArena
from pilot_core import PilotCore
from profile_manager import GameAction, GameProfile

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class GamePilotHUD(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Jev-GamePilot // Universal Autonomous Gaming AI")
        self.geometry("1160x760")
        self.minsize(1040, 700)
        self.configure(fg_color="#0a0b10")

        self.core = PilotCore()
        self.core.set_telemetry_callback(self._on_telemetry_update)

        # State tracking
        self.current_img_tk: ImageTk.PhotoImage | None = None
        self.is_armed = False
        self.active_arena_mode = "native"  # "native" or "external"
        self.countdown_timer = None
        self._active_bound_key = None
        self.is_mini_mode = False

        self._build_header()
        self._build_main_layout()
        self._build_mini_layout()

        # Keyboard shortcuts
        self.bind("<Escape>", lambda e: self.emergency_stop())

        # Auto-detect game window on start
        self.after(500, self._auto_detect_game_window)

    def _build_header(self):
        self.header_frame = ctk.CTkFrame(
            self, fg_color="#12131c", corner_radius=0, height=60
        )
        self.header_frame.pack(fill="x", side="top", padx=0, pady=0)

        # Title & Subtitle
        title_box = ctk.CTkFrame(self.header_frame, fg_color="transparent")
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
            text=" // UNIVERSAL GAMING AI",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#70758a",
        )
        sub_lbl.pack(side="left", padx=5)

        # Right side: Mini Mode Toggle & Status Badge
        r_box = ctk.CTkFrame(self.header_frame, fg_color="transparent")
        r_box.pack(side="right", padx=15)

        self.mini_mode_btn = ctk.CTkButton(
            r_box,
            text="🪟 Mini Floating HUD",
            font=ctk.CTkFont(size=11, weight="bold"),
            width=140,
            height=28,
            fg_color="#1a1c26",
            hover_color="#2b2f42",
            text_color="#00ffcc",
            command=self._toggle_mini_mode,
        )
        self.mini_mode_btn.pack(side="left", padx=(0, 10))

        self.status_badge = ctk.CTkLabel(
            r_box,
            text="● SYSTEM STANDBY",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#70758a",
            fg_color="#1a1c26",
            corner_radius=12,
            padx=15,
            pady=4,
        )
        self.status_badge.pack(side="left")

    def _build_main_layout(self):
        self.main_container = ctk.CTkFrame(self, fg_color="transparent")
        self.main_container.pack(fill="both", expand=True, padx=15, pady=15)

        # Left Column: Profile Selector, Actions, Calibration, Start
        left_col = ctk.CTkFrame(
            self.main_container, fg_color="#12131c", corner_radius=10, width=350
        )
        left_col.pack(side="left", fill="y", padx=(0, 10), pady=0)
        left_col.pack_propagate(False)

        # Right Column: Viewport & Jev Brain Telemetry
        right_col = ctk.CTkFrame(self.main_container, fg_color="transparent")
        right_col.pack(side="right", fill="both", expand=True, padx=0, pady=0)

        self._build_left_controls(left_col)
        self._build_right_viewport(right_col)

    def _build_left_controls(self, parent):
        pad_x = 15

        # 1. Universal Game Profile Selector
        sec1_lbl = ctk.CTkLabel(
            parent,
            text="GAME PROFILE",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#00ffcc",
        )
        sec1_lbl.pack(anchor="w", padx=pad_x, pady=(12, 4))

        self.profile_selector = ctk.CTkOptionMenu(
            parent,
            values=self._get_profile_dropdown_values(),
            command=self._on_profile_selected,
            fg_color="#1a1c26",
            button_color="#2b2f42",
            button_hover_color="#00ffcc",
            text_color="#ffffff",
            dropdown_fg_color="#1a1c26",
        )
        self.profile_selector.pack(fill="x", padx=pad_x, pady=(0, 6))

        # Action Keys Preview Badge
        self.actions_preview_lbl = ctk.CTkLabel(
            parent,
            text=self._format_actions_preview(self.core.current_profile),
            font=ctk.CTkFont(size=10),
            text_color="#70758a",
            justify="left",
            wraplength=310,
        )
        self.actions_preview_lbl.pack(anchor="w", padx=pad_x, pady=(0, 8))

        # + Add Custom Profile Button
        new_prof_btn = ctk.CTkButton(
            parent,
            text="➕ Create Custom Game Profile...",
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color="#1a1c28",
            hover_color="#282c40",
            text_color="#00ffcc",
            command=self._open_new_profile_dialog,
            height=28,
        )
        new_prof_btn.pack(fill="x", padx=pad_x, pady=(0, 10))

        # Separator
        ctk.CTkFrame(parent, height=1, fg_color="#1f2230").pack(
            fill="x", padx=pad_x, pady=3
        )

        # 2. Player Avatar Calibration & Screen Snapping
        sec2_lbl = ctk.CTkLabel(
            parent,
            text="SCREEN & AVATAR CALIBRATION",
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
            width=220,
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

        # Player Avatar Calibrator Row
        calib_row = ctk.CTkFrame(parent, fg_color="transparent")
        calib_row.pack(fill="x", padx=pad_x, pady=(0, 6))

        self.calib_btn = ctk.CTkButton(
            calib_row,
            text="📌 Calibrate Player Avatar",
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color="#1f2232",
            hover_color="#2c3044",
            text_color="#00d26a",
            height=26,
            command=self._calibrate_player_avatar,
        )
        self.calib_btn.pack(side="left", fill="x", expand=True, padx=(0, 4))

        clear_calib_btn = ctk.CTkButton(
            calib_row,
            text="✕",
            width=26,
            height=26,
            fg_color="#1f2232",
            hover_color="#ff2a55",
            text_color="#ff2a55",
            command=self._clear_player_calibration,
        )
        clear_calib_btn.pack(side="right")

        self.coords_lbl = ctk.CTkLabel(
            parent,
            text="Region: Auto Center (800x400)",
            font=ctk.CTkFont(size=10),
            text_color="#70758a",
        )
        self.coords_lbl.pack(anchor="w", padx=pad_x, pady=(0, 8))

        # Separator
        ctk.CTkFrame(parent, height=1, fg_color="#1f2230").pack(
            fill="x", padx=pad_x, pady=3
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
        opt_frame.pack(fill="x", padx=pad_x, pady=(0, 6))

        self.delay_switch = ctk.CTkSwitch(
            opt_frame,
            text="3s Focus Delay",
            font=ctk.CTkFont(size=11),
            text_color="#c0c0c0",
            progress_color="#00ffcc",
        )
        self.delay_switch.deselect()
        self.delay_switch.pack(side="left")

        self.hotkey_selector = ctk.CTkOptionMenu(
            opt_frame,
            values=["No Hotkey (Click Only)", "Enter", "F2", "Tab", "F8"],
            font=ctk.CTkFont(size=10),
            width=135,
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
            height=48,
            command=lambda: self.toggle_pilot(armed=True),
        )
        self.arm_btn.pack(fill="x", padx=pad_x, pady=(0, 6))

        # Reset / Replay Button
        self.reset_btn = ctk.CTkButton(
            parent,
            text="🔄 Reset Game Arena",
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color="#1f2230",
            hover_color="#2e3146",
            text_color="#ffffff",
            height=28,
            command=self._on_reset_game,
        )
        self.reset_btn.pack(fill="x", padx=pad_x, pady=(0, 6))

        # Big Red Emergency Stop Button
        self.stop_btn = ctk.CTkButton(
            parent,
            text="🛑 STOP / HALT [ESC]",
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#ff2a55",
            hover_color="#d61f43",
            text_color="#ffffff",
            height=38,
            command=self.emergency_stop,
        )
        self.stop_btn.pack(fill="x", padx=pad_x, pady=(0, 8))

        notice_lbl = ctk.CTkLabel(
            parent,
            text="💡 Tip: Works on ANY game! Switch profiles or\ncreate custom ones for Roblox, Subway, etc.",
            font=ctk.CTkFont(size=10),
            text_color="#70758a",
            justify="center",
        )
        notice_lbl.pack(fill="x", padx=pad_x, pady=(2, 4))

    def _build_right_viewport(self, parent):
        # 1. Viewport Card
        viewport_card = ctk.CTkFrame(
            parent, fg_color="#12131c", corner_radius=10
        )
        viewport_card.pack(fill="both", expand=True, padx=0, pady=(0, 10))

        vp_top = ctk.CTkFrame(viewport_card, fg_color="transparent", height=38)
        vp_top.pack(fill="x", padx=15, pady=(10, 5))

        self.arena_tab = ctk.CTkSegmentedButton(
            vp_top,
            values=[
                "🎮 Native Dino Arena (Instant Play)",
                "🖥️ Universal Screen Grabber (Any Game)",
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

        self.content_container = ctk.CTkFrame(
            viewport_card, fg_color="#08090d", corner_radius=6
        )
        self.content_container.pack(
            fill="both", expand=True, padx=15, pady=(0, 15)
        )

        # Native Dino Arena Canvas
        self.embedded_arena = EmbeddedDinoArena(
            self.content_container, width=740, height=270
        )
        self.embedded_arena.pack(fill="both", expand=True, padx=10, pady=10)
        self.embedded_arena.start()

        # External Universal Video Label (for ANY game)
        self.video_lbl = ctk.CTkLabel(
            self.content_container,
            text="[ Universal Screen Grabber Offline // Click 'Focus & Snap Window' then 'Start Autopilot' ]",
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
            text="WAIT / STANDBY",
            font=ctk.CTkFont(size=17, weight="bold"),
            text_color="#00ffcc",
            fg_color="#12131c",
            corner_radius=6,
            height=38,
        )
        self.action_badge.pack(fill="x", padx=12, pady=(0, 6))

        urg_title = ctk.CTkLabel(
            col1,
            text="THREAT / URGENCY GAUGE",
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
            text="Profile: 🦖 Chrome Dino",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#00ffcc",
        )
        self.score_lbl.pack(anchor="w", pady=1)

        self.obs_lbl = ctk.CTkLabel(
            grid_inner,
            text="Threat / Target: None Detected",
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
            text="Tracking: Auto Contour / Template",
            font=ctk.CTkFont(size=11),
            text_color="#70758a",
        )
        self.speed_lbl.pack(anchor="w", pady=1)

        # Col 3: Jev Brain Intuition & Counters
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
            text="Model: JEV-LATEST",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#00ffcc",
        )
        self.brain_source_lbl.pack(anchor="w", padx=12, pady=1)

        self.counters_lbl = ctk.CTkLabel(
            col3,
            text="Total Maneuvers:\nActions: 0",
            font=ctk.CTkFont(size=11),
            text_color="#e0e0e0",
            justify="left",
        )
        self.counters_lbl.pack(anchor="w", padx=12, pady=2)

        self.fast_fall_lbl = ctk.CTkLabel(
            col3,
            text="Universal Schema: ACTIVE",
            font=ctk.CTkFont(size=10),
            text_color="#00d26a",
        )
        self.fast_fall_lbl.pack(anchor="w", padx=12)

    def _build_mini_layout(self):
        """Compact floating bar layout for playing full screen games."""
        self.mini_container = ctk.CTkFrame(self, fg_color="#0e1017")

        row = ctk.CTkFrame(self.mini_container, fg_color="transparent")
        row.pack(fill="both", expand=True, padx=10, pady=8)

        # Profile & Action
        info_col = ctk.CTkFrame(row, fg_color="transparent")
        info_col.pack(side="left", fill="both", expand=True)

        self.mini_profile_lbl = ctk.CTkLabel(
            info_col,
            text="⚡ JEV-GAMEPILOT",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#00ffcc",
        )
        self.mini_profile_lbl.pack(anchor="w")

        self.mini_action_badge = ctk.CTkLabel(
            info_col,
            text="STANDBY",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#ffffff",
        )
        self.mini_action_badge.pack(anchor="w")

        self.mini_urgency_bar = ctk.CTkProgressBar(
            info_col, height=6, fg_color="#1a1c26", progress_color="#00ffcc"
        )
        self.mini_urgency_bar.set(0.0)
        self.mini_urgency_bar.pack(fill="x", pady=(2, 0))

        # Controls
        ctrl_col = ctk.CTkFrame(row, fg_color="transparent")
        ctrl_col.pack(side="right", padx=(10, 0))

        self.mini_arm_btn = ctk.CTkButton(
            ctrl_col,
            text="▶ START",
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color="#00d26a",
            hover_color="#00b058",
            text_color="#000000",
            width=70,
            height=30,
            command=lambda: self.toggle_pilot(armed=True),
        )
        self.mini_arm_btn.pack(side="top", pady=(0, 4))

        expand_btn = ctk.CTkButton(
            ctrl_col,
            text="🔲 Expand",
            font=ctk.CTkFont(size=10),
            fg_color="#1a1c26",
            hover_color="#2b2f42",
            width=70,
            height=24,
            command=self._toggle_mini_mode,
        )
        expand_btn.pack(side="top")

    def _toggle_mini_mode(self):
        if not self.is_mini_mode:
            # Switch to Mini Floating Bar
            self.is_mini_mode = True
            self.main_container.pack_forget()
            self.header_frame.pack_forget()
            self.mini_container.pack(fill="both", expand=True)
            self.geometry("380x105")
            self.attributes("-topmost", True)
        else:
            # Switch to Full HUD
            self.is_mini_mode = False
            self.mini_container.pack_forget()
            self.header_frame.pack(fill="x", side="top")
            self.main_container.pack(fill="both", expand=True, padx=15, pady=15)
            self.geometry("1160x760")
            self.attributes("-topmost", False)

    def _get_profile_dropdown_values(self) -> list[str]:
        names = [p.name for p in self.core.profile_mgr.list_profiles()]
        names.append("➕ Create Custom Game Profile...")
        return names

    def _format_actions_preview(self, profile: GameProfile) -> str:
        keys_summary = " | ".join(
            [f"{a.name}: {a.key.upper()}" for a in profile.actions[:4]]
        )
        return f"Control Map: {keys_summary}"

    def _on_profile_selected(self, choice: str):
        if "Create Custom" in choice:
            self._open_new_profile_dialog()
            return

        # Find matching profile
        for prof in self.core.profile_mgr.list_profiles():
            if prof.name == choice:
                self.core.set_profile(prof.id)
                self.actions_preview_lbl.configure(
                    text=self._format_actions_preview(prof)
                )
                self.score_lbl.configure(text=f"Profile: {prof.name}")

                # If non-dino profile, switch automatically to Universal External Grabber
                if prof.id != "runner_dino":
                    self.arena_tab.set(
                        "🖥️ Universal Screen Grabber (Any Game)"
                    )
                    self._on_arena_tab_changed("Universal")
                    if prof.default_window_keyword:
                        self.core.vision.snap_to_window(
                            prof.default_window_keyword
                        )
                break

    def _open_new_profile_dialog(self):
        CustomProfileDialog(
            self,
            self.core.profile_mgr,
            on_created_cb=self._on_custom_profile_created,
        )

    def _on_custom_profile_created(self, new_profile: GameProfile):
        # Refresh dropdown values
        self.profile_selector.configure(values=self._get_profile_dropdown_values())
        self.profile_selector.set(new_profile.name)
        self._on_profile_selected(new_profile.name)

    def _calibrate_player_avatar(self):
        """Grabs the current focal avatar to lock template tracking."""
        frame = self.core.vision.last_frame
        if frame is not None:
            h, w, _ = frame.shape
            # Take candidate player region from left-center
            crop = frame[
                int(h * 0.4) : int(h * 0.75), int(w * 0.1) : int(w * 0.35)
            ]
            self.core.set_player_template(crop)
            self.calib_btn.configure(
                text="✓ Avatar Calibrated", fg_color="#00b058"
            )
            self.speed_lbl.configure(text="Tracking: Locked Template (60 FPS)")
        else:
            self.calib_btn.configure(text="⚠️ Capture Frame First")

    def _clear_player_calibration(self):
        self.core.clear_player_template()
        self.calib_btn.configure(
            text="📌 Calibrate Player Avatar", fg_color="#1f2232"
        )
        self.speed_lbl.configure(text="Tracking: Auto Contour Dynamics")

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
            self.fps_lbl.configure(text="0.0 FPS // Universal Capture")

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
                    "roblox",
                    "minecraft",
                    "steam",
                    "osu",
                    "aim",
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
        for keyword in ["dino", "t-rex", "edge", "chrome", "subway", "bluestack"]:
            if self.core.vision.snap_to_window(keyword):
                reg = self.core.vision.region
                self.coords_lbl.configure(
                    text=f"Snapped: '{keyword}' ({reg['width']}x{reg['height']})"
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
                # Switch tab to Universal Screen Grabber
                self.arena_tab.set("🖥️ Universal Screen Grabber (Any Game)")
                self._on_arena_tab_changed("Universal")

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
                        ctypes.windll.user32.ShowWindow(w._hWnd, 9)
                        ctypes.windll.user32.SetForegroundWindow(w._hWnd)
                except Exception:
                    pass
            else:
                self.coords_lbl.configure(
                    text="Could not lock to window dimensions."
                )

    def toggle_pilot(self, armed: bool):
        if self.countdown_timer:
            self.after_cancel(self.countdown_timer)
            self.countdown_timer = None
            self._update_status_idle()
            return

        if self.active_arena_mode == "native":
            if self.embedded_arena.autopilot_enabled:
                self.embedded_arena.set_autopilot(False)
                self._update_status_idle()
            else:
                if self.delay_switch.get() == 1:
                    self._start_countdown(3, mode="native")
                else:
                    self._engage_native()
        else:
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
            txt = f"⏳ Starting in {seconds_left}s...\nGet ready!"
            self.arm_btn.configure(text=txt, fg_color="#ff9900")
            self.mini_arm_btn.configure(text=f"⏳ {seconds_left}s", fg_color="#ff9900")
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
            text="● NATIVE AUTOPILOT ACTIVE",
            text_color="#000000",
            fg_color="#00d26a",
        )
        self.arm_btn.configure(
            text="🛑 STOP AUTOPILOT\n(Click or press ESC)", fg_color="#ff2a55"
        )
        self.mini_arm_btn.configure(text="🛑 STOP", fg_color="#ff2a55")

    def _engage_external(self):
        self.core.start(arm_inputs=True)
        self.status_badge.configure(
            text=f"● AUTOPILOT: {self.core.current_profile.name[:18]}",
            text_color="#000000",
            fg_color="#00d26a",
        )
        self.arm_btn.configure(
            text="🛑 STOP AUTOPILOT\n(Click or press ESC)", fg_color="#ff2a55"
        )
        self.mini_arm_btn.configure(text="🛑 STOP", fg_color="#ff2a55")

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
        self.mini_arm_btn.configure(text="▶ START", fg_color="#00d26a")
        self.action_badge.configure(
            text="STANDBY", text_color="#70758a", fg_color="#12131c"
        )
        self.mini_action_badge.configure(text="STANDBY")
        self.urgency_bar.set(0.0)
        self.mini_urgency_bar.set(0.0)

    def _on_embedded_telemetry(self, data: dict):
        """Telemetry callback from embedded Dino arena."""
        state = data.get("state")
        decision = data.get("decision", {})
        score = data.get("score", 0)
        hi_score = data.get("hi_score", 0)

        self.score_lbl.configure(
            text=f"Score: {str(score).zfill(5)} | High: {str(hi_score).zfill(5)}"
        )

        act = decision.get("action", "run_normal").upper()
        self.action_badge.configure(text=act)
        self.mini_action_badge.configure(text=act)

        if act == "JUMP":
            self.action_badge.configure(text_color="#ffffff", fg_color="#ff2a55")
        elif act == "DUCK":
            self.action_badge.configure(text_color="#ffffff", fg_color="#ff9900")
        else:
            self.action_badge.configure(text_color="#00ffcc", fg_color="#12131c")

        urg = float(decision.get("threat_score", 0.0))
        self.urgency_bar.set(urg)
        self.mini_urgency_bar.set(urg)

        self.urgency_pct.configure(
            text=f"{int(urg * 100)}% ({'CRITICAL THREAT' if urg > 0.75 else 'Approaching' if urg > 0.2 else 'Clear'})"
        )

        if state and state.nearest_obstacle:
            obs = state.nearest_obstacle
            self.obs_lbl.configure(text=f"Threat: {obs.obstacle_type.upper()}")
            self.dist_lbl.configure(
                text=f"Distance: {obs.distance_from_dino}px | Impact: {int(obs.time_to_impact_ms)}ms"
            )
        else:
            self.obs_lbl.configure(text="Obstacle: Horizon Clear")
            self.dist_lbl.configure(text="Distance / Impact: -- px (-- ms)")

        jumps = data.get("jumps", 0)
        ducks = data.get("ducks", 0)
        self.counters_lbl.configure(
            text=f"Total Maneuvers:\nJumps: {jumps}  |  Ducks: {ducks}"
        )

    def _on_telemetry_update(self, data: dict):
        """Telemetry callback from universal pilot core."""
        self.after(0, self._render_telemetry_ui, data)

    def _render_telemetry_ui(self, data: dict):
        self.fps_lbl.configure(
            text=f"{data['fps']} FPS // Universal Vision Active"
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
        prof = data.get("profile", self.core.current_profile)

        act = decision.get("action", "wait").upper()
        self.action_badge.configure(text=act)
        self.mini_action_badge.configure(text=act)

        urg = float(decision.get("threat_score", 0.0))
        self.urgency_bar.set(urg)
        self.mini_urgency_bar.set(urg)
        self.urgency_pct.configure(text=f"{int(urg * 100)}% Threat Urgency")

        # Telemetry fields for UniversalSceneState
        if hasattr(state, "threats") and state.threats:
            t = state.threats[0]
            self.obs_lbl.configure(text=f"Threat: {int(t.distance_to_player)}px away")
            self.dist_lbl.configure(
                text=f"Threat Box: ({t.x}, {t.y}) {t.w}x{t.h}px"
            )
        elif hasattr(state, "targets") and state.targets:
            tgt = state.targets[0]
            self.obs_lbl.configure(text=f"Target: ({tgt.click_x}, {tgt.click_y})")
            self.dist_lbl.configure(text="Aim Lock: Active")
        elif hasattr(state, "nearest_obstacle") and state.nearest_obstacle:
            obs = state.nearest_obstacle
            self.obs_lbl.configure(text=f"Obstacle: {obs.obstacle_type.upper()}")
            self.dist_lbl.configure(
                text=f"Distance: {obs.distance_from_dino}px | Impact: {int(obs.time_to_impact_ms)}ms"
            )
        else:
            self.obs_lbl.configure(text="Horizon / Target: Clear")
            self.dist_lbl.configure(text="Distance / Impact: --")

        actions_c = data.get("actions_count", 0)
        self.counters_lbl.configure(
            text=f"Total Actions: {actions_c}\nMode: {prof.category.upper()}"
        )


if __name__ == "__main__":
    app = GamePilotHUD()
    app.mainloop()
