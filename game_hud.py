"""
Universal Cyber Desktop GUI for Jev-GamePilot.
Built with CustomTkinter to deliver a universal autonomous gaming AI:
1. 📱 Android Phone Pilot & Live Mirror via ADB (Samsung Galaxy, etc.).
2. 🖥️ Universal PC Screen Grabber for ANY Desktop Game (Balatro, Slay the Spire, Solitaire, etc.).
3. 🎮 Native 60 FPS Dino Arena for Instant Playground Testing.
4. Interactive Phone Touch: Click the video screen on PC to physically tap the phone!
5. Real-Time Jev System One Brain Telemetry Deck & Floating Mini-Bar Mode.
"""

import ctypes
import os
import re
import subprocess
import sys
import threading
import time
import webbrowser
from typing import Any, Dict, List, Optional, Tuple

import cv2
import customtkinter as ctk
import numpy as np
from PIL import Image, ImageTk

from adapters.phone_adapter import AdbController, PACKAGE_PROFILE_MAP
from all_games_dialog import AllGamesDialog
from custom_profile_dialog import CustomProfileDialog
from embedded_dino import EmbeddedDinoArena
from pilot_core import PilotCore
from profile_manager import GameAction, GameProfile, ProfileManager
from universal_brain import UniversalBrain
from universal_vision import UniversalSceneState, UniversalVision

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class GamePilotHUD(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("⚡ Jev-GamePilot // Universal Autonomous AI Gaming Station")
        self.geometry("1180x780")
        self.minsize(1060, 720)
        self.configure(fg_color="#0a0b10")

        # Core Engines
        self.core = PilotCore()
        self.core.set_telemetry_callback(self._on_telemetry_update)

        # Phone Pilot Engine
        self.phone_adb = AdbController()
        self.phone_vision = UniversalVision()
        self.phone_brain = UniversalBrain()
        self.phone_profile_mgr = ProfileManager()
        self.phone_profile: GameProfile = self.phone_profile_mgr.get_profile("mobile_solitaire")
        self.is_phone_pilot_running = False
        self.phone_total_actions = 0
        self.phone_last_action_time = 0.0
        self._phone_worker_thread: Optional[threading.Thread] = None
        self._phone_preview_running = False
        self.phone_auto_detect_enabled = True
        self.phone_current_pkg = "Unknown"
        self._last_phone_pkg_check = 0.0

        # Touch geometry tracking for phone mirror
        self.phone_img_rect = (0, 0, 1, 1)  # (offset_x, offset_y, rendered_w, rendered_h)
        self.phone_rendered_dim = (1080, 2340)

        # State tracking
        self.current_img_tk: ImageTk.PhotoImage | None = None
        self.active_arena_mode = "phone" if self.phone_adb.is_connected else "native"
        self.countdown_timer = None
        self._active_bound_key = None
        self.is_mini_mode = False

        self._build_header()
        self._build_main_layout()
        self._build_mini_layout()

        # Keyboard shortcuts
        self.bind("<Escape>", lambda e: self.emergency_stop())

        # Start continuous phone background stream if connected
        if self.phone_adb.is_connected:
            self.phone_adb.start_background_stream()
            self._start_phone_preview()
            self._update_phone_device_info()
        else:
            self.after(500, self._auto_detect_game_window)

    # =========================================================================
    # HEADER
    # =========================================================================

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
            text=" // CYBER AI GAMING STATION",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#70758a",
        )
        sub_lbl.pack(side="left", padx=5)

        # Right side: Mini Mode Toggle & Status Badge
        r_box = ctk.CTkFrame(self.header_frame, fg_color="transparent")
        r_box.pack(side="right", padx=15)

        self.all_games_btn = ctk.CTkButton(
            r_box,
            text="📚 All 28+ Games",
            font=ctk.CTkFont(size=11, weight="bold"),
            width=135,
            height=28,
            fg_color="#1e2136",
            hover_color="#2f3454",
            text_color="#00ffcc",
            command=self._open_all_games_dialog,
        )
        self.all_games_btn.pack(side="left", padx=(0, 10))

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

    # =========================================================================
    # MAIN LAYOUT
    # =========================================================================

    def _build_main_layout(self):
        self.main_container = ctk.CTkFrame(self, fg_color="transparent")
        self.main_container.pack(fill="both", expand=True, padx=15, pady=15)

        # Left Column: Mode Switcher & Mode-Specific Controls
        self.left_col = ctk.CTkFrame(
            self.main_container, fg_color="#12131c", corner_radius=10, width=360
        )
        self.left_col.pack(side="left", fill="y", padx=(0, 10), pady=0)
        self.left_col.pack_propagate(False)

        # Right Column: Viewport & Jev Brain Telemetry
        right_col = ctk.CTkFrame(self.main_container, fg_color="transparent")
        right_col.pack(side="right", fill="both", expand=True, padx=0, pady=0)

        self._build_left_controls(self.left_col)
        self._build_right_viewport(right_col)

    # =========================================================================
    # LEFT CONTROLS (MODE SELECTOR + CONTEXT PANELS)
    # =========================================================================

    def _build_left_controls(self, parent):
        pad_x = 15

        # 0. Primary Gaming Mode Selector
        mode_lbl = ctk.CTkLabel(
            parent,
            text="AI PILOT STATION MODE",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#00ffcc",
        )
        mode_lbl.pack(anchor="w", padx=pad_x, pady=(12, 4))

        self.mode_selector = ctk.CTkSegmentedButton(
            parent,
            values=[
                "📱 Phone (ADB)",
                "🖥️ PC Window",
                "🎮 Dino Arena",
            ],
            command=self._on_mode_switched,
            fg_color="#1a1c26",
            selected_color="#00d26a",
            selected_hover_color="#00b058",
            unselected_color="#1a1c26",
            unselected_hover_color="#2b2f42",
            text_color="#ffffff",
            font=ctk.CTkFont(size=11, weight="bold"),
        )
        init_val = "📱 Phone (ADB)" if self.phone_adb.is_connected else "🎮 Dino Arena"
        self.mode_selector.set(init_val)
        self.mode_selector.pack(fill="x", padx=pad_x, pady=(0, 10))

        # Separator
        ctk.CTkFrame(parent, height=1, fg_color="#1f2230").pack(
            fill="x", padx=pad_x, pady=(0, 8)
        )

        # Container for context-specific panels
        self.control_container = ctk.CTkFrame(parent, fg_color="transparent")
        self.control_container.pack(fill="both", expand=True, padx=0, pady=0)

        # Build individual control sub-frames
        self._build_phone_control_panel(self.control_container)
        self._build_pc_control_panel(self.control_container)
        self._build_dino_control_panel(self.control_container)

        # Display initial mode panel
        self._show_active_control_panel(self.mode_selector.get())

    # --- 1. PHONE CONTROL PANEL ---
    def _build_phone_control_panel(self, parent):
        self.phone_panel = ctk.CTkFrame(parent, fg_color="transparent")
        pad_x = 15

        # Device connection status card
        dev_card = ctk.CTkFrame(self.phone_panel, fg_color="#1a1c28", corner_radius=8)
        dev_card.pack(fill="x", padx=pad_x, pady=(0, 8))

        dev_row = ctk.CTkFrame(dev_card, fg_color="transparent")
        dev_row.pack(fill="x", padx=10, pady=(8, 4))

        self.phone_dev_lbl = ctk.CTkLabel(
            dev_row,
            text="📱 Checking ADB...",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#ffffff",
        )
        self.phone_dev_lbl.pack(side="left")

        refresh_dev_btn = ctk.CTkButton(
            dev_row,
            text="🔄",
            width=32,
            height=24,
            fg_color="#232638",
            hover_color="#333750",
            command=self._refresh_phone_connection,
        )
        refresh_dev_btn.pack(side="right")

        self.phone_status_lbl = ctk.CTkLabel(
            dev_card,
            text="Status: Inspecting USB...",
            font=ctk.CTkFont(size=10),
            text_color="#70758a",
        )
        self.phone_status_lbl.pack(anchor="w", padx=10, pady=(0, 6))

        # Active App Info & Auto Switch
        app_card = ctk.CTkFrame(self.phone_panel, fg_color="#1a1c28", corner_radius=8)
        app_card.pack(fill="x", padx=pad_x, pady=(0, 8))

        self.phone_app_lbl = ctk.CTkLabel(
            app_card,
            text="App: Detecting...",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#00ffcc",
        )
        self.phone_app_lbl.pack(anchor="w", padx=10, pady=(6, 2))

        self.phone_auto_switch = ctk.CTkSwitch(
            app_card,
            text="Auto-Detect Game Switch",
            font=ctk.CTkFont(size=10),
            text_color="#c0c0c0",
            progress_color="#00ffcc",
            command=self._on_toggle_phone_auto_detect,
        )
        self.phone_auto_switch.select()
        self.phone_auto_switch.pack(anchor="w", padx=10, pady=(0, 6))

        # Phone Game Profile Selector
        prof_lbl = ctk.CTkLabel(
            self.phone_panel,
            text="PHONE GAME PROFILE",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#00ffcc",
        )
        prof_lbl.pack(anchor="w", padx=pad_x, pady=(4, 2))

        # Search & Filter bar for Phone Games
        self.phone_search_entry = ctk.CTkEntry(
            self.phone_panel,
            placeholder_text="🔍 Search phone games (e.g. 8 Ball, Solitaire)...",
            height=28,
            font=ctk.CTkFont(size=11),
            fg_color="#141620",
            border_color="#2b2f42",
        )
        self.phone_search_entry.pack(fill="x", padx=pad_x, pady=(0, 4))
        self.phone_search_entry.bind("<KeyRelease>", self._on_phone_search_changed)

        self.phone_prof_selector = ctk.CTkOptionMenu(
            self.phone_panel,
            values=self._get_phone_profile_dropdown_values(),
            command=self._on_phone_profile_selected,
            fg_color="#1a1c26",
            button_color="#2b2f42",
            button_hover_color="#00ffcc",
            text_color="#ffffff",
            dropdown_fg_color="#1a1c26",
            height=30,
        )
        phone_vals = self._get_phone_profile_dropdown_values()
        self.phone_prof_selector.set(phone_vals[0] if phone_vals else "📱 Universal Android AI")
        self.phone_prof_selector.pack(fill="x", padx=pad_x, pady=(0, 8))

        # Dynamic Quick Manual Touch Shortcuts
        quick_lbl = ctk.CTkLabel(
            self.phone_panel,
            text="QUICK TOUCH SHORTCUTS",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#00ffcc",
        )
        quick_lbl.pack(anchor="w", padx=pad_x, pady=(4, 4))

        self.phone_quick_frame = ctk.CTkFrame(self.phone_panel, fg_color="transparent")
        self.phone_quick_frame.pack(fill="x", padx=pad_x, pady=(0, 8))
        self._update_phone_quick_actions_ui()

        # Big Action Buttons
        self.phone_arm_btn = ctk.CTkButton(
            self.phone_panel,
            text="🚀 START PHONE AUTOPILOT\n(Laya + Jev Fusion)",
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#00d26a",
            hover_color="#00b058",
            text_color="#000000",
            height=48,
            command=self._toggle_phone_autopilot,
        )
        self.phone_arm_btn.pack(fill="x", padx=pad_x, pady=(6, 6))

        self.phone_stop_btn = ctk.CTkButton(
            self.phone_panel,
            text="🛑 STOP AUTOPILOT [ESC]",
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#ff2a55",
            hover_color="#d61f43",
            text_color="#ffffff",
            height=36,
            command=self.emergency_stop,
        )
        self.phone_stop_btn.pack(fill="x", padx=pad_x, pady=(0, 8))

        phone_tip = ctk.CTkLabel(
            self.phone_panel,
            text="💡 Interactive Touch Active:\nClick anywhere on the phone video screen to tap!",
            font=ctk.CTkFont(size=10),
            text_color="#70758a",
            justify="center",
        )
        phone_tip.pack(fill="x", padx=pad_x, pady=(4, 0))

    # --- 2. PC CONTROL PANEL ---
    def _build_pc_control_panel(self, parent):
        self.pc_panel = ctk.CTkFrame(parent, fg_color="transparent")
        pad_x = 15

        sec1_lbl = ctk.CTkLabel(
            self.pc_panel,
            text="PC GAME PROFILE",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#00ffcc",
        )
        sec1_lbl.pack(anchor="w", padx=pad_x, pady=(4, 2))

        # Search & Filter bar for PC Games
        self.pc_search_entry = ctk.CTkEntry(
            self.pc_panel,
            placeholder_text="🔍 Search PC games (e.g. Balatro, 8 Ball, Spire)...",
            height=28,
            font=ctk.CTkFont(size=11),
            fg_color="#141620",
            border_color="#2b2f42",
        )
        self.pc_search_entry.pack(fill="x", padx=pad_x, pady=(0, 4))
        self.pc_search_entry.bind("<KeyRelease>", self._on_pc_search_changed)

        self.profile_selector = ctk.CTkOptionMenu(
            self.pc_panel,
            values=self._get_profile_dropdown_values(),
            command=self._on_profile_selected,
            fg_color="#1a1c26",
            button_color="#2b2f42",
            button_hover_color="#00ffcc",
            text_color="#ffffff",
            dropdown_fg_color="#1a1c26",
        )
        self.profile_selector.pack(fill="x", padx=pad_x, pady=(0, 6))

        self.actions_preview_lbl = ctk.CTkLabel(
            self.pc_panel,
            text=self._format_actions_preview(self.core.current_profile),
            font=ctk.CTkFont(size=10),
            text_color="#70758a",
            justify="left",
            wraplength=310,
        )
        self.actions_preview_lbl.pack(anchor="w", padx=pad_x, pady=(0, 8))

        # Add Custom Profile Button
        new_prof_btn = ctk.CTkButton(
            self.pc_panel,
            text="➕ Create Custom Game Profile...",
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color="#1a1c28",
            hover_color="#282c40",
            text_color="#00ffcc",
            command=self._open_new_profile_dialog,
            height=28,
        )
        new_prof_btn.pack(fill="x", padx=pad_x, pady=(0, 10))

        # Window Snapping
        sec2_lbl = ctk.CTkLabel(
            self.pc_panel,
            text="SCREEN & WINDOW SNAPPING",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#00ffcc",
        )
        sec2_lbl.pack(anchor="w", padx=pad_x, pady=(6, 4))

        win_row = ctk.CTkFrame(self.pc_panel, fg_color="transparent")
        win_row.pack(fill="x", padx=pad_x, pady=(0, 5))

        self.window_dropdown = ctk.CTkOptionMenu(
            win_row,
            values=["(Auto-Detect Game Window)"],
            fg_color="#1a1c26",
            button_color="#2b2f42",
            dropdown_fg_color="#1a1c26",
            width=220,
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
            self.pc_panel,
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
        calib_row = ctk.CTkFrame(self.pc_panel, fg_color="transparent")
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
            self.pc_panel,
            text="Region: Auto Center (800x400)",
            font=ctk.CTkFont(size=10),
            text_color="#70758a",
        )
        self.coords_lbl.pack(anchor="w", padx=pad_x, pady=(0, 8))

        # PC Autopilot Buttons
        self.pc_arm_btn = ctk.CTkButton(
            self.pc_panel,
            text="🚀 START PC AUTOPILOT\n(Jev System One Play)",
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#00d26a",
            hover_color="#00b058",
            text_color="#000000",
            height=48,
            command=lambda: self.toggle_pilot(armed=True),
        )
        self.pc_arm_btn.pack(fill="x", padx=pad_x, pady=(6, 6))

        self.pc_stop_btn = ctk.CTkButton(
            self.pc_panel,
            text="🛑 STOP AUTOPILOT [ESC]",
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#ff2a55",
            hover_color="#d61f43",
            text_color="#ffffff",
            height=36,
            command=self.emergency_stop,
        )
        self.pc_stop_btn.pack(fill="x", padx=pad_x, pady=(0, 8))

    # --- 3. DINO ARENA CONTROL PANEL ---
    def _build_dino_control_panel(self, parent):
        self.dino_panel = ctk.CTkFrame(parent, fg_color="transparent")
        pad_x = 15

        dino_title = ctk.CTkLabel(
            self.dino_panel,
            text="NATIVE DINO RUNNER ARENA",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#00ffcc",
        )
        dino_title.pack(anchor="w", padx=pad_x, pady=(4, 6))

        dino_desc = ctk.CTkLabel(
            self.dino_panel,
            text="Instant 60 FPS offline benchmark playground.\nTest Jev spatial reflex arc and obstacle dodging.",
            font=ctk.CTkFont(size=11),
            text_color="#80859c",
            justify="left",
            wraplength=310,
        )
        dino_desc.pack(anchor="w", padx=pad_x, pady=(0, 10))

        self.dino_arm_btn = ctk.CTkButton(
            self.dino_panel,
            text="🚀 START DINO AUTOPILOT\n(Jev System One)",
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#00d26a",
            hover_color="#00b058",
            text_color="#000000",
            height=48,
            command=lambda: self.toggle_pilot(armed=True),
        )
        self.dino_arm_btn.pack(fill="x", padx=pad_x, pady=(6, 6))

        self.dino_reset_btn = ctk.CTkButton(
            self.dino_panel,
            text="🔄 Reset Dino Arena",
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color="#1f2230",
            hover_color="#2e3146",
            text_color="#ffffff",
            height=30,
            command=self._on_reset_game,
        )
        self.dino_reset_btn.pack(fill="x", padx=pad_x, pady=(0, 6))

        self.dino_stop_btn = ctk.CTkButton(
            self.dino_panel,
            text="🛑 STOP DINO [ESC]",
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#ff2a55",
            hover_color="#d61f43",
            text_color="#ffffff",
            height=36,
            command=self.emergency_stop,
        )
        self.dino_stop_btn.pack(fill="x", padx=pad_x, pady=(0, 8))

    def _show_active_control_panel(self, mode_str: str):
        self.phone_panel.pack_forget()
        self.pc_panel.pack_forget()
        self.dino_panel.pack_forget()

        if "Phone" in mode_str:
            self.phone_panel.pack(fill="both", expand=True)
        elif "PC" in mode_str:
            self.pc_panel.pack(fill="both", expand=True)
        else:
            self.dino_panel.pack(fill="both", expand=True)

    def _on_mode_switched(self, choice: str):
        self._show_active_control_panel(choice)
        if "Phone" in choice:
            self.arena_tab.set("📱 Phone Pilot (ADB Mirror)")
            self._on_arena_tab_changed("Phone")
        elif "PC" in choice:
            self.arena_tab.set("🖥️ Universal Screen Grabber (Any Game)")
            self._on_arena_tab_changed("Universal")
        else:
            self.arena_tab.set("🎮 Native Dino Arena (Instant Play)")
            self._on_arena_tab_changed("Native")

    # =========================================================================
    # RIGHT VIEWPORT & TELEMETRY DECK
    # =========================================================================

    def _build_right_viewport(self, parent):
        # 1. Viewport Card
        viewport_card = ctk.CTkFrame(parent, fg_color="#12131c", corner_radius=10)
        viewport_card.pack(fill="both", expand=True, padx=0, pady=(0, 10))

        vp_top = ctk.CTkFrame(viewport_card, fg_color="transparent", height=38)
        vp_top.pack(fill="x", padx=15, pady=(10, 5))

        self.arena_tab = ctk.CTkSegmentedButton(
            vp_top,
            values=[
                "📱 Phone Pilot (ADB Mirror)",
                "🖥️ Universal Screen Grabber (Any Game)",
                "🎮 Native Dino Arena (Instant Play)",
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
        init_tab = "📱 Phone Pilot (ADB Mirror)" if self.phone_adb.is_connected else "🎮 Native Dino Arena (Instant Play)"
        self.arena_tab.set(init_tab)
        self.arena_tab.pack(side="left")

        self.fps_lbl = ctk.CTkLabel(
            vp_top,
            text="30.0 FPS // Active Arena",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#70758a",
        )
        self.fps_lbl.pack(side="right")

        self.content_container = ctk.CTkFrame(
            viewport_card, fg_color="#08090d", corner_radius=6
        )
        self.content_container.pack(fill="both", expand=True, padx=15, pady=(0, 15))

        # Native Dino Arena Canvas
        self.embedded_arena = EmbeddedDinoArena(
            self.content_container, width=740, height=270
        )

        # Video Canvas (Used for both Phone Mirror and PC Screen Grabber)
        self.video_lbl = ctk.CTkLabel(
            self.content_container,
            text="[ Initializing Video Stream... ]",
            font=ctk.CTkFont(size=13),
            text_color="#53586d",
        )
        # Bind interactive touch click
        self.video_lbl.bind("<Button-1>", self._on_video_click)

        if "Phone" in init_tab:
            self.video_lbl.pack(fill="both", expand=True)
            self.active_arena_mode = "phone"
        else:
            self.embedded_arena.pack(fill="both", expand=True, padx=10, pady=10)
            self.embedded_arena.start()
            self.active_arena_mode = "native"

        # 2. Bottom Jev Brain Telemetry Deck
        telemetry_deck = ctk.CTkFrame(
            parent, fg_color="#12131c", corner_radius=10, height=180
        )
        telemetry_deck.pack(fill="x", padx=0, pady=0)
        telemetry_deck.pack_propagate(False)

        deck_grid = ctk.CTkFrame(telemetry_deck, fg_color="transparent")
        deck_grid.pack(fill="both", expand=True, padx=15, pady=12)

        # Col 1: Action Badge & Urgency Gauge
        col1 = ctk.CTkFrame(deck_grid, fg_color="#1a1c26", corner_radius=8, width=220)
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
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="#00ffcc",
            fg_color="#12131c",
            corner_radius=6,
            height=38,
        )
        self.action_badge.pack(fill="x", padx=12, pady=(0, 6))

        urg_title = ctk.CTkLabel(
            col1,
            text="THREAT URGENCY / STRATEGY",
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
            text="0% (Standby)",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#00ffcc",
        )
        self.urgency_pct.pack(anchor="w", padx=12)

        # Col 2: Spatial & Threat Tracking
        col2 = ctk.CTkFrame(deck_grid, fg_color="#1a1c26", corner_radius=8)
        col2.pack(side="left", fill="both", expand=True, padx=(0, 10))

        c2_title = ctk.CTkLabel(
            col2,
            text="SPATIAL RECOGNITION & PERCEPTION",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color="#70758a",
        )
        c2_title.pack(anchor="w", padx=12, pady=(8, 2))

        grid_inner = ctk.CTkFrame(col2, fg_color="transparent")
        grid_inner.pack(fill="both", expand=True, padx=12, pady=0)

        self.score_lbl = ctk.CTkLabel(
            grid_inner,
            text="Session: Ready",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#ffffff",
        )
        self.score_lbl.pack(anchor="w", pady=1)

        self.obs_lbl = ctk.CTkLabel(
            grid_inner,
            text="Target / Phase: Standby",
            font=ctk.CTkFont(size=11),
            text_color="#00ffcc",
        )
        self.obs_lbl.pack(anchor="w", pady=1)

        self.dist_lbl = ctk.CTkLabel(
            grid_inner,
            text="Telemetry: Live Stream Ready",
            font=ctk.CTkFont(size=11),
            text_color="#c0c0c0",
        )
        self.dist_lbl.pack(anchor="w", pady=1)

        self.speed_lbl = ctk.CTkLabel(
            grid_inner,
            text="Input Mode: Physical ADB Injection (Microsecond)",
            font=ctk.CTkFont(size=11),
            text_color="#70758a",
        )
        self.speed_lbl.pack(anchor="w", pady=1)

        # Col 3: Jev Brain Intuition & Counters
        col3 = ctk.CTkFrame(deck_grid, fg_color="#1a1c26", corner_radius=8, width=230)
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
            text="Total Actions: 0",
            font=ctk.CTkFont(size=11),
            text_color="#e0e0e0",
            justify="left",
        )
        self.counters_lbl.pack(anchor="w", padx=12, pady=2)

        self.fast_fall_lbl = ctk.CTkLabel(
            col3,
            text="Neural Latency: <25ms",
            font=ctk.CTkFont(size=10),
            text_color="#00d26a",
        )
        self.fast_fall_lbl.pack(anchor="w", padx=12)

    # =========================================================================
    # MINI FLOATING HUD LAYOUT
    # =========================================================================

    def _build_mini_layout(self):
        self.mini_container = ctk.CTkFrame(self, fg_color="#0e1017")

        row = ctk.CTkFrame(self.mini_container, fg_color="transparent")
        row.pack(fill="both", expand=True, padx=10, pady=8)

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
            command=self._on_mini_arm_click,
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
            self.is_mini_mode = True
            self.main_container.pack_forget()
            self.header_frame.pack_forget()
            self.mini_container.pack(fill="both", expand=True)
            self.geometry("380x105")
            self.attributes("-topmost", True)
        else:
            self.is_mini_mode = False
            self.mini_container.pack_forget()
            self.header_frame.pack(fill="x", side="top")
            self.main_container.pack(fill="both", expand=True, padx=15, pady=15)
            self.geometry("1180x780")
            self.attributes("-topmost", False)

    def _on_mini_arm_click(self):
        if self.active_arena_mode == "phone":
            self._toggle_phone_autopilot()
        else:
            self.toggle_pilot(armed=True)

    # =========================================================================
    # TAB & ARENA SWITCHING
    # =========================================================================

    def _on_arena_tab_changed(self, choice: str):
        if "Native" in choice:
            self.active_arena_mode = "native"
            self.video_lbl.pack_forget()
            self.embedded_arena.pack(fill="both", expand=True, padx=10, pady=10)
            self.fps_lbl.configure(text="60.0 FPS // Native Dino Arena")
            self.mode_selector.set("🎮 Dino Arena")
            self._show_active_control_panel("Dino Arena")
        elif "Phone" in choice:
            self.active_arena_mode = "phone"
            self.embedded_arena.pack_forget()
            self.video_lbl.pack(fill="both", expand=True)
            self.fps_lbl.configure(text="30.0 FPS // Phone Mirror Active")
            self.mode_selector.set("📱 Phone (ADB)")
            self._show_active_control_panel("Phone")
            self._start_phone_preview()
            self._update_phone_device_info()
        else:
            self.active_arena_mode = "external"
            self.embedded_arena.pack_forget()
            self.video_lbl.pack(fill="both", expand=True)
            self.fps_lbl.configure(text="0.0 FPS // PC Screen Grabber")
            self.mode_selector.set("🖥️ PC Window")
            self._show_active_control_panel("PC Window")

    # =========================================================================
    # PHONE PILOT & LIVE TOUCH MIRROR
    # =========================================================================

    def _refresh_phone_connection(self):
        self.phone_adb._check_connection()
        self._update_phone_device_info()

    def _update_phone_device_info(self):
        if self.phone_adb.is_connected:
            # Query brand & model
            brand = "Samsung"
            model = "Galaxy"
            try:
                cmd = self.phone_adb._cmd_prefix() + ["shell", "getprop", "ro.product.model"]
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=2)
                model = res.stdout.strip() or "Android Device"
                cmd2 = self.phone_adb._cmd_prefix() + ["shell", "getprop", "ro.product.brand"]
                res2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=2)
                brand = res2.stdout.strip().capitalize() or "Android"
            except Exception:
                pass

            w, h = self.phone_adb.screen_width, self.phone_adb.screen_height
            self.phone_dev_lbl.configure(
                text=f"📱 {brand} {model}", text_color="#ffffff"
            )
            self.phone_status_lbl.configure(
                text=f"🟢 CONNECTED: {self.phone_adb.device_serial} ({w}x{h})",
                text_color="#00ffcc",
            )
            self._update_phone_app_badge()
        else:
            self.phone_dev_lbl.configure(
                text="📱 No Device Detected", text_color="#ff2a55"
            )
            self.phone_status_lbl.configure(
                text="🔴 Connect phone via USB with USB Debugging enabled",
                text_color="#ff2a55",
            )
            self.phone_app_lbl.configure(text="App: No Device", text_color="#70758a")

    def _update_phone_app_badge(self):
        if not self.phone_adb.is_connected:
            return
        pkg = self.phone_adb.detect_foreground_package() or "Homescreen / Idle"
        self.phone_current_pkg = pkg
        clean_pkg = pkg.split("/")[-1].replace("com.", "").replace("at.ner.", "")
        self.phone_app_lbl.configure(text=f"App: {clean_pkg[:28]}")

        # Auto select matching profile if enabled
        if self.phone_auto_detect_enabled:
            det_prof_id, _ = self.phone_adb.auto_detect_game_profile()
            new_p = self.phone_profile_mgr.get_profile(det_prof_id)
            if new_p and new_p.id != self.phone_profile.id:
                self.phone_profile = new_p
                self._sync_phone_profile_dropdown(new_p.id)

    def _get_phone_profile_dropdown_values(self, query: str = "") -> list[str]:
        q = query.strip().lower()
        items = []
        for p in self.phone_profile_mgr.list_profiles():
            if (
                p.category == "mobile"
                or p.id.startswith("mobile_")
                or p.id in ["runner_3lane", "mobile_8ball_pool"]
            ):
                if not q or q in p.name.lower() or q in p.id.lower() or q in (p.description or "").lower():
                    items.append(f"{p.icon or '📱'} {p.name} ({p.id})")
        if not items:
            items = ["📱 Universal Android AI (mobile_universal)"]
        return items

    def _on_phone_search_changed(self, event=None):
        q = self.phone_search_entry.get()
        vals = self._get_phone_profile_dropdown_values(q)
        self.phone_prof_selector.configure(values=vals)
        if vals:
            self.phone_prof_selector.set(vals[0])
            self._on_phone_profile_selected(vals[0])

    def _update_phone_quick_actions_ui(self):
        if not hasattr(self, "phone_quick_frame"):
            return
        for child in self.phone_quick_frame.winfo_children():
            child.destroy()

        prof_id = getattr(self.phone_profile, "id", "")
        if prof_id == "mobile_8ball_pool":
            shortcuts = [
                ("🎱 Power Break", "break_shot"),
                ("🎯 Aim Pocket", "aim_target_ball"),
                ("📐 Fine-Tune", "fine_tune_aim_right"),
                ("⚡ Shoot Cue", "shoot_power"),
            ]
        elif prof_id == "mobile_solitaire":
            shortcuts = [
                ("♠️ Draw Stock", "draw_stock"),
                ("⚡ Sweep Table", "sweep_all_columns"),
                ("🎯 Center Tap", "tap_center"),
                ("🔄 Auto-Deal", "new_deal"),
            ]
        elif prof_id == "mobile_clash_royale":
            shortcuts = [
                ("⚔️ Play Left", "play_card_1_left"),
                ("🛡️ Play Right", "play_card_2_right"),
                ("💥 Spell Tower", "cast_spell_enemy_tower"),
                ("🔄 Start Battle", "start_battle"),
            ]
        elif prof_id == "runner_3lane":
            shortcuts = [
                ("⬅️ Dodge Left", "swipe_left"),
                ("➡️ Dodge Right", "swipe_right"),
                ("⬆️ Jump", "swipe_up"),
                ("⬇️ Roll", "swipe_down"),
            ]
        elif prof_id == "mobile_fruit_ninja":
            shortcuts = [
                ("🍉 Slice Slash", "swipe_up"),
                ("⚡ Diagonal Cut", "swipe_left"),
                ("🎯 Center Tap", "tap_center"),
                ("🔄 Restart", "tap_center"),
            ]
        else:
            shortcuts = []
            if hasattr(self.phone_profile, "actions") and self.phone_profile.actions:
                for act in self.phone_profile.actions[:4]:
                    shortcuts.append((f"⚡ {act.name.replace('_', ' ').title()}", act.name))
            while len(shortcuts) < 4:
                if len(shortcuts) == 0:
                    shortcuts.append(("🎯 Center Tap", "tap_center"))
                elif len(shortcuts) == 1:
                    shortcuts.append(("🔄 Wait", "wait"))
                elif len(shortcuts) == 2:
                    shortcuts.append(("⬆️ Swipe Up", "swipe_up"))
                else:
                    shortcuts.append(("⬇️ Swipe Down", "swipe_down"))

        for idx, (label, act_name) in enumerate(shortcuts[:4]):
            row = idx // 2
            col = idx % 2
            padx = (0, 4) if col == 0 else (4, 0)
            btn = ctk.CTkButton(
                self.phone_quick_frame,
                text=label,
                font=ctk.CTkFont(size=10, weight="bold"),
                fg_color="#1f2232",
                hover_color="#2c3044",
                text_color="#ffffff",
                height=26,
                command=lambda an=act_name: self._dispatch_quick_touch(an),
            )
            btn.grid(row=row, column=col, padx=padx, pady=2, sticky="ew")

        self.phone_quick_frame.columnconfigure(0, weight=1)
        self.phone_quick_frame.columnconfigure(1, weight=1)

    def _sync_phone_profile_dropdown(self, prof_id: str):
        for val in self.phone_prof_selector._values:
            if prof_id in val:
                self.phone_prof_selector.set(val)
                self._update_phone_quick_actions_ui()
                break

    def _on_toggle_phone_auto_detect(self):
        self.phone_auto_detect_enabled = bool(self.phone_auto_switch.get())

    def _on_phone_profile_selected(self, choice: str):
        match = re.search(r"\((.*?)\)", choice)
        prof_id = match.group(1) if match else "mobile_universal"
        p = self.phone_profile_mgr.get_profile(prof_id)
        if p:
            self.phone_profile = p
            self.score_lbl.configure(text=f"Profile: {p.name}")
            self._update_phone_quick_actions_ui()

    def _dispatch_quick_touch(self, action_name: str):
        if not self.phone_adb.is_connected:
            return
        try:
            if action_name == "tap_center":
                self.phone_adb.tap(
                    self.phone_adb.screen_width // 2, self.phone_adb.screen_height // 2
                )
            else:
                self.phone_adb.dispatch_action(action_name)
            self.phone_total_actions += 1
            self.action_badge.configure(text=action_name.upper())
            self.counters_lbl.configure(text=f"Total Actions: {self.phone_total_actions}")
        except Exception as e:
            pass

    def _on_video_click(self, event):
        """Interactive touch mirror: Clicking on the video frame sends physical tap to the phone!"""
        if self.active_arena_mode != "phone" or not self.phone_adb.is_connected:
            return

        off_x, off_y, disp_w, disp_h = self.phone_img_rect
        click_x = event.x - off_x
        click_y = event.y - off_y

        if 0 <= click_x <= disp_w and 0 <= click_y <= disp_h and disp_w > 0 and disp_h > 0:
            norm_x = click_x / disp_w
            norm_y = click_y / disp_h
            phone_w, phone_h = self.phone_rendered_dim
            real_x = int(norm_x * phone_w)
            real_y = int(norm_y * phone_h)

            # Fire microsecond tap via ADB
            self.phone_adb.tap(real_x, real_y)
            self.phone_total_actions += 1

            self.action_badge.configure(text=f"TOUCH ({real_x},{real_y})")
            self.counters_lbl.configure(text=f"Total Actions: {self.phone_total_actions}")

    def _start_phone_preview(self):
        if self._phone_preview_running:
            return
        self._phone_preview_running = True
        self._phone_preview_tick()

    def _phone_preview_tick(self):
        if not self._phone_preview_running:
            return

        # If phone pilot is running, the pilot worker already updates frames
        if not self.is_phone_pilot_running and self.active_arena_mode == "phone":
            frame = self.phone_adb.get_latest_frame()
            if frame is not None:
                self._render_frame_to_viewport(frame)

        self.after(33, self._phone_preview_tick)

    def _toggle_phone_autopilot(self):
        if self.is_phone_pilot_running:
            self.emergency_stop()
        else:
            if not self.phone_adb.is_connected:
                self.phone_status_lbl.configure(
                    text="❌ Cannot start: No device connected!", text_color="#ff2a55"
                )
                return
            self._start_phone_pilot_worker()

    def _start_phone_pilot_worker(self):
        self.is_phone_pilot_running = True
        self.phone_arm_btn.configure(
            text="🛑 STOP PHONE AUTOPILOT\n(Running Laya + Jev)",
            fg_color="#ff2a55",
            hover_color="#d61f43",
            text_color="#ffffff",
        )
        self.mini_arm_btn.configure(text="🛑 STOP", fg_color="#ff2a55")
        self.status_badge.configure(
            text=f"● AUTOPILOT: {self.phone_profile.name[:18].upper()}",
            text_color="#000000",
            fg_color="#00d26a",
        )

        def worker():
            last_act_time = 0.0
            fps_t0 = time.time()
            f_count = 0
            curr_fps = 30.0

            while self.is_phone_pilot_running:
                frame = self.phone_adb.get_fresh_frame(timeout=1.0)
                if frame is None:
                    time.sleep(0.02)
                    continue

                f_count += 1
                if time.time() - fps_t0 >= 1.0:
                    curr_fps = f_count / (time.time() - fps_t0)
                    f_count = 0
                    fps_t0 = time.time()

                # Check auto game switch
                if self.phone_auto_detect_enabled:
                    now_t = time.time()
                    if now_t - self._last_phone_pkg_check > 2.5:
                        self._last_phone_pkg_check = now_t
                        det_id, pkg = self.phone_adb.auto_detect_game_profile()
                        if pkg and pkg != self.phone_current_pkg and "launcher" not in pkg.lower():
                            self.phone_current_pkg = pkg
                            new_p = self.phone_profile_mgr.get_profile(det_id)
                            if new_p and new_p.id != self.phone_profile.id:
                                self.phone_profile = new_p
                                self.after(0, self._sync_phone_profile_dropdown, new_p.id)

                # 1. Perception
                scene: UniversalSceneState = self.phone_vision.analyze_frame(
                    frame, self.phone_profile
                )

                # 2. Decision
                decision = self.phone_brain.get_action(self.phone_profile, scene)
                action_name = decision.get("action", "wait")

                # 3. Action Dispatch
                now = time.time()
                if "8ball" in self.phone_profile.id or "pool" in self.phone_profile.id:
                    cooldown = 7.5  # Realistic billiards turn cooldown; balls roll for 5-10s
                elif "clash" in self.phone_profile.id:
                    cooldown = 1.2
                elif "solitaire" in self.phone_profile.id:
                    cooldown = 0.25
                else:
                    cooldown = 0.18

                if action_name not in ["wait", "maintain_course", "stand_idle"] and (now - last_act_time > cooldown):
                    try:
                        if "8ball" in self.phone_profile.id or "pool" in self.phone_profile.id:
                            t_coords = getattr(scene, "target_coords", None) or decision.get("target_coords")
                            cb = getattr(scene, "cue_ball", None)
                            power = getattr(scene, "shot_power", 0.65)
                            if action_name in ["execute_shot", "break_shot", "pot_ball"]:
                                if t_coords:
                                    self.phone_adb.execute_8ball_shot(
                                        t_coords[0],
                                        t_coords[1],
                                        power_pct=power,
                                        cue_x=cb[0] if cb else None,
                                        cue_y=cb[1] if cb else None,
                                    )
                                else:
                                    self.phone_adb.shoot_8ball_cue(power_pct=power)
                                self.phone_total_actions += 1
                                last_act_time = now
                                continue

                        self.phone_adb.dispatch_action(
                            action_name, target_coords=decision.get("target_coords")
                        )
                        self.phone_total_actions += 1
                        last_act_time = now
                    except Exception:
                        pass

                # 4. Render overlay
                annotated = self.phone_vision.render_debug_overlay(
                    frame, scene, self.phone_profile
                )

                # 5. Push telemetry
                telemetry = {
                    "frame": annotated,
                    "scene": scene,
                    "decision": decision,
                    "action": action_name,
                    "actions_count": self.phone_total_actions,
                    "profile": self.phone_profile,
                    "fps": curr_fps,
                }
                self.after(0, self._render_phone_telemetry_ui, telemetry)
                time.sleep(0.01)

        self._phone_worker_thread = threading.Thread(target=worker, daemon=True)
        self._phone_worker_thread.start()

    def _render_phone_telemetry_ui(self, data: dict):
        if not self.is_phone_pilot_running:
            return

        fps = data.get("fps", 30.0)
        self.fps_lbl.configure(text=f"{fps:.1f} FPS // Phone Pilot Active")

        frame = data.get("frame")
        if frame is not None and self.active_arena_mode == "phone":
            self._render_frame_to_viewport(frame)

        decision = data.get("decision", {})
        scene = data.get("scene")
        prof = data.get("profile", self.phone_profile)
        act = data.get("action", "wait").upper()

        self.action_badge.configure(text=act)
        self.mini_action_badge.configure(text=act)

        urg = float(getattr(scene, "threat_urgency", 0.0))
        self.urgency_bar.set(urg)
        self.mini_urgency_bar.set(urg)
        self.urgency_pct.configure(text=f"{int(urg * 100)}% Threat Urgency")

        self.score_lbl.configure(text=f"Game: {prof.name}")
        self.obs_lbl.configure(text=f"Action: {act} | Conf: {int(decision.get('confidence', 1.0)*100)}%")
        self.dist_lbl.configure(text=f"Strategy: {decision.get('strategy', 'Adaptive')}")
        self.counters_lbl.configure(text=f"Total Actions: {data.get('actions_count', 0)}")

        lat = decision.get("latency_ms", 18.5)
        src = decision.get("source", "Laya+Jev").upper()
        self.brain_source_lbl.configure(text=f"{src} ({lat:.1f}ms)")

    def _render_frame_to_viewport(self, frame: np.ndarray):
        container_w = max(300, self.content_container.winfo_width())
        container_h = max(200, self.content_container.winfo_height())

        h, w = frame.shape[:2]
        self.phone_rendered_dim = (w, h)

        scale = min((container_w - 20) / w, (container_h - 20) / h)
        new_w = max(50, int(w * scale))
        new_h = max(50, int(h * scale))

        # Record exact rendered geometry for touch mapping
        off_x = max(0, (container_w - new_w) // 2)
        off_y = max(0, (container_h - new_h) // 2)
        self.phone_img_rect = (off_x, off_y, new_w, new_h)

        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        img_tk = ImageTk.PhotoImage(image=pil_img)
        self.video_lbl.configure(image=img_tk, text="")
        self.current_img_tk = img_tk

    # =========================================================================
    # PC PILOT & CALIBRATION
    # =========================================================================

    def _get_profile_dropdown_values(self, query: str = "") -> list[str]:
        q = query.strip().lower()
        names = []
        for p in self.core.profile_mgr.list_profiles():
            if not q or q in p.name.lower() or q in p.id.lower() or q in (p.description or "").lower():
                names.append(p.name)
        if not names:
            names = [p.name for p in self.core.profile_mgr.list_profiles()]
        names.append("➕ Create Custom Game Profile...")
        return names

    def _on_pc_search_changed(self, event=None):
        q = self.pc_search_entry.get()
        vals = self._get_profile_dropdown_values(q)
        self.profile_selector.configure(values=vals)
        if vals:
            self.profile_selector.set(vals[0])
            self._on_profile_selected(vals[0])

    def _open_all_games_dialog(self):
        AllGamesDialog(self, on_select_profile_cb=self._on_catalog_game_selected)

    def _on_catalog_game_selected(self, prof_id: str):
        target_prof = self.phone_profile_mgr.get_profile(prof_id)
        if not target_prof:
            return

        is_mobile = (
            target_prof.category == "mobile"
            or prof_id.startswith("mobile_")
            or prof_id in ["runner_3lane", "mobile_8ball_pool"]
        )

        if is_mobile:
            self.mode_selector.set("📱 Phone (ADB)")
            self._on_mode_switched("📱 Phone (ADB)")
            self.phone_profile = target_prof
            if hasattr(self, "phone_search_entry"):
                self.phone_search_entry.delete(0, "end")
            self.phone_prof_selector.configure(
                values=self._get_phone_profile_dropdown_values()
            )
            self._sync_phone_profile_dropdown(prof_id)
            self._update_phone_quick_actions_ui()
            self.score_lbl.configure(text=f"Selected: {target_prof.name}")
        elif prof_id == "runner_dino":
            self.mode_selector.set("🎮 Dino Arena")
            self._on_mode_switched("🎮 Dino Arena")
            self.score_lbl.configure(text="Selected: Dino Arena")
        else:
            self.mode_selector.set("🖥️ PC Window")
            self._on_mode_switched("🖥️ PC Window")
            self.core.set_profile(prof_id)
            if hasattr(self, "pc_search_entry"):
                self.pc_search_entry.delete(0, "end")
            self.profile_selector.configure(
                values=self._get_profile_dropdown_values()
            )
            self.profile_selector.set(target_prof.name)
            self.actions_preview_lbl.configure(
                text=self._format_actions_preview(target_prof)
            )
            self.score_lbl.configure(text=f"Selected: {target_prof.name}")
            if target_prof.default_window_keyword:
                self.core.vision.snap_to_window(target_prof.default_window_keyword)

    def _format_actions_preview(self, profile: GameProfile) -> str:
        keys_summary = " | ".join(
            [f"{a.name}: {a.key.upper()}" for a in profile.actions[:4]]
        )
        return f"Control Map: {keys_summary}"

    def _on_profile_selected(self, choice: str):
        if "Create Custom" in choice:
            self._open_new_profile_dialog()
            return

        for prof in self.core.profile_mgr.list_profiles():
            if prof.name == choice:
                self.core.set_profile(prof.id)
                self.actions_preview_lbl.configure(
                    text=self._format_actions_preview(prof)
                )
                self.score_lbl.configure(text=f"Profile: {prof.name}")

                if prof.id != "runner_dino":
                    self.arena_tab.set("🖥️ Universal Screen Grabber (Any Game)")
                    self._on_arena_tab_changed("Universal")
                    if prof.default_window_keyword:
                        self.core.vision.snap_to_window(prof.default_window_keyword)
                break

    def _open_new_profile_dialog(self):
        CustomProfileDialog(
            self,
            self.core.profile_mgr,
            on_created_cb=self._on_custom_profile_created,
        )

    def _on_custom_profile_created(self, new_profile: GameProfile):
        self.profile_selector.configure(values=self._get_profile_dropdown_values())
        self.profile_selector.set(new_profile.name)
        self._on_profile_selected(new_profile.name)

    def _calibrate_player_avatar(self):
        frame = self.core.vision.last_frame
        if frame is not None:
            h, w, _ = frame.shape
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

    def _on_reset_game(self):
        self.embedded_arena.reset_game()

    def _refresh_windows(self):
        titles = self.core.vision.list_windows()
        filtered = [
            t
            for t in titles
            if any(
                k in t.lower()
                for k in [
                    "dino", "edge", "chrome", "surf", "subway", "bluestack",
                    "roblox", "minecraft", "steam", "osu", "aim", "solitaire",
                    "balatro", "spire", "hearthstone",
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
        for keyword in ["solitaire", "balatro", "spire", "dino", "t-rex", "edge", "chrome"]:
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
                self.arena_tab.set("🖥️ Universal Screen Grabber (Any Game)")
                self._on_arena_tab_changed("Universal")

                try:
                    import pygetwindow as gw
                    wins = [
                        w for w in gw.getAllWindows() if selected.lower() in w.title.lower()
                    ]
                    if wins:
                        w = wins[0]
                        ctypes.windll.user32.ShowWindow(w._hWnd, 9)
                        ctypes.windll.user32.SetForegroundWindow(w._hWnd)
                except Exception:
                    pass
            else:
                self.coords_lbl.configure(text="Could not lock to window dimensions.")

    def toggle_pilot(self, armed: bool):
        if self.active_arena_mode == "phone":
            self._toggle_phone_autopilot()
            return

        if self.active_arena_mode == "native":
            if self.embedded_arena.autopilot_enabled:
                self.embedded_arena.set_autopilot(False)
                self._update_status_idle()
            else:
                self._engage_native()
        else:
            if self.core.is_running:
                self.core.stop()
                self._update_status_idle()
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
        self.dino_arm_btn.configure(
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
        self.pc_arm_btn.configure(
            text="🛑 STOP AUTOPILOT\n(Click or press ESC)", fg_color="#ff2a55"
        )
        self.mini_arm_btn.configure(text="🛑 STOP", fg_color="#ff2a55")

    def emergency_stop(self):
        if self.countdown_timer:
            self.after_cancel(self.countdown_timer)
            self.countdown_timer = None

        self.is_phone_pilot_running = False
        self.embedded_arena.set_autopilot(False)
        self.core.stop()
        self._update_status_idle()

    def _update_status_idle(self):
        self.status_badge.configure(
            text="● SYSTEM STANDBY",
            text_color="#70758a",
            fg_color="#1a1c26",
        )
        self.phone_arm_btn.configure(
            text="🚀 START PHONE AUTOPILOT\n(Laya + Jev Fusion)",
            fg_color="#00d26a",
            hover_color="#00b058",
            text_color="#000000",
        )
        self.pc_arm_btn.configure(
            text="🚀 START PC AUTOPILOT\n(Jev System One Play)",
            fg_color="#00d26a",
            hover_color="#00b058",
            text_color="#000000",
        )
        self.dino_arm_btn.configure(
            text="🚀 START DINO AUTOPILOT\n(Jev System One)",
            fg_color="#00d26a",
            hover_color="#00b058",
            text_color="#000000",
        )
        self.mini_arm_btn.configure(text="▶ START", fg_color="#00d26a")
        self.action_badge.configure(
            text="STANDBY", text_color="#70758a", fg_color="#12131c"
        )
        self.mini_action_badge.configure(text="STANDBY")
        self.urgency_bar.set(0.0)
        self.mini_urgency_bar.set(0.0)

    def _on_embedded_telemetry(self, data: dict):
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

        urg = float(decision.get("threat_score", 0.0))
        self.urgency_bar.set(urg)
        self.mini_urgency_bar.set(urg)
        self.urgency_pct.configure(text=f"{int(urg * 100)}% Threat Urgency")

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
        self.after(0, self._render_telemetry_ui, data)

    def _render_telemetry_ui(self, data: dict):
        if self.active_arena_mode != "external":
            return

        self.fps_lbl.configure(
            text=f"{data['fps']} FPS // Universal Vision Active"
        )
        frame = data.get("frame")
        if frame is not None:
            self._render_frame_to_viewport(frame)

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

        if hasattr(state, "fen"):
            turn = "White" if state.turn else "Black"
            self.obs_lbl.configure(text=f"Chess Turn: {turn} (Move {state.fullmove_number})")
            self.dist_lbl.configure(text=f"FEN: {state.fen()[:32]}...")
        elif hasattr(state, "threats") and state.threats:
            t = state.threats[0]
            self.obs_lbl.configure(text=f"Threat: {int(t.distance_to_player)}px away")
            self.dist_lbl.configure(text=f"Threat Box: ({t.x}, {t.y}) {t.w}x{t.h}px")
        elif hasattr(state, "targets") and state.targets:
            tgt = state.targets[0]
            self.obs_lbl.configure(text=f"Target: ({tgt.click_x}, {tgt.click_y})")
            self.dist_lbl.configure(text="Aim Lock: Active")
        else:
            self.obs_lbl.configure(text="Horizon / Target: Clear")
            self.dist_lbl.configure(text="Distance / Impact: --")

        actions_c = data.get("actions_count", 0)
        self.counters_lbl.configure(
            text=f"Total Actions: {actions_c}\nMode: {prof.category.upper()}"
        )

        src = decision.get("source", "brain").replace("_", " ").upper()
        lat = decision.get("latency_ms", 0.0)
        conf = int(decision.get("confidence", 1.0) * 100)
        self.brain_source_lbl.configure(text=f"{src} ({lat:.1f}ms)")
        self.fast_fall_lbl.configure(text=f"Confidence: {conf}%")


if __name__ == "__main__":
    app = GamePilotHUD()
    app.mainloop()
