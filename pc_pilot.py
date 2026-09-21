"""
PC Game Pilot for Jev-GamePilot & Laya.
Autonomous AI Gaming Agent for PC titles:
1. High-speed desktop screen capture (60+ FPS lock-free) with interactive desktop attachment.
2. Auto-detection & real-time tracking of active PC game windows (Balatro, Slay the Spire, Hearthstone, Solitaire, Aim Lab, Roblox, Minecraft, Chrome Dino, etc.).
3. Low-level Windows hardware input injection (SendInput scan codes, mouse clicks, and smooth drags for card games).
4. Dual-Tier System One decision engine: local sub-30ms Laya + Jev cloud/keyless consensus.
5. Live Cyber HUD preview window with visual bounding boxes, threat telemetry, and emergency failsafes (ESC / Q / F8).
"""

import argparse
import ctypes
import os
import re
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import cv2
import numpy as np
import pygetwindow as gw
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from input_controller import InputController
from profile_manager import GameAction, GameProfile, ProfileManager
from universal_brain import UniversalBrain
from universal_vision import UniversalSceneState, UniversalVision
from vision_engine import VisionEngine

load_dotenv()
console = Console()

# Known PC game window title heuristics
PC_WINDOW_TITLE_MAP = {
    "balatro": "pc_balatro",
    "slay the spire": "pc_slaythespire",
    "hearthstone": "pc_hearthstone",
    "magic": "pc_hearthstone",
    "master duel": "pc_hearthstone",
    "solitaire": "pc_solitaire",
    "freecell": "pc_solitaire",
    "spider": "pc_solitaire",
    "aim lab": "pc_aimlab",
    "kovaak": "pc_aimlab",
    "roblox": "pc_roblox",
    "minecraft": "pc_minecraft",
    "trackmania": "pc_trackmania",
    "dino": "runner_dino",
    "subway": "runner_3lane",
    "chess": "chess_copilot",
}


def attach_input_desktop():
    """Attaches current thread to Windows interactive input desktop."""
    try:
        user32 = ctypes.windll.user32
        hdesk = user32.OpenInputDesktop(0, False, 0x10000000)
        if not hdesk:
            hdesk = user32.OpenDesktopW("default", 0, False, 0x10000000)
        if hdesk:
            user32.SetThreadDesktop(hdesk)
    except Exception:
        pass


class PcGamePilot:
    def __init__(self, profile_id: str = "auto", window_keyword: str = "", arm_inputs: bool = True):
        attach_input_desktop()

        self.profile_mgr = ProfileManager()
        self.vision = VisionEngine()
        self.universal_vision = UniversalVision()
        self.brain = UniversalBrain()
        self.input_ctrl = InputController()

        self.requested_profile_id = profile_id
        self.window_keyword = window_keyword
        self.target_window: Optional[gw.Win32Window] = None
        self.target_window_title: str = "Full Desktop"

        self.is_armed = arm_inputs
        if self.is_armed:
            self.input_ctrl.enable()
        else:
            self.input_ctrl.disable()

        self.profile: GameProfile = self._resolve_game_profile()
        self.total_actions = 0
        self.last_action_time = 0.0
        self.last_dispatched_action: str = "INIT"
        self.fps = 0.0

    def _find_matching_window(self, keyword: str) -> Optional[gw.Win32Window]:
        """Finds open visible window matching keyword."""
        attach_input_desktop()
        kw = keyword.lower().strip()
        try:
            for win in gw.getAllWindows():
                if self._is_game_window(win) and kw in win.title.lower():
                    return win
        except Exception:
            pass
        return None

    @staticmethod
    def _is_game_window(win) -> bool:
        title = (win.title or "").lower()
        return bool(title and "jev-gamepilot" not in title and "game pilot" not in title
                    and not win.isMinimized and win.width > 120 and win.height > 120)

    def _auto_detect_pc_game(self) -> Tuple[GameProfile, Optional[gw.Win32Window]]:
        """Scans open windows to automatically pick the right PC game profile."""
        attach_input_desktop()
        try:
            windows = gw.getAllWindows()
            for win in windows:
                if not self._is_game_window(win):
                    continue
                title_lower = win.title.lower()
                for kw, pid in PC_WINDOW_TITLE_MAP.items():
                    if kw in title_lower:
                        prof = self.profile_mgr.get_profile(pid)
                        if prof:
                            return prof, win
        except Exception:
            pass

        # Fallback to Universal PC profile
        prof = self.profile_mgr.get_profile("pc_universal") or self.profile_mgr.list_profiles()[0]
        return prof, None

    def _resolve_game_profile(self) -> GameProfile:
        """Resolves target profile and snaps capture bounding box."""
        if self.requested_profile_id in ["auto", "detect", ""]:
            if self.window_keyword:
                win = self._find_matching_window(self.window_keyword)
                if win is None:
                    raise ValueError(f"No game window matches: {self.window_keyword}")
                pid = next((pid for kw, pid in PC_WINDOW_TITLE_MAP.items() if kw in win.title.lower()), "pc_universal")
                prof = self.profile_mgr.get_profile(pid)
            else:
                prof, win = self._auto_detect_pc_game()
            if win:
                self.target_window = win
                self.target_window_title = win.title
                self.vision.set_region(win.left, win.top, win.width, win.height)
            return prof

        prof = self.profile_mgr.get_profile(self.requested_profile_id)
        if not prof:
            prof = self.profile_mgr.get_profile("pc_universal") or self.profile_mgr.list_profiles()[0]

        kw = self.window_keyword or prof.default_window_keyword
        if kw:
            win = self._find_matching_window(kw)
            if win:
                self.target_window = win
                self.target_window_title = win.title
                self.vision.set_region(win.left, win.top, win.width, win.height)

        return prof

    def _sync_window_coordinates(self):
        """Dynamically tracks moving or resizing PC game windows in real time."""
        if self.target_window:
            try:
                # Refresh window geometry
                if self.target_window.isMinimized:
                    return
                l, t, w, h = self.target_window.left, self.target_window.top, self.target_window.width, self.target_window.height
                if w > 100 and h > 100:
                    reg = self.vision.region
                    if abs(reg["left"] - l) > 4 or abs(reg["top"] - t) > 4 or abs(reg["width"] - w) > 4 or abs(reg["height"] - h) > 4:
                        self.vision.set_region(l, t, w, h)
            except Exception:
                pass

    def run(self, show_preview: bool = True):
        """Main autonomous perception-action loop for PC games."""
        reg = self.vision.region
        console.print(
            Panel.fit(
                f"[bold cyan]⚡ PC GAME PILOT // DUAL-TIER AUTONOMOUS AI[/bold cyan]\n"
                f"Target Game: [bold green]{self.profile.name}[/bold green] ([dim]{self.profile.id}[/dim])\n"
                f"Window: [bold white]{self.target_window_title}[/bold white]\n"
                f"Capture Region: [yellow]({reg['left']}, {reg['top']}) {reg['width']}x{reg['height']}[/yellow]\n"
                f"Inputs: [bold {'red' if self.is_armed else 'yellow'}]{'ARMED (Live PC Controls)' if self.is_armed else 'MONITOR ONLY (Disarmed)'}[/]\n"
                f"Safety Controls: [bold white on blue] ESC / Q [/] to Quit | [bold white on red] F8 / Space [/] Toggle Arm/Disarm",
                border_style="cyan",
            )
        )

        preview_win = "Jev-GamePilot // PC Game Preview (ESC to Exit)"
        if show_preview:
            cv2.namedWindow(preview_win, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(preview_win, 960, 540)

        loop_count = 0
        while True:
            t0 = time.perf_counter()
            loop_count += 1

            if loop_count % 30 == 0:
                self._sync_window_coordinates()

            # 1. High-speed Desktop Frame Capture
            frame = self.vision.capture_frame()
            if frame is None or frame.size == 0:
                time.sleep(0.01)
                continue

            cur_h, cur_w = frame.shape[:2]

            # 2. Universal Vision Perception
            scene: UniversalSceneState = self.universal_vision.analyze_frame(frame, self.profile)

            # 3. Dual-Tier Consensus Brain (Laya + Jev)
            decision = self.brain.get_action(self.profile, scene)
            action_name = decision.get("action", "wait")
            target_coords = decision.get("target_coords")

            # 4. Action Cooldown & Physical Dispatch
            now = time.time()
            cooldown = 0.15
            if "balatro" in self.profile.id:
                cooldown = 0.40
            elif "spire" in self.profile.id:
                cooldown = 0.50
            elif "hearthstone" in self.profile.id or "card" in self.profile.id:
                cooldown = 0.60
            elif "solitaire" in self.profile.id:
                cooldown = 0.25
            elif "aim" in self.profile.id:
                cooldown = 0.08
            elif "roblox" in self.profile.id or "minecraft" in self.profile.id:
                cooldown = 0.18
            elif "trackmania" in self.profile.id:
                cooldown = 0.12

            is_cooling = (now - self.last_action_time <= cooldown)
            if action_name not in ["wait", "maintain_course", "stand_idle"] and not is_cooling:
                reg = self.vision.region
                vp_offset = (reg["left"], reg["top"])
                vp_size = (reg["width"], reg["height"])

                if self.is_armed:
                    self.input_ctrl.dispatch_pc_action(
                        action_name,
                        target_coords=target_coords,
                        viewport_offset=vp_offset,
                        viewport_size=vp_size,
                        action_binding=next((a for a in self.profile.actions if a.name == action_name), None),
                    )
                self.total_actions += 1
                self.last_action_time = now
                self.last_dispatched_action = action_name

                console.print(
                    f"[bold green]⚡ [PC ACTION][/bold green] [bold white]{action_name.upper()}[/] "
                    f"({decision.get('source')} | conf: {decision.get('confidence'):.2f} | lat: {decision.get('latency_ms', 0)}ms)"
                )

            # 5. Live Debug HUD Rendering
            if show_preview:
                annotated = self.universal_vision.render_debug_overlay(frame, scene, self.profile)

                # Header Overlay
                header_h = 72
                cv2.rectangle(annotated, (0, 0), (cur_w, header_h), (20, 22, 32), -1)
                title_text = f"LAYA + JEV // {self.profile.name.upper()}"
                cv2.putText(
                    annotated,
                    title_text,
                    (20, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.72,
                    (0, 255, 204),
                    2,
                    cv2.LINE_AA,
                )

                arm_str = "ARMED" if self.is_armed else "DISARMED"
                cool_str = "COOLDOWN" if is_cooling else "ACTIVE"
                stats_text = (
                    f"Action: {action_name.upper()} [{cool_str}] | Dispatched: {self.last_dispatched_action.upper()} | "
                    f"Inputs: {arm_str} | Total: {self.total_actions} | FPS: {self.fps:.1f}"
                )
                cv2.putText(
                    annotated,
                    stats_text,
                    (20, 56),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                # Downscale for smooth zero-lag monitor viewing
                disp = cv2.resize(annotated, (960, int(960 * cur_h / max(1, cur_w))), interpolation=cv2.INTER_AREA)
                cv2.imshow(preview_win, disp)

                key = cv2.waitKey(1) & 0xFF
                if key in [27, ord('q'), ord('Q')]:
                    console.print("\n[bold yellow]🛑 Emergency Stop Triggered by User (ESC / Q).[/bold yellow]")
                    break
                elif key in [ord(' '), 0x77]:  # Space or F8
                    self.is_armed = not self.is_armed
                    if self.is_armed:
                        self.input_ctrl.enable()
                    else:
                        self.input_ctrl.disable()
                    console.print(f"[bold yellow]⚠️ Inputs toggled: {'ARMED' if self.is_armed else 'DISARMED'}[/bold yellow]")

            dt = time.perf_counter() - t0
            self.fps = 1.0 / dt if dt > 0 else 60.0
            if dt < 0.012:
                time.sleep(0.012 - dt)

        cv2.destroyAllWindows()
        self.input_ctrl.disable()
        self.vision.close()
        console.print("[bold cyan]✓ PC Game Pilot Session Terminated Cleanly.[/bold cyan]")


def main():
    parser = argparse.ArgumentParser(description="Universal PC Game Pilot powered by Laya + Jev System One")
    parser.add_argument("--profile", default="auto", help="Game profile ID (e.g. auto, pc_balatro, pc_slaythespire, pc_hearthstone, pc_solitaire, pc_aimlab, pc_roblox, pc_minecraft, runner_dino)")
    parser.add_argument("--window", default="", help="Game window title keyword to snap to (e.g. balatro, spire, hearthstone, solitaire, roblox, chrome)")
    parser.add_argument("--disarm", action="store_true", help="Start in monitor-only mode (inputs disarmed)")
    parser.add_argument("--no-hud", action="store_true", help="Disable OpenCV debug preview window")

    args = parser.parse_args()
    pilot = PcGamePilot(profile_id=args.profile, window_keyword=args.window, arm_inputs=not args.disarm)
    pilot.run(show_preview=not args.no_hud)


if __name__ == "__main__":
    main()
