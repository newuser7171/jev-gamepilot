"""
Universal Autonomous Phone Game Pilot powered by Laya / Jev System One.
Plays ANY game running on an attached Android phone in real time:
- Auto-detects active foreground game package (Subway Surfers, Fruit Ninja, Earn to Die, Flappy, etc.).
- Sub-30ms local Laya decisions & smart spatial reflexes.
- Microsecond ADB touch/swipe/slice injection.
- Live debug HUD overlay on PC with emergency ESC abort.
"""

import os
import sys
import time
import argparse
import cv2
import numpy as np
from rich.console import Console
from rich.panel import Panel

# Ensure Windows cp1252 handles Unicode cleanly
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from adapters.phone_adapter import AdbController
from profile_manager import ProfileManager, GameProfile
from universal_brain import UniversalBrain
from universal_vision import UniversalVision, UniversalSceneState

console = Console()


class PhoneGamePilot:
    def __init__(self, profile_id: str = "auto", device_serial: str = None):
        self.adb = AdbController(device_serial=device_serial)
        self.profile_mgr = ProfileManager()
        self.auto_mode = (profile_id == "auto")
        self.profile: GameProfile = self._resolve_initial_profile(profile_id)
        self.vision = UniversalVision()
        self.brain = UniversalBrain()

        self.is_running = False
        self.total_actions = 0
        self.last_action_time = 0.0
        self.last_battle_launch_time = 0.0
        self.last_pkg_check_time = 0.0
        self.current_package = "Unknown"

    def _resolve_initial_profile(self, profile_id: str) -> GameProfile:
        if self.auto_mode:
            prof_id, pkg = self.adb.auto_detect_game_profile()
            self.current_package = pkg
            return self.profile_mgr.get_profile(prof_id) or self.profile_mgr.get_profile("mobile_universal")
        return self.profile_mgr.get_profile(profile_id) or self.profile_mgr.get_profile("mobile_universal")

    def _check_auto_profile_switch(self):
        """Periodically checks if the user opened a different game on the phone."""
        now = time.time()
        if now - self.last_pkg_check_time < 2.5:
            return
        self.last_pkg_check_time = now

        detected_prof_id, pkg = self.adb.auto_detect_game_profile()
        is_launcher = any(k in pkg.lower() for k in ["launcher", "homescreen", "systemui", "flower", "homestar"])
        if pkg != self.current_package and pkg not in ["Unknown", ""] and not is_launcher:
            self.current_package = pkg
            new_prof = self.profile_mgr.get_profile(detected_prof_id)
            if new_prof and new_prof.id != self.profile.id:
                self.profile = new_prof
                console.print(
                    f"\n[bold yellow]🎮 Auto-Detected Game Switch:[/] [bold cyan]{new_prof.name}[/] "
                    f"([dim]{pkg}[/dim])\n"
                )

    def start(self, show_preview: bool = True):
        devices = self.adb.get_devices()
        if not devices:
            console.print(
                Panel(
                    "[bold red]❌ No Android device connected via ADB![/bold red]\n\n"
                    "1. Connect your phone via USB cable.\n"
                    "2. Enable [bold cyan]Developer Options[/bold cyan] and [bold cyan]USB Debugging[/bold cyan] on your phone.\n"
                    "3. Tap [bold green]'Allow USB Debugging'[/bold green] on your phone screen.\n"
                    "4. Re-run this script.",
                    title="[bold yellow]Device Not Found[/bold yellow]",
                    border_style="red",
                )
            )
            return

        self.is_running = True
        w, h = self.adb.query_screen_size()
        self.adb.start_background_stream()

        console.print(
            Panel(
                f"[bold green]📱 Device Connected:[/] {self.adb.device_serial}\n"
                f"[bold green]Screen Resolution:[/] {w}x{h}\n"
                f"[bold green]Active Game Profile:[/] {self.profile.name} ({self.profile.category.upper()})\n"
                f"[bold green]Auto-Game Detection:[/] {'ENABLED (Adapts to any game automatically)' if self.auto_mode else 'LOCKED'}\n"
                f"[bold green]Neural Brain:[/] Dual Laya + Jev Fusion (Local Reflexes + System One Consensus)\n\n"
                f"[bold yellow]👉 Launch any game on your phone. Laya + Jev will drive automatically.[/bold yellow]\n"
                f"[bold red]Press ESC or Q in the PC preview window to STOP immediately.[/bold red]",
                title="[bold cyan]⚡ Universal Phone Game Pilot Active (Laya + Jev Fusion)[/bold cyan]",
                border_style="cyan",
            )
        )

        preview_win = f"Laya + Jev Phone Pilot // {self.profile.name}"
        current_orientation = None
        preview_w, preview_h = 420, 900

        if show_preview:
            cv2.namedWindow(preview_win, cv2.WINDOW_NORMAL)
            is_init_landscape = (w > h)
            if is_init_landscape:
                preview_w = 880
                preview_h = int(preview_w * (h / max(1, w)))
            else:
                preview_w = 420
                preview_h = min(920, int(preview_w * (h / max(1, w))))
            cv2.resizeWindow(preview_win, preview_w, preview_h)

            # Instant splash frame so window never renders unpainted Windows grey
            splash = np.zeros((preview_h, preview_w, 3), dtype=np.uint8)
            splash[:] = (20, 22, 32)
            cv2.putText(splash, "LAYA + JEV PILOT", (preview_w // 2 - 130, preview_h // 2 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 204), 2, cv2.LINE_AA)
            cv2.putText(splash, "Streaming phone screen...", (preview_w // 2 - 110, preview_h // 2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 180, 180), 1, cv2.LINE_AA)
            cv2.imshow(preview_win, splash)
            cv2.waitKey(1)

        fps = 0.0
        frame_count = 0
        fps_t0 = time.time()

        try:
            while self.is_running:
                loop_t0 = time.perf_counter()

                # Check auto game switch
                if self.auto_mode:
                    self._check_auto_profile_switch()

                # 1. Grab FRESH Frame from Phone Buffer (strictly sequential, zero stale frames)
                frame = self.adb.get_fresh_frame(timeout=2.0)
                if frame is None:
                    if show_preview:
                        key = cv2.waitKey(20) & 0xFF
                        if key in [27, ord('q'), ord('Q')]:
                            break
                    time.sleep(0.05)
                    continue

                cur_h, cur_w = frame.shape[:2]
                is_landscape = (cur_w > cur_h)
                new_orientation = "landscape" if is_landscape else "portrait"

                if new_orientation != current_orientation:
                    current_orientation = new_orientation
                    if is_landscape:
                        preview_w = 880
                        preview_h = int(preview_w * (cur_h / max(1, cur_w)))
                    else:
                        preview_w = 420
                        preview_h = min(920, int(preview_w * (cur_h / max(1, cur_w))))
                    if show_preview:
                        cv2.resizeWindow(preview_win, preview_w, preview_h)
                    console.print(
                        f"[bold cyan]📱 Screen Orientation:[/] [bold green]{new_orientation.upper()}[/] "
                        f"({cur_w}x{cur_h} display -> {preview_w}x{preview_h} HUD window)"
                    )

                # 2. Perception (Universal Vision)
                scene: UniversalSceneState = self.vision.analyze_frame(frame, self.profile)

                # 3. Decision (Laya / Jev System One)
                decision = self.brain.get_action(self.profile, scene)
                action_name = decision.get("action", "wait")

                # 4. Dispatch Physical Android Gesture
                now = time.time()
                if action_name == "start_battle":
                    if now - self.last_battle_launch_time < 6.0:
                        action_name = "wait"
                    else:
                        self.last_battle_launch_time = now

                # Adaptive Max Idle Watchdog per Game Genre (Ensures Continuous Active Play for ALL Games)
                if "earntodie" in self.profile.id:
                    max_idle = 0.20  # Continuous gas pedal taps
                elif "flappy" in self.profile.id:
                    max_idle = 0.32  # Steady gravity buoyancy
                elif "fifa" in self.profile.id:
                    max_idle = 0.45  # Continuous sprint & pass
                elif "fruit" in self.profile.id:
                    max_idle = 0.75  # Active slice sweep across rising fruit
                elif "snake" in self.profile.id:
                    max_idle = 0.40  # Rapid grid cornering
                elif "runner" in self.profile.id or self.profile.category == "runner":
                    max_idle = 1.10  # Active slide/jump to maintain momentum
                elif "clash" in self.profile.id:
                    max_idle = 2.20  # Elixir pacing
                elif "solitaire" in self.profile.id:
                    max_idle = 0.95  # Fast continuous puzzle cadence
                elif "card" in self.profile.id:
                    max_idle = 3.50  # Card battle turn pacing
                else:
                    max_idle = 1.50  # General default

                phase = getattr(scene, "game_phase", "")
                if action_name in ["wait", "maintain_course", "stand_idle"] and (now - self.last_action_time > max_idle):
                    if phase != "matchmaking":
                        if "clash" in self.profile.id:
                            if phase == "in_battle":
                                action_name = "deploy_card_right" if (self.total_actions % 2 == 0) else "deploy_card_left"
                            elif phase == "main_menu":
                                action_name = "start_battle"
                            elif phase == "game_over":
                                action_name = "confirm_ok"
                        elif "earntodie" in self.profile.id:
                            action_name = "accelerate"
                        elif "fruit" in self.profile.id:
                            action_name = "combo_slice" if (self.total_actions % 2 == 0) else "slice_target"
                        elif "flappy" in self.profile.id:
                            action_name = "flap"
                        elif "fifa" in self.profile.id:
                            fifa_rot = ["dribble_forward", "sprint_tackle", "pass", "through_pass"]
                            action_name = fifa_rot[self.total_actions % len(fifa_rot)]
                        elif "snake" in self.profile.id:
                            snake_rot = ["turn_right", "turn_down", "turn_left", "turn_up"]
                            action_name = snake_rot[self.total_actions % len(snake_rot)]
                        elif "solitaire" in self.profile.id:
                            sol_rot = [
                                "tap_waste_card",
                                "tap_col_1", "tap_col_2", "tap_col_3",
                                "tap_col_4", "tap_col_5", "tap_col_6", "tap_col_7",
                                "drag_column_transfer",
                                "draw_stock",
                                "auto_complete",
                            ]
                            action_name = sol_rot[self.total_actions % len(sol_rot)]
                        elif "card" in self.profile.id:
                            card_rot = ["play_card_center", "play_card_left", "play_card_right", "attack_face", "end_turn"]
                            action_name = card_rot[self.total_actions % len(card_rot)]
                        elif "runner" in self.profile.id or self.profile.category == "runner":
                            action_name = "slide" if (self.total_actions % 2 == 0) else "jump"
                        elif "solar" in self.profile.id:
                            action_name = "fire_laser" if (self.total_actions % 2 == 0) else "orbital_strike"
                        elif "bitlife" in self.profile.id:
                            action_name = "age_up"
                        elif self.profile.actions:
                            action_name = self.profile.actions[0].name

                if "clash" in self.profile.id and action_name != "start_battle":
                    cooldown = 1.2
                elif "card" in self.profile.id:
                    cooldown = 0.65
                elif "solitaire" in self.profile.id:
                    cooldown = 0.25
                elif "snake" in self.profile.id:
                    cooldown = 0.12
                elif "solar" in self.profile.id:
                    cooldown = 0.20
                elif "earntodie" in self.profile.id:
                    cooldown = 0.25
                elif "fifa" in self.profile.id:
                    cooldown = 0.20
                else:
                    cooldown = 0.15

                is_cooling_down = (now - self.last_action_time <= cooldown)
                if action_name not in ["wait", "maintain_course", "stand_idle"] and not is_cooling_down:
                    self.adb.dispatch_action(action_name, target_coords=decision.get("target_coords"))
                    self.total_actions += 1
                    self.last_action_time = now
                    self.last_dispatched_action = action_name
                    console.print(
                        f"[bold green]⚡ [ACTION][/bold green] [bold white]{action_name.upper()}[/] "
                        f"({decision.get('source')} | conf: {decision.get('confidence'):.2f})"
                    )

                # 5. Live Debug Overlay
                if show_preview:
                    annotated = self.vision.render_debug_overlay(frame, scene, self.profile)

                    # Status Header Banner (adaptive height & font for landscape vs portrait)
                    header_bg = (20, 22, 32)
                    elixir_info = f" | Elixir: {getattr(scene, 'elixir', 0)}" if "clash" in self.profile.id else ""
                    last_act = getattr(self, "last_dispatched_action", action_name)
                    act_status = "COOLDOWN" if is_cooling_down else "ACTIVE"
                    if is_landscape:
                        header_h = 72
                        cv2.rectangle(annotated, (0, 0), (cur_w, header_h), header_bg, -1)
                        title_text = f"LAYA + JEV // {self.profile.name.upper()}"
                        cv2.putText(
                            annotated,
                            title_text,
                            (25, 28),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.75,
                            (0, 255, 204),
                            2,
                            cv2.LINE_AA,
                        )
                        stats_text = (
                            f"Dispatched: {last_act.upper()} [{act_status}] | Urgency: {int(scene.threat_urgency*100)}%{elixir_info} | "
                            f"Actions: {self.total_actions} | FPS: {fps:.1f} | LANDSCAPE"
                        )
                        cv2.putText(
                            annotated,
                            stats_text,
                            (25, 56),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.62,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )
                    else:
                        header_h = 135
                        cv2.rectangle(annotated, (0, 0), (cur_w, header_h), header_bg, -1)
                        title_text = f"LAYA + JEV FUSION // {self.profile.name.upper()}"
                        cv2.putText(
                            annotated,
                            title_text,
                            (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.1,
                            (0, 255, 204),
                            3,
                            cv2.LINE_AA,
                        )
                        stats_text = (
                            f"Dispatched: {last_act.upper()} [{act_status}] | Urgency: {int(scene.threat_urgency*100)}%{elixir_info} | "
                            f"Actions: {self.total_actions} | FPS: {fps:.1f} | PORTRAIT"
                        )
                        cv2.putText(
                            annotated,
                            stats_text,
                            (30, 105),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.85,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )

                    # Downscale display buffer to exact window size for crisp zero-latency rendering
                    disp = cv2.resize(annotated, (preview_w, preview_h), interpolation=cv2.INTER_AREA)
                    cv2.imshow(preview_win, disp)

                    key = cv2.waitKey(1) & 0xFF
                    if key in [27, ord('q'), ord('Q')]:
                        console.print("\n[bold yellow]🛑 Emergency Stop Triggered by User (ESC/Q).[/bold yellow]")
                        break

                # FPS tracking
                frame_count += 1
                if time.time() - fps_t0 >= 1.0:
                    fps = frame_count / (time.time() - fps_t0)
                    frame_count = 0
                    fps_t0 = time.time()

                elapsed = time.perf_counter() - loop_t0
                if elapsed < 0.016:
                    time.sleep(0.016 - elapsed)

        finally:
            self.is_running = False
            self.adb.stop_background_stream()
            if show_preview:
                cv2.destroyAllWindows()
            console.print(f"\n[bold cyan]Pilot stopped. Total actions executed: {self.total_actions}[/bold cyan]")


def main():
    parser = argparse.ArgumentParser(description="Universal Autonomous Phone Game Pilot with Laya")
    parser.add_argument(
        "--profile",
        default="auto",
        help="Game profile (auto, runner_3lane, mobile_fruit_ninja, mobile_earntodie2, flappy_tap, aim_clicker, mobile_universal)",
    )
    parser.add_argument("--device", default=None, help="ADB device serial if multiple attached")
    parser.add_argument("--no-preview", action="store_true", help="Run headless without preview window")

    args = parser.parse_args()
    pilot = PhoneGamePilot(profile_id=args.profile, device_serial=args.device)
    pilot.start(show_preview=not args.no_preview)


if __name__ == "__main__":
    main()
