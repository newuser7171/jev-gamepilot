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
        if show_preview:
            cv2.namedWindow(preview_win, cv2.WINDOW_NORMAL)
            # Scale preview window to comfortable aspect ratio
            aspect = h / max(1, w)
            preview_w = 420
            preview_h = min(900, int(preview_w * aspect))
            cv2.resizeWindow(preview_win, preview_w, preview_h)

            # Instant splash frame so window never renders unpainted Windows grey
            splash = np.zeros((preview_h, preview_w, 3), dtype=np.uint8)
            splash[:] = (20, 22, 32)
            cv2.putText(splash, "LAYA + JEV PILOT", (preview_w // 2 - 130, preview_h // 2 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 255, 204), 2, cv2.LINE_AA)
            cv2.putText(splash, "Streaming phone screen...", (preview_w // 2 - 110, preview_h // 2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
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

                # 2. Perception (Universal Vision)
                scene: UniversalSceneState = self.vision.analyze_frame(frame, self.profile)

                # 3. Decision (Laya / Jev System One)
                decision = self.brain.get_action(self.profile, scene)
                action_name = decision.get("action", "wait")

                # 4. Dispatch Physical Android Gesture
                now = time.time()
                if action_name == "start_battle":
                    if now - self.last_battle_launch_time < 7.0:
                        action_name = "wait"
                    else:
                        self.last_battle_launch_time = now

                cooldown = 1.4 if ("clash" in self.profile.id and action_name != "start_battle") else 0.16
                if action_name not in ["wait", "maintain_course", "stand_idle"] and (now - self.last_action_time > cooldown):
                    self.adb.dispatch_action(action_name, target_coords=decision.get("target_coords"))
                    self.total_actions += 1
                    self.last_action_time = now
                    console.print(
                        f"[bold green]⚡ [ACTION][/bold green] [bold white]{action_name.upper()}[/] "
                        f"({decision.get('source')} | conf: {decision.get('confidence'):.2f})"
                    )

                # 5. Live Debug Overlay
                if show_preview:
                    annotated = self.vision.render_debug_overlay(frame, scene, self.profile)

                    # Status Header Banner
                    header_bg = (20, 22, 32)
                    cv2.rectangle(annotated, (0, 0), (w, 140), header_bg, -1)

                    title_text = f"LAYA + JEV FUSION // {self.profile.name.upper()}"
                    cv2.putText(
                        annotated,
                        title_text,
                        (30, 55),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (0, 255, 204),
                        3,
                        cv2.LINE_AA,
                    )

                    elixir_info = f" | Elixir: {getattr(scene, 'elixir', 0)}" if "clash" in self.profile.id else ""
                    stats_text = (
                        f"Action: {action_name.upper()} | Urgency: {int(scene.threat_urgency*100)}%{elixir_info} | "
                        f"Actions: {self.total_actions} | FPS: {fps:.1f}"
                    )
                    cv2.putText(
                        annotated,
                        stats_text,
                        (30, 110),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
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
