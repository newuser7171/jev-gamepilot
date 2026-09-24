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

# Ensure Windows cp1252 handles Unicode cleanly + line-buffered pipe logs
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

from adapters.phone_adapter import AdbController
from profile_manager import ProfileManager, GameProfile
from universal_brain import UniversalBrain
from universal_vision import UniversalVision, UniversalSceneState

console = Console()


def clash_max_idle(
    *,
    action_name: str,
    strat: str,
    phase: str,
    threat_urgency: float,
    elixir: int,
    default: float = 2.20,
) -> float:
    """Hard-hold only for NO_PLAY / calm adapter waits. Real pressure gets a finite timeout
    so a stuck wait() cannot freeze the pilot while troops close in."""
    no_play = strat in {"save_elixir", "hold_elixir_for_threat", "wait_for_full"}
    adapter_wait = action_name in ("wait", "maintain_course", "stand_idle") and (
        no_play
        or strat.startswith("defend")
        or strat.startswith("push")
        or strat.startswith("counter")
        or strat in ("cycle", "build_push")
    )
    if no_play and phase not in ("main_menu", "game_over", "matchmaking"):
        return 9999.0
    if phase == "main_menu":
        return 1.50
    if phase == "game_over":
        return 1.00
    if adapter_wait and threat_urgency < 0.40:
        return 9999.0
    if adapter_wait:
        # Finite pressure timeout only when we can actually pay for a play.
        # Below the blind-safe floor the adapter's wait means "nothing
        # affordable" — trust it instead of force-tapping into a toast.
        if elixir < 4:
            return 9999.0
        return 2.50
    if no_play or elixir < 5:
        return 5.50
    return default


def clash_blind_force_action(
    *,
    strat: str,
    phase: str,
    threat_urgency: float,
    nearest_threat: object,
    elixir: int,
    total_actions: int,
) -> str:
    """Last-resort action when the adapter returned wait and idle expired.

    Never blind-taps a card slot below elixir 4: the hand may be all 4-5 cost
    cards, and slot rotation picks blindly — that was the live
    "Not enough Elixir!" toast path.
    """
    if phase == "main_menu":
        return "start_battle"
    if phase == "game_over":
        return "confirm_ok"
    if phase not in ("in_battle", "active", ""):
        return "wait"
    if strat in {"save_elixir", "hold_elixir_for_threat"}:
        return "wait"
    if elixir < 4:
        return "wait"
    if threat_urgency >= 0.40 and nearest_threat is not None:
        return "deploy_defense_center"
    return "deploy_card_right" if (total_actions % 2 == 0) else "deploy_card_left"


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
        # Seed to now: a 0.0 epoch makes (now - last) astronomical, so the first
        # idle tick instantly beats even a 9999s clash hold window.
        self.last_action_time = time.time()
        self.last_battle_launch_time = 0.0
        self.last_pkg_check_time = 0.0
        self.current_package = "Unknown"
        # Consecutive confirm_ok dispatches; over limit → BACK escapes a missed dismiss.
        self._confirm_ok_streak = 0

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
                action_name = self.brain.coerce_action_name(
                    self.profile, decision.get("action", "wait"), scene
                )
                if action_name != decision.get("action"):
                    decision = dict(decision)
                    decision["action"] = action_name

                # 4. Dispatch Physical Android Gesture
                now = time.time()

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
                elif self.profile.id == "mobile_coc":
                    max_idle = 1.60  # Deploy cadence / search wait
                elif self.profile.id == "mobile_brawlstars":
                    max_idle = 0.40  # Joystick roam must stay hot
                elif "solitaire" in self.profile.id:
                    max_idle = 0.95  # Fast continuous puzzle cadence
                elif "card" in self.profile.id:
                    max_idle = 3.50  # Card battle turn pacing
                elif "8ball" in self.profile.id or "pool" in self.profile.id:
                    max_idle = 9999.0  # Turn-based billiards: NEVER force blind actions on timeout
                else:
                    max_idle = 1.50  # General default

                phase = getattr(scene, "game_phase", "")
                # Clash: honor NO_PLAY / save_elixir — only force a deploy when
                # elixir is actually spendable or a threat is in our half.
                if "clash" in self.profile.id:
                    strat = str(decision.get("strategy") or "")
                    max_idle = clash_max_idle(
                        action_name=action_name,
                        strat=strat,
                        phase=phase,
                        threat_urgency=float(getattr(scene, "threat_urgency", 0.0) or 0.0),
                        elixir=int(getattr(scene, "elixir", 0) or 0),
                        default=max_idle,
                    )
                if action_name in ["wait", "maintain_course", "stand_idle"] and (now - self.last_action_time > max_idle):
                    if phase != "matchmaking" and "8ball" not in self.profile.id and "pool" not in self.profile.id:
                        scene_pick = (getattr(scene, "recommended_action", "") or "").strip()
                        valid_names = {a.name for a in self.profile.actions}
                        # Never promote menu/game_over actions from a battle-phase idle.
                        if phase in ("in_battle", "active", "") and scene_pick in ("start_battle", "confirm_ok"):
                            scene_pick = ""
                        if scene_pick and scene_pick not in {"wait", "maintain_course", "stand_idle"} and (
                            scene_pick in valid_names
                            or any(scene_pick in a.name or a.name in scene_pick for a in self.profile.actions)
                        ):
                            action_name = scene_pick
                        elif "clash" in self.profile.id:
                            # Blind slot-rotate deploy only as last resort when adapter gave nothing.
                            strat = str(decision.get("strategy") or "")
                            action_name = clash_blind_force_action(
                                strat=strat,
                                phase=phase,
                                threat_urgency=float(getattr(scene, "threat_urgency", 0.0) or 0.0),
                                nearest_threat=getattr(scene, "nearest_threat", None),
                                elixir=int(getattr(scene, "elixir", 0) or 0),
                                total_actions=self.total_actions,
                            )
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
                            # No blind slide/jump rotation — leave idle; threat path
                            # is handled by get_action / lane_evasion_reflex.
                            pass
                        elif "solar" in self.profile.id:
                            action_name = "fire_laser" if (self.total_actions % 2 == 0) else "orbital_strike"
                        elif "bitlife" in self.profile.id:
                            action_name = "age_up"
                        elif self.profile.actions:
                            action_name = self.profile.actions[0].name

                # Single launch gate AFTER idle may re-promote start_battle.
                if action_name == "start_battle":
                    if now - self.last_battle_launch_time < 6.0:
                        action_name = "wait"
                    else:
                        self.last_battle_launch_time = now

                if "clash" in self.profile.id and action_name != "start_battle":
                    cooldown = 1.8  # defend/attack taps need elixir+deploy-time gap
                elif self.profile.id == "mobile_coc" and action_name != "find_match":
                    cooldown = 0.55  # slot→target double-tap rhythm
                elif self.profile.id == "mobile_brawlstars":
                    cooldown = 0.30  # attack cooldown ~0.85s handled in adapter; floor here
                elif "card" in self.profile.id:
                    cooldown = 0.65
                elif "8ball" in self.profile.id or "pool" in self.profile.id:
                    cooldown = 7.5  # Realistic turn cooldown; balls roll for 5-10 seconds
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
                    try:
                        # Specialized 8 Ball Pool shot handling
                        if "8ball" in self.profile.id or "pool" in self.profile.id:
                            t_coords = getattr(scene, "target_coords", None) or decision.get("target_coords")
                            cb = getattr(scene, "cue_ball", None)
                            power = getattr(scene, "shot_power", 0.65)
                            cut = getattr(scene, "cut_angle_deg", 0.0)
                            if action_name == "place_cue_ball":
                                pt = t_coords if t_coords else (int(self.adb.screen_width * 0.50), int(self.adb.screen_height * 0.50))
                                console.print(f"[bold yellow]🎱 [BALL IN HAND][/bold yellow] Placing cue ball at {pt}...")
                                self.adb.tap(pt[0], pt[1])
                                self.total_actions += 1
                                self.last_action_time = now
                                self.last_dispatched_action = action_name
                                continue
                            elif action_name in ["execute_shot", "break_shot", "pot_ball"]:
                                if t_coords:
                                    console.print(
                                        f"[bold cyan]🎱 [8-BALL SHOT LOCK][/bold cyan] Ghost Target: {t_coords} | Cut: {cut}° | Power: {int(power*100)}%"
                                    )
                                    self.adb.execute_8ball_shot(
                                        t_coords[0],
                                        t_coords[1],
                                        power_pct=power,
                                        cue_x=cb[0] if cb else None,
                                        cue_y=cb[1] if cb else None,
                                    )
                                else:
                                    self.adb.shoot_8ball_cue(power_pct=power)
                                self.total_actions += 1
                                self.last_action_time = now
                                self.last_dispatched_action = action_name
                                continue

                        self.adb.dispatch_action(action_name, target_coords=decision.get("target_coords"))
                        self.total_actions += 1
                        self.last_action_time = now
                        self.last_dispatched_action = action_name
                        # confirm_ok death-spiral escape: hit tap misses (wrong overlay /
                        # stale post_ok_pair) → 3000× loop in the wild. BACK every 8.
                        if action_name == "confirm_ok":
                            self._confirm_ok_streak += 1
                            if self._confirm_ok_streak % 8 == 0:
                                console.print(
                                    f"[bold yellow]↩ [CONFIRM ESCAPE][/bold yellow] "
                                    f"{self._confirm_ok_streak}× confirm_ok → BACK keyevent"
                                )
                                try:
                                    self.adb.press_back()
                                except Exception as be:
                                    console.print(f"[red]BACK failed: {be}[/red]")
                        else:
                            self._confirm_ok_streak = 0
                        strat_str = f" | strat: {decision.get('strategy')}" if decision.get("strategy") else ""
                        card_str = f" | card: {decision.get('card_name')}" if decision.get("card_name") else ""
                        tile_str = f" -> {decision.get('square_name')}" if decision.get("square_name") else ""
                        console.print(
                            f"[bold green]⚡ [ACTION][/bold green] [bold white]{action_name.upper()}[/] "
                            f"({decision.get('source')}{strat_str}{card_str}{tile_str} | conf: {decision.get('confidence'):.2f})"
                        )
                        # Pipe-safe audit trail: Rich may buffer on a redirected stdout.
                        try:
                            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "live_actions.log"), "a", encoding="utf-8") as _af:
                                _af.write(
                                    f"{time.strftime('%H:%M:%S')} {action_name} "
                                    f"src={decision.get('source')} strat={decision.get('strategy')} "
                                    f"card={decision.get('card_name')} sq={decision.get('square_name')} "
                                    f"conf={decision.get('confidence')} phase={phase} elixir={getattr(scene, 'elixir', '?')} "
                                    f"n={self.total_actions}\n"
                                )
                        except OSError:
                            pass
                    except Exception as e:
                        console.print(f"[bold red]❌ [ADB DISPATCH ERROR][/bold red] Failed to actuate {action_name}: {e}")

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
    parser.add_argument(
        "--style",
        default=None,
        help="Playstyle id to clone for Clash battles (e.g. hog_cycle, golem_beatdown, custom name)",
    )

    args = parser.parse_args()
    if args.style:
        try:
            from clash_jev.playstyle import set_active

            style = set_active(args.style)
            console.print(f"[bold green]Cloned playstyle: {style.name}[/]")
        except KeyError as exc:
            console.print(f"[bold red]{exc}[/]")
            return
    pilot = PhoneGamePilot(profile_id=args.profile, device_serial=args.device)
    pilot.start(show_preview=not args.no_preview)


if __name__ == "__main__":
    main()
