"""
Pilot Core Orchestrator for Universal Jev-GamePilot.
Executes real-time perception, dynamic Jev System One decision-making,
and universal input automation across any game profile.
"""

import threading
import time
from typing import Any, Callable, Dict, Optional, Tuple
import cv2
import numpy as np
from adapters.dino_adapter import DinoAdapter, DinoGameState
from input_controller import InputController
from profile_manager import GameAction, GameProfile, ProfileManager
from universal_brain import UniversalBrain
from universal_vision import UniversalSceneState, UniversalVision
from vision_engine import VisionEngine


class PilotCore:
    def __init__(self):
        self.profile_mgr = ProfileManager()
        self.vision = VisionEngine()
        self.universal_vision = UniversalVision()
        self.universal_brain = UniversalBrain()
        self.input_ctrl = InputController()

        # Dino specialized adapter for pixel-perfect Chrome Dino
        self.dino_adapter = DinoAdapter()

        # Active game profile
        self.current_profile: GameProfile = self.profile_mgr.get_profile(
            "runner_dino"
        ) or list(self.profile_mgr.profiles.values())[0]

        self.is_running = False
        self.is_armed = False
        self.loop_thread: Optional[threading.Thread] = None

        # Telemetry & State
        self.fps = 0.0
        self.last_decision: Dict[str, Any] = {}
        self.last_rendered_frame: Optional[np.ndarray] = None
        self.total_actions = 0
        self.total_jumps = 0
        self.total_ducks = 0

        self.telemetry_callback: Optional[Callable[[Dict[str, Any]], None]] = (
            None
        )

    def set_profile(self, profile_id: str) -> bool:
        prof = self.profile_mgr.get_profile(profile_id)
        if prof:
            self.current_profile = prof
            print(f"[PilotCore] Switched profile to: {prof.name}")
            return True
        return False

    def set_player_template(self, template_bgr: np.ndarray):
        """Calibrates player avatar tracking from user-selected image."""
        self.universal_vision.set_player_template(template_bgr)

    def clear_player_template(self):
        self.universal_vision.clear_player_template()

    def set_telemetry_callback(self, cb: Callable[[Dict[str, Any]], None]):
        self.telemetry_callback = cb

    def start(self, arm_inputs: bool = True):
        if self.is_running:
            return

        self.is_running = True
        self.is_armed = arm_inputs
        if arm_inputs:
            self.input_ctrl.enable()
        else:
            self.input_ctrl.disable()

        self.loop_thread = threading.Thread(
            target=self._run_loop, daemon=True, name="UniversalPilotLoop"
        )
        self.loop_thread.start()
        print(f"[PilotCore] Started in universal mode: {self.current_profile.name}")

    def stop(self):
        self.is_running = False
        self.is_armed = False
        self.input_ctrl.disable()
        print("[PilotCore] Stopped.")

    def _run_loop(self):
        last_action_time = 0.0

        while self.is_running:
            t0 = time.perf_counter()

            try:
                # 1. Grab Frame
                raw_frame = self.vision.capture_frame()
                reg = self.vision.region

                # 2. Perception
                if self.current_profile.id == "runner_dino":
                    # Specialized high-speed Dino pipeline
                    state: DinoGameState = self.dino_adapter.analyze_frame(
                        raw_frame
                    )
                    annotated = self.dino_adapter.render_debug_overlay(
                        raw_frame, state
                    )

                    # Dynamic Brain decision
                    decision = {
                        "action": state.recommended_action,
                        "threat_score": state.action_urgency,
                        "confidence": 0.96,
                        "latency_ms": 0.6,
                        "source": "dino_reflex",
                    }
                    self.last_decision = decision

                    # Dispatch Inputs
                    now = time.time()
                    if state.is_game_over:
                        if now - last_action_time > 1.4:
                            if self.is_armed:
                                self.input_ctrl.trigger_restart()
                            last_action_time = now
                    else:
                        act = decision.get("action")
                        if (
                            act == "jump"
                            and (state.dino and state.dino.state != "jumping")
                            and (now - last_action_time > 0.18)
                        ):
                            if self.is_armed:
                                self.input_ctrl.trigger_jump()
                            self.total_jumps += 1
                            self.total_actions += 1
                            last_action_time = now
                        elif (
                            act == "duck"
                            and (state.dino and state.dino.state != "jumping")
                            and (now - last_action_time > 0.14)
                        ):
                            if self.is_armed:
                                self.input_ctrl.trigger_duck()
                            self.total_ducks += 1
                            self.total_actions += 1
                            last_action_time = now

                    # Telemetry payload
                    telemetry_state = state

                else:
                    # Universal Vision & Scene Perception pipeline
                    scene: UniversalSceneState = (
                        self.universal_vision.analyze_frame(
                            raw_frame, self.current_profile
                        )
                    )
                    annotated = self.universal_vision.render_debug_overlay(
                        raw_frame, scene, self.current_profile
                    )

                    # Query Universal Jev System One Brain
                    decision = self.universal_brain.get_action(
                        self.current_profile, scene
                    )
                    self.last_decision = decision

                    # Find and dispatch matching action
                    chosen_name = decision.get("action", "")
                    matched_action = next(
                        (
                            a
                            for a in self.current_profile.actions
                            if a.name == chosen_name
                        ),
                        None,
                    )

                    now = time.time()
                    if matched_action and (now - last_action_time > 0.12):
                        if self.is_armed:
                            self.input_ctrl.dispatch_action(
                                matched_action,
                                target_coords=decision.get("target_coords"),
                                viewport_offset=(reg["left"], reg["top"]),
                            )
                        self.total_actions += 1
                        last_action_time = now

                    telemetry_state = scene

                self.last_rendered_frame = annotated

                # FPS
                dt = time.perf_counter() - t0
                self.fps = 1.0 / dt if dt > 0 else 60.0

                # Notify GUI
                if self.telemetry_callback:
                    self.telemetry_callback(
                        {
                            "fps": round(self.fps, 1),
                            "state": telemetry_state,
                            "decision": decision,
                            "profile": self.current_profile,
                            "actions_count": self.total_actions,
                            "jumps": self.total_jumps,
                            "ducks": self.total_ducks,
                            "is_armed": self.is_armed,
                            "frame": annotated,
                        }
                    )

            except Exception as e:
                print(f"[PilotCore] Universal loop error: {e}")
                time.sleep(0.05)

            elapsed = time.perf_counter() - t0
            if elapsed < 0.016:
                time.sleep(0.016 - elapsed)
