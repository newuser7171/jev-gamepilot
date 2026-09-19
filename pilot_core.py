"""
Pilot Core Orchestrator for Jev-GamePilot.
Executes the closed-loop perception-decision-action cycle at high frame rates.
"""

import threading
import time
from typing import Any, Callable, Dict, Optional
import cv2
import numpy as np
from adapters.dino_adapter import DinoAdapter, DinoGameState
from adapters.runner_adapter import RunnerAdapter
from input_controller import InputController
from jev_brain import JevBrain
from vision_engine import VisionEngine


class PilotCore:
    def __init__(self):
        self.vision = VisionEngine()
        self.dino_adapter = DinoAdapter()
        self.runner_adapter = RunnerAdapter()
        self.brain = JevBrain()
        self.input_ctrl = InputController()

        self.current_game_mode = "dino"  # "dino" or "runner"
        self.is_running = False
        self.is_armed = False  # If True, inputs are actually dispatched
        self.loop_thread: Optional[threading.Thread] = None

        # Statistics & Telemetry
        self.fps = 0.0
        self.last_state: Optional[DinoGameState] = None
        self.last_decision: Dict[str, Any] = {}
        self.last_rendered_frame: Optional[np.ndarray] = None
        self.total_jumps = 0
        self.total_ducks = 0

        # UI Callback
        self.telemetry_callback: Optional[Callable[[Dict[str, Any]], None]] = (
            None
        )

    def set_telemetry_callback(self, cb: Callable[[Dict[str, Any]], None]):
        self.telemetry_callback = cb

    def start(self, arm_inputs: bool = True):
        """Starts the autonomous perception and pilot loop."""
        if self.is_running:
            return

        self.is_running = True
        self.is_armed = arm_inputs
        if arm_inputs:
            self.input_ctrl.enable()
        else:
            self.input_ctrl.disable()

        self.loop_thread = threading.Thread(
            target=self._run_loop, daemon=True, name="PilotCoreLoop"
        )
        self.loop_thread.start()
        print(f"[PilotCore] Started in mode: {self.current_game_mode}")

    def stop(self):
        """Emergency stops the pilot loop and releases all inputs."""
        self.is_running = False
        self.is_armed = False
        self.input_ctrl.disable()
        print("[PilotCore] Stopped.")

    def _run_loop(self):
        last_action_executed = "run_normal"
        action_cooldown = 0.0

        while self.is_running:
            t0 = time.perf_counter()

            try:
                # 1. Grab Frame
                raw_frame = self.vision.capture_frame()

                # 2. Analyze Frame with selected Adapter
                if self.current_game_mode == "dino":
                    state: DinoGameState = self.dino_adapter.analyze_frame(
                        raw_frame
                    )
                    self.last_state = state

                    # 3. Format State for Jev System One
                    state_dict = {
                        "game": "Chrome Dino Runner",
                        "dino_state": (
                            state.dino.state if state.dino else "running"
                        ),
                        "game_speed_px_sec": round(
                            state.game_speed_px_sec, 1
                        ),
                        "nearest_obstacle": (
                            {
                                "type": state.nearest_obstacle.obstacle_type,
                                "distance_px": state.nearest_obstacle.distance_from_dino,
                                "time_to_impact_ms": round(
                                    state.nearest_obstacle.time_to_impact_ms, 1
                                ),
                            }
                            if state.nearest_obstacle
                            else None
                        ),
                        "total_obstacles_in_view": len(state.obstacles),
                        "ground_y": state.ground_y,
                    }

                    # 4. Get Tactical Action from Hybrid Jev Brain
                    decision = self.brain.get_action(
                        vision_action=state.recommended_action,
                        vision_urgency=state.action_urgency,
                        state_dict=state_dict,
                    )
                    self.last_decision = decision

                    # 5. Dispatch Inputs
                    now = time.time()
                    if state.is_game_over:
                        if now - action_cooldown > 1.5:
                            print(
                                "[PilotCore] Game Over detected -> Auto Restarting"
                            )
                            if self.is_armed:
                                self.input_ctrl.trigger_restart()
                            action_cooldown = now
                    else:
                        chosen_action = decision.get("action", "run_normal")
                        if (
                            chosen_action == "jump"
                            and (state.dino and state.dino.state != "jumping")
                            and (now - action_cooldown > 0.20)
                        ):
                            if self.is_armed:
                                self.input_ctrl.trigger_jump()
                            self.total_jumps += 1
                            action_cooldown = now
                            last_action_executed = "jump"

                        elif (
                            chosen_action == "duck"
                            and (state.dino and state.dino.state != "jumping")
                            and (now - action_cooldown > 0.15)
                        ):
                            if self.is_armed:
                                self.input_ctrl.trigger_duck(hold_sec=0.35)
                            self.total_ducks += 1
                            action_cooldown = now
                            last_action_executed = "duck"

                    # 6. Render Debug Overlay for Preview
                    annotated = self.dino_adapter.render_debug_overlay(
                        raw_frame, state
                    )
                    self.last_rendered_frame = annotated

                    # Calculate FPS
                    dt = time.perf_counter() - t0
                    self.fps = 1.0 / dt if dt > 0 else 60.0

                    # 7. Notify GUI Callback
                    if self.telemetry_callback:
                        self.telemetry_callback(
                            {
                                "fps": round(self.fps, 1),
                                "state": state,
                                "decision": decision,
                                "jumps": self.total_jumps,
                                "ducks": self.total_ducks,
                                "is_armed": self.is_armed,
                                "frame": annotated,
                            }
                        )

            except Exception as e:
                print(f"[PilotCore] Loop iteration error: {e}")
                time.sleep(0.05)

            # Cap frame loop to ~60 FPS
            elapsed = time.perf_counter() - t0
            if elapsed < 0.016:
                time.sleep(0.016 - elapsed)
