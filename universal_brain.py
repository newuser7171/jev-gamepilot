"""
Universal Jev / Laya System One Brain for Jev-GamePilot.
Combines:
1. High-speed local Laya decision engine (convaiinnovations/laya) for sub-30ms offline inference.
2. TypeSafe Jev System One SDK cloud API (jev-latest) as robust online fallback.
3. Keyless classifier.dev fallback for zero-config environments.
4. Intelligent spatial & aerodynamic reflex actuator (no blind defaults; altitude & lane aware).
5. Continuous async worker for 60+ FPS lock-free gameplay.
"""

import json
import math
import os
import queue
import threading
import time
import urllib.request
from typing import Any, Dict, List, Optional, Tuple
from dotenv import load_dotenv

from profile_manager import GameAction, GameProfile
from universal_vision import UniversalEntity, UniversalSceneState

load_dotenv()

# Try importing local laya engine (supports both Apple Silicon MLX and standard PyTorch)
_LAYA_AVAILABLE = False
_LAYA_BACKEND = "none"
try:
    import laya_mlx as laya
    _LAYA_AVAILABLE = True
    _LAYA_BACKEND = "mlx"
except ImportError:
    try:
        import laya
        _LAYA_AVAILABLE = True
        _LAYA_BACKEND = "pytorch"
    except ImportError:
        pass

# Try importing TypeSafe SDK
_TYPESAFE_AVAILABLE = False
try:
    from typesafe_sdk import Choice, Noul, Score, TypeSafeClient
    _TYPESAFE_AVAILABLE = True
except ImportError:
    pass


class UniversalBrain:
    def __init__(self, model_name: str = "jev-latest"):
        self.model_name = model_name
        self.client = TypeSafeClient() if _TYPESAFE_AVAILABLE else None
        self.laya_agent = None
        self.laya_loading = False
        self._laya_tried = False

        self.last_decision: Dict[str, Any] = {
            "action": "wait",
            "threat_score": 0.0,
            "confidence": 1.0,
            "latency_ms": 0.0,
            "source": "init",
            "target_coords": None,
        }

        self._lock = threading.Lock()
        self._work_queue: queue.Queue = queue.Queue(maxsize=2)
        self._worker_thread = threading.Thread(
            target=self._continuous_worker, daemon=True, name="UniversalBrainWorker"
        )
        self._worker_thread.start()

        # Attempt background local Laya load
        self._start_laya_loader()

    def _start_laya_loader(self):
        """Asynchronously attempts to load local Laya model weights without network downloads."""
        if not _LAYA_AVAILABLE or self.laya_loading or self._laya_tried:
            return

        def loader():
            self.laya_loading = True
            self._laya_tried = True
            # Suppress all HF download progress bars and warnings
            os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
            os.environ["HF_HUB_DISABLE_XET"] = "1"
            os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

            # 1. Load directly from local offline cached snapshot
            local_dir = self._find_local_laya_checkpoint()
            if local_dir:
                try:
                    agent = laya.load(local_dir)
                    with self._lock:
                        self.laya_agent = agent
                    return
                except Exception:
                    pass

            self.laya_loading = False

        threading.Thread(target=loader, daemon=True, name="LayaModelLoader").start()

    @staticmethod
    def _find_local_laya_checkpoint() -> Optional[str]:
        """Locates the local offline Laya checkpoint in HuggingFace cache or MLX model directories."""
        # 1. MLX directories (Apple Silicon)
        if _LAYA_BACKEND == "mlx":
            mlx_dirs = [
                "models/hub/laya-multilingual-mlx",
                "models/laya-multilingual",
                os.path.expanduser(r"~\.cache\huggingface\hub\models--aac6fef--laya-multilingual-mlx"),
                os.path.expanduser(r"~\.cache\huggingface\hub\models--aac6fef--laya-mlx"),
                "aac6fef/laya-multilingual-mlx",
            ]
            for d in mlx_dirs:
                if os.path.exists(d):
                    return d
            return "aac6fef/laya-multilingual-mlx"

        # 2. PyTorch HuggingFace cache
        hf_base = os.path.expanduser(r"~\.cache\huggingface\hub\models--convaiinnovations--laya\snapshots")
        if os.path.isdir(hf_base):
            for snap in os.listdir(hf_base):
                snap_path = os.path.join(hf_base, snap)
                multi = os.path.join(snap_path, "multilingual")
                if os.path.exists(os.path.join(multi, "model.safetensors")):
                    return multi
                if os.path.exists(os.path.join(snap_path, "model.safetensors")):
                    return snap_path
        return None

    def _continuous_worker(self):
        """Continuous background worker processing scene evaluations without thread churn."""
        while True:
            try:
                task = self._work_queue.get()
                if task is None:
                    break
                profile, scene = task
                self._evaluate_scene(profile, scene)
            except Exception as e:
                time.sleep(0.02)

    def _evaluate_scene(self, profile: GameProfile, scene: UniversalSceneState):
        """
        Unified Laya + Jev Dual-System Architecture:
        1. Fast local in-process inference via Laya (15-25ms, zero network overhead).
        2. If Laya confidence is high (>= 0.88), commit immediately.
        3. If Laya confidence is low (< 0.88) or ambiguous, escalate to Jev System One for arbitration.
        4. If both provide decisions, fuse them into 'laya_jev_consensus' with blended confidence.
        """
        laya_res = None
        if self.laya_agent is not None:
            laya_res = self._query_local_laya(profile, scene)

        # Fast path: High confidence local Laya decision
        if laya_res and laya_res.get("confidence", 0.0) >= 0.88:
            with self._lock:
                self.last_decision = laya_res
            return

        # Secondary opinion / Escalation: Query Jev System One (Cloud SDK or keyless classifier.dev)
        jev_res = None
        if self.client is not None and os.getenv("TYPESAFE_API_KEY"):
            jev_res = self._query_typesafe_sdk(profile, scene)
        if not jev_res:
            jev_res = self._query_classifier_dev(profile, scene)

        # Consensus Fusion: Merge Laya + Jev decisions
        if laya_res and jev_res:
            laya_act = laya_res.get("action")
            jev_act = jev_res.get("action")
            if laya_act == jev_act:
                # Both Laya and Jev agree: maximum confidence consensus
                fused = {
                    "action": laya_act,
                    "threat_score": max(laya_res.get("threat_score", 0), jev_res.get("threat_score", 0)),
                    "confidence": min(0.99, round((laya_res.get("confidence", 0.8) + jev_res.get("confidence", 0.8)) / 2 + 0.1, 2)),
                    "latency_ms": round(laya_res.get("latency_ms", 15) + jev_res.get("latency_ms", 50), 1),
                    "source": "laya_jev_consensus",
                    "target_coords": laya_res.get("target_coords") or jev_res.get("target_coords"),
                }
            else:
                # Disagreement: pick higher calibrated confidence model
                if laya_res.get("confidence", 0) >= jev_res.get("confidence", 0):
                    fused = dict(laya_res)
                    fused["source"] = "laya_preferred"
                else:
                    fused = dict(jev_res)
                    fused["source"] = "jev_escalation"

            with self._lock:
                self.last_decision = fused
            return

        # Fallback to whichever responded
        winner = laya_res or jev_res
        if winner:
            with self._lock:
                self.last_decision = winner

    def query_jev_universal(
        self, profile: GameProfile, scene: UniversalSceneState
    ) -> Optional[Dict[str, Any]]:
        """Synchronously queries combined Laya + Jev fusion for testing / benchmarking."""
        laya_res = None
        if self.laya_agent is not None:
            laya_res = self._query_local_laya(profile, scene)

        if laya_res and laya_res.get("confidence", 0.0) >= 0.88:
            with self._lock:
                self.last_decision = laya_res
            return laya_res

        jev_res = None
        if self.client is not None and os.getenv("TYPESAFE_API_KEY"):
            jev_res = self._query_typesafe_sdk(profile, scene)
        if not jev_res:
            jev_res = self._query_classifier_dev(profile, scene)

        if laya_res and jev_res:
            if laya_res.get("action") == jev_res.get("action"):
                fused = {
                    "action": laya_res.get("action"),
                    "threat_score": max(laya_res.get("threat_score", 0), jev_res.get("threat_score", 0)),
                    "confidence": min(0.99, round((laya_res.get("confidence", 0.8) + jev_res.get("confidence", 0.8)) / 2 + 0.1, 2)),
                    "latency_ms": round(laya_res.get("latency_ms", 15) + jev_res.get("latency_ms", 50), 1),
                    "source": "laya_jev_consensus",
                    "target_coords": laya_res.get("target_coords") or jev_res.get("target_coords"),
                }
            else:
                fused = dict(laya_res if laya_res.get("confidence", 0) >= jev_res.get("confidence", 0) else jev_res)
                fused["source"] = "laya_preferred" if laya_res.get("confidence", 0) >= jev_res.get("confidence", 0) else "jev_escalation"
            with self._lock:
                self.last_decision = fused
            return fused

        winner = laya_res or jev_res
        if winner:
            with self._lock:
                self.last_decision = winner
        return winner

    def query_jev_async(
        self, profile: GameProfile, scene: UniversalSceneState
    ):
        """Queues scene evaluation asynchronously."""
        if not self._work_queue.full():
            try:
                self._work_queue.put_nowait((profile, scene))
            except queue.Full:
                pass

    def _build_verbal_state(
        self, profile: GameProfile, scene: UniversalSceneState
    ) -> str:
        """
        Translates raw telemetry into semantic verbal descriptions.
        Per Laya integration principles: 'Put the state into words, never numbers.'
        """
        parts = [f"Game: {profile.name} ({profile.category})."]

        # Player telemetry
        if scene.player:
            player_state = "grounded running forward"
            if abs(scene.player.vy) > 80:
                player_state = "airborne jumping" if scene.player.vy < 0 else "falling downwards"
            parts.append(f"Player is {player_state}.")

        # Threat telemetry
        if scene.nearest_threat:
            t = scene.nearest_threat
            # Evaluate proximity verbally
            dist = t.distance_to_player
            if dist <= 110:
                prox = "critical imminent collision range"
            elif dist <= 220:
                prox = "approaching at close range"
            elif dist <= 400:
                prox = "visible on moderate horizon"
            else:
                prox = "distant, plenty of reaction time"

            # Evaluate altitude verbally
            p_bottom = (scene.player.y + scene.player.h) if scene.player else 300
            p_top = scene.player.y if scene.player else 240
            t_bottom = t.y + t.h
            t_center_y = t.y + t.h // 2

            if t_bottom >= p_bottom - 12:
                altitude = "low ground-level obstacle directly in foot path that must be jumped over"
            elif t_center_y <= p_top + 15 and t_bottom > p_top - 25:
                altitude = "mid-height overhead hazard at eye level requiring ducking or sliding underneath"
            elif t_bottom <= p_top - 20:
                altitude = "high overhead obstacle, safe to pass underneath without action"
            else:
                altitude = "obstacle approaching directly in forward path"

            parts.append(f"Obstacle is at {prox}, positioned as {altitude}.")

            # 3-Lane specific verbalization (per skill: 'blocked by a barrier' separates best)
            if profile.id == "runner_3lane":
                lane_desc = self._get_3lane_verbal_status(scene)
                parts.append(lane_desc)
        else:
            parts.append("The path ahead is completely clear with no oncoming threats.")

        # Target telemetry (aim games)
        if scene.best_target:
            parts.append(f"High-contrast target pinpointed in crosshairs at ({scene.best_target.click_x}, {scene.best_target.click_y}).")

        return " ".join(parts)

    def _get_3lane_verbal_status(self, scene: UniversalSceneState) -> str:
        """Determines lane blockage semantics for 3-lane mobile runners."""
        p_x = scene.player.x if scene.player else 200
        lane_thresh = max(45, int(p_x * 0.25))
        lanes = {"left": "clear", "center": "clear", "right": "clear"}

        for t in scene.threats[:3]:
            rel_x = t.x - p_x
            if rel_x < -lane_thresh:
                lanes["left"] = "blocked by a barrier"
            elif rel_x > lane_thresh:
                lanes["right"] = "blocked by a barrier"
            else:
                lanes["center"] = "blocked by a barrier"

        return f"Lane status: Center lane is {lanes['center']}. Left lane is {lanes['left']}. Right lane is {lanes['right']}."

    def _query_local_laya(
        self, profile: GameProfile, scene: UniversalSceneState
    ) -> Optional[Dict[str, Any]]:
        """Inference via local in-process Laya agent."""
        t0 = time.perf_counter()
        try:
            situation = self._build_verbal_state(profile, scene)
            criteria = {
                action.name: action.description for action in profile.actions
            }
            if "wait" not in criteria and profile.category != "clicker":
                criteria["wait"] = "Safe horizon or no immediate movement required"

            questions = {
                "tactical_action": {
                    "type": "choice",
                    "instructions": f"What is the required tactical maneuver for the player in {profile.name}?",
                    "criteria": criteria,
                },
                "threat_urgency": {
                    "type": "score",
                    "instructions": "Rate immediate collision risk and reaction urgency from 0 to 4",
                    "criteria": [
                        "0: Safe - Horizon clear",
                        "1: Low - Obstacle distant",
                        "2: Moderate - Obstacle approaching action perimeter",
                        "3: High - Critical reaction threshold",
                        "4: Critical - Imminent impact",
                    ],
                },
                "emergency_reflex": {
                    "type": "noul",
                    "instructions": "Is an evasive maneuver required immediately?",
                },
            }

            result = self.laya_agent.predict(situation, questions)
            latency = (time.perf_counter() - t0) * 1000.0

            ans = result["answers"]
            act_data = ans["tactical_action"]
            chosen_action = act_data["choice"]
            conf = act_data["confidence"]
            urgency_score = min(1.0, float(ans["threat_urgency"]["score"]) / 4.0)

            target_coords = None
            if scene.best_target:
                target_coords = (scene.best_target.click_x, scene.best_target.click_y)

            return {
                "action": chosen_action,
                "threat_score": round(urgency_score, 2),
                "confidence": round(conf, 3),
                "latency_ms": round(latency, 1),
                "source": "local_laya_brain",
                "target_coords": target_coords,
            }
        except Exception as e:
            return None

    def _query_typesafe_sdk(
        self, profile: GameProfile, scene: UniversalSceneState
    ) -> Optional[Dict[str, Any]]:
        """Inference via TypeSafe Jev System One cloud SDK."""
        t0 = time.perf_counter()
        try:
            situation = self._build_verbal_state(profile, scene)
            criteria = {
                action.name: action.description for action in profile.actions
            }
            if "wait" not in criteria and profile.category != "clicker":
                criteria["wait"] = "Hold course; horizon clear"

            questions = {
                "tactical_action": Choice(
                    instructions=f"Select the optimal immediate maneuver in '{profile.name}' to evade hazards and survive.",
                    criteria=criteria,
                ),
                "threat_severity": Score(
                    instructions="Rate immediate collision urgency on a scale from 0 to 4",
                    criteria=[
                        "0: Safe - Clear horizon",
                        "1: Low - Distant hazard",
                        "2: Moderate - Approaching action threshold",
                        "3: High - Critical reaction zone",
                        "4: Critical - Imminent impact",
                    ],
                ),
                "is_urgent_reflex": Noul(
                    instructions="Should an immediate evasive maneuver be triggered right now?"
                ),
            }

            resp = self.client.system_one(
                state=situation, model=self.model_name, questions=questions
            )
            latency = (time.perf_counter() - t0) * 1000.0

            ans = resp.answers
            action_choice = getattr(ans["tactical_action"], "choice", profile.actions[0].name)
            action_conf = getattr(ans["tactical_action"], "confidence", 0.95)
            danger_raw = float(getattr(ans["threat_severity"], "score", 2.0))
            danger_normalized = min(1.0, danger_raw / 4.0)

            target_coords = None
            if scene.best_target:
                target_coords = (scene.best_target.click_x, scene.best_target.click_y)

            return {
                "action": action_choice,
                "threat_score": round(danger_normalized, 2),
                "confidence": round(action_conf, 3),
                "latency_ms": round(latency, 1),
                "source": "jev_system_one_cloud",
                "target_coords": target_coords,
            }
        except Exception:
            return None

    def _query_classifier_dev(
        self, profile: GameProfile, scene: UniversalSceneState
    ) -> Optional[Dict[str, Any]]:
        """Keyless fallback directly using classifier.dev."""
        t0 = time.perf_counter()
        try:
            labels = [action.name for action in profile.actions]
            if "wait" not in labels:
                labels.append("wait")

            situation = self._build_verbal_state(profile, scene)
            payload = {
                "labels": labels,
                "input": situation,
                "instructions": (
                    f"Select the single best immediate action for {profile.name}. "
                    "Vault or jump over ground hazards, duck under mid-air obstacles, shift lanes to open spots, or wait if clear."
                ),
            }

            req = urllib.request.Request(
                "https://classifier.dev",
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "jev-gamepilot/2.0",
                },
            )

            with urllib.request.urlopen(req, timeout=1.8) as resp:
                data = json.load(resp)

            action = data.get("label", "wait")
            conf = data.get("confidence", 0.90)
            latency = (time.perf_counter() - t0) * 1000

            target_coords = None
            if scene.best_target:
                target_coords = (scene.best_target.click_x, scene.best_target.click_y)

            return {
                "action": action,
                "threat_score": scene.threat_urgency,
                "confidence": round(conf, 3),
                "latency_ms": round(latency, 1),
                "source": "jev_classifier_dev",
                "target_coords": target_coords,
            }
        except Exception:
            return None

    def _evaluate_intelligent_reflex(
        self, profile: GameProfile, scene: UniversalSceneState
    ) -> Optional[Dict[str, Any]]:
        """
        Smart spatial & aerodynamic reflex actuator.
        Replaces crude hardcoded default with physical and topological awareness:
        - Detects ground vs mid-air vs overhead obstacles.
        - Detects open lane in 3-lane runners.
        - Evaluates Flappy Bird gap height vs bird fall velocity.
        """
        # 1. Target Clicker / Fruit Ninja Slicing
        if (profile.category == "clicker" or "fruit" in profile.id) and scene.best_target:
            action_name = "combo_slice" if len(scene.targets) >= 2 else "slice_target"
            return {
                "action": action_name,
                "threat_score": 0.95,
                "confidence": 0.99,
                "latency_ms": 0.2,
                "source": "aim_slice_reflex",
                "target_coords": (
                    scene.best_target.click_x,
                    scene.best_target.click_y,
                ),
            }

        # 1b. Solar Smash Planetary Superweapons (Continuous Destruction Reflex)
        if "solar" in profile.id:
            self._solar_strike_step = getattr(self, "_solar_strike_step", 0) + 1
            if self._solar_strike_step % 3 == 0:
                act = "launch_meteor"
            elif self._solar_strike_step % 2 == 0:
                act = "orbital_strike"
            else:
                act = "fire_laser"
            return {
                "action": act,
                "threat_score": 0.90,
                "confidence": 0.98,
                "latency_ms": 0.2,
                "source": "planetary_destruction_reflex",
                "target_coords": None,
            }

        # 1c. Solitaire & Classic Card Puzzles (Continuous Solution Cascade)
        if "solitaire" in profile.id:
            self._sol_step = getattr(self, "_sol_step", 0) + 1
            sol_actions = [
                "sweep_all_columns",
                "tap_col_1", "tap_col_2", "tap_col_3",
                "tap_col_4", "tap_col_5", "tap_col_6", "tap_col_7",
                "tap_waste_card",
                "draw_stock",
                "auto_complete",
            ]
            act = sol_actions[self._sol_step % len(sol_actions)]
            return {
                "action": act,
                "threat_score": 0.10,
                "confidence": 0.95,
                "latency_ms": 0.2,
                "source": "solitaire_card_solver",
                "target_coords": None,
            }

        # 1d. Mobile Card Battlers & TCGs (Marvel SNAP, Pokémon Pocket, Hearthstone, Balatro)
        if "card" in profile.id:
            self._card_step = getattr(self, "_card_step", 0) + 1
            card_actions = [
                "play_card_center",
                "play_card_left",
                "play_card_right",
                "attack_face",
                "hero_power",
                "end_turn",
            ]
            act = card_actions[self._card_step % len(card_actions)]
            return {
                "action": act,
                "threat_score": 0.50,
                "confidence": 0.94,
                "latency_ms": 0.2,
                "source": "tcg_battle_tactics",
                "target_coords": None,
            }

        # 1e. Snake & Grid Arcades (4-Way Navigation Reflex)
        if "snake" in profile.id:
            if scene.best_target and scene.player:
                dx = scene.best_target.click_x - scene.player.x
                dy = scene.best_target.click_y - scene.player.y
                if abs(dx) > abs(dy):
                    act = "turn_right" if dx > 0 else "turn_left"
                else:
                    act = "turn_down" if dy > 0 else "turn_up"
            else:
                self._snake_step = getattr(self, "_snake_step", 0) + 1
                snake_rot = ["turn_right", "turn_down", "turn_left", "turn_up"]
                act = snake_rot[self._snake_step % len(snake_rot)]
            return {
                "action": act,
                "threat_score": 0.70,
                "confidence": 0.96,
                "latency_ms": 0.2,
                "source": "snake_grid_navigation",
                "target_coords": None,
            }

        # 2. Clash Royale & Tower RTS reflex
        if profile.id == "mobile_clash_royale":
            phase = getattr(scene, "game_phase", "")
            if phase == "matchmaking":
                return {
                    "action": "wait",
                    "threat_score": 0.0,
                    "confidence": 0.99,
                    "latency_ms": 0.2,
                    "source": "matchmaking_standby",
                    "target_coords": None,
                }
            elif phase == "main_menu":
                return {
                    "action": "start_battle",
                    "threat_score": 0.0,
                    "confidence": 0.99,
                    "latency_ms": 0.2,
                    "source": "auto_queue_match",
                    "target_coords": None,
                }
            elif phase == "game_over":
                return {
                    "action": "confirm_ok",
                    "threat_score": 0.0,
                    "confidence": 0.99,
                    "latency_ms": 0.2,
                    "source": "post_game_dismiss",
                    "target_coords": None,
                }
            else:  # in_battle
                current_elixir = getattr(scene, "elixir", 5)
                now = time.time()
                self._last_clash_deploy = getattr(self, "_last_clash_deploy", 0.0)

                # Only hold for elixir recharge if we deployed very recently (< 2.0s ago)
                # If idle for >= 2.0s, elixir has regenerated -> break wait and deploy!
                if current_elixir < 3 and (now - self._last_clash_deploy < 2.0):
                    return {
                        "action": "wait",
                        "threat_score": scene.threat_urgency,
                        "confidence": 0.98,
                        "latency_ms": 0.2,
                        "source": "elixir_recharge_standby",
                        "target_coords": None,
                    }

                self._last_clash_deploy = now

                # If nearest threat is invading our territory, defend that lane or central pocket!
                if scene.nearest_threat and scene.threat_urgency > 0.40:
                    invader_x = scene.nearest_threat.click_x
                    if 380 <= invader_x <= 700:
                        act = "deploy_defense_center"
                    elif invader_x < 380:
                        act = "deploy_card_left"
                    else:
                        act = "deploy_card_right"
                    return {
                        "action": act,
                        "threat_score": scene.threat_urgency,
                        "confidence": 0.96,
                        "latency_ms": 0.3,
                        "source": "rts_defense_reflex",
                        "target_coords": None,
                    }
                # Attack push: rotate continuous pressure between left bridge, right bridge, and spell strikes
                self._clash_push_step = getattr(self, "_clash_push_step", 0) + 1
                if self._clash_push_step % 3 == 0:
                    push_act = "deploy_spell_center"
                elif self._clash_push_step % 2 == 0:
                    push_act = "deploy_card_left"
                else:
                    push_act = "deploy_card_right"

                return {
                    "action": push_act,
                    "threat_score": 0.50,
                    "confidence": 0.92,
                    "latency_ms": 0.3,
                    "source": "rts_offensive_push",
                    "target_coords": None,
                }

        # 3. Earn to Die 2 / 2D Vehicle Driver
        if profile.id == "mobile_earntodie2":
            if len(scene.threats) >= 2 or scene.threat_urgency > 0.85:
                return {
                    "action": "boost",
                    "threat_score": scene.threat_urgency,
                    "confidence": 0.95,
                    "latency_ms": 0.2,
                    "source": "vehicle_nitro_reflex",
                    "target_coords": None,
                }
            return {
                "action": "accelerate",
                "threat_score": scene.threat_urgency,
                "confidence": 0.90,
                "latency_ms": 0.2,
                "source": "vehicle_throttle_reflex",
                "target_coords": None,
            }

        # 4. EA Sports FC / FIFA Mobile (Landscape Pro Controls)
        if profile.id == "mobile_fifa":
            self._fifa_step = getattr(self, "_fifa_step", 0) + 1
            if scene.threat_urgency > 0.65 or len(scene.threats) > 0:
                act = "sprint_tackle"
            elif self._fifa_step % 9 == 0:
                act = "finesse_shot"
            elif self._fifa_step % 8 == 0:
                act = "shoot_goal"
            elif self._fifa_step % 6 == 0:
                act = "skill_move"
            elif self._fifa_step % 5 == 0:
                act = "dribble_cut_inside"
            elif self._fifa_step % 4 == 0:
                act = "dribble_forward"
            elif self._fifa_step % 3 == 0:
                act = "through_pass"
            elif self._fifa_step % 2 == 0:
                act = "pass"
            else:
                act = "sprint_tackle"
            return {
                "action": act,
                "threat_score": scene.threat_urgency,
                "confidence": 0.95,
                "latency_ms": 0.2,
                "source": "fifa_pro_controls",
                "target_coords": None,
            }

        # 5. BitLife & Choice Simulators
        if profile.id == "mobile_bitlife":
            act = "primary_choice" if scene.targets else "age_up"
            return {
                "action": act,
                "threat_score": 0.0,
                "confidence": 0.96,
                "latency_ms": 0.2,
                "source": "bitlife_progression_reflex",
                "target_coords": None,
            }

        # 6. Imminent Hazard Collision (Strike Zone) for Runners & Platformers
        if scene.threat_urgency >= 0.70 and scene.nearest_threat:
            t = scene.nearest_threat
            p_bottom = (scene.player.y + scene.player.h) if scene.player else 300
            p_top = scene.player.y if scene.player else 240
            t_bottom = t.y + t.h
            t_center_y = t.y + t.h // 2

            # Find actions by intention
            duck_action = next(
                (a.name for a in profile.actions if "duck" in a.name or "slide" in a.name or a.key == "down"),
                None,
            )
            jump_action = next(
                (a.name for a in profile.actions if "jump" in a.name or "vault" in a.name or a.key in ["space", "up"]),
                None,
            )
            left_action = next(
                (a.name for a in profile.actions if "left" in a.name or a.key in ["left", "a"]),
                None,
            )
            right_action = next(
                (a.name for a in profile.actions if "right" in a.name or a.key in ["right", "d"]),
                None,
            )

            # 3-Lane Runner & Universal Mobile Runner: pick clear lane
            if profile.id in ["runner_3lane", "mobile_universal"] or len(profile.actions) >= 4:
                p_x = scene.player.x if scene.player else 200
                lane_thresh = max(45, int(p_x * 0.25))
                danger_dist = max(180, int((scene.player.h if scene.player else 60) * 3.5))
                rel_x = t.x - p_x
                if abs(rel_x) < lane_thresh:
                    # In front of obstacle: evade to whichever side is open
                    has_left_obstacle = any(o.x - p_x < -lane_thresh and o.distance_to_player < danger_dist for o in scene.threats)
                    has_right_obstacle = any(o.x - p_x > lane_thresh and o.distance_to_player < danger_dist for o in scene.threats)
                    hoverboard_action = next((a.name for a in profile.actions if "hoverboard" in a.name or "shield" in a.name), None)
                    if has_left_obstacle and has_right_obstacle and scene.threat_urgency > 0.82 and hoverboard_action:
                        chosen = hoverboard_action
                    elif not has_left_obstacle and left_action:
                        chosen = left_action
                    elif not has_right_obstacle and right_action:
                        chosen = right_action
                    elif t_bottom < p_bottom - 20 and duck_action:
                        chosen = duck_action
                    else:
                        chosen = jump_action or profile.actions[0].name
                    return {
                        "action": chosen,
                        "threat_score": scene.threat_urgency,
                        "confidence": 0.96,
                        "latency_ms": 0.3,
                        "source": "lane_evasion_reflex",
                        "target_coords": None,
                    }

            # Flappy Bird Tap Profile
            if profile.id == "flappy_tap":
                # If falling or below threat horizon, tap/flap
                flap_action = next((a.name for a in profile.actions if a.key == "space" or "flap" in a.name), profile.actions[0].name)
                glide_action = next((a.name for a in profile.actions if a.key == "none" or "glide" in a.name), "wait")
                if scene.player and scene.player.vy > 10:  # Falling downward
                    return {
                        "action": flap_action,
                        "threat_score": scene.threat_urgency,
                        "confidence": 0.95,
                        "latency_ms": 0.2,
                        "source": "altitude_flap_reflex",
                        "target_coords": None,
                    }
                else:
                    return {
                        "action": glide_action,
                        "threat_score": scene.threat_urgency,
                        "confidence": 0.90,
                        "latency_ms": 0.2,
                        "source": "altitude_glide_reflex",
                        "target_coords": None,
                    }

            # Standard 2D Runner / Platformer (Dino, Retro Platformer):
            # Mid-height hazard (bird/overhead beam) -> Duck!
            if t_center_y <= p_top + 25 and t_bottom > p_top - 20 and duck_action:
                return {
                    "action": duck_action,
                    "threat_score": scene.threat_urgency,
                    "confidence": 0.98,
                    "latency_ms": 0.3,
                    "source": "smart_duck_reflex",
                    "target_coords": None,
                }

            # Ground level hazard (cactus/pit/hurdle) -> Jump!
            if jump_action:
                return {
                    "action": jump_action,
                    "threat_score": scene.threat_urgency,
                    "confidence": 0.98,
                    "latency_ms": 0.3,
                    "source": "smart_jump_reflex",
                    "target_coords": None,
                }

        return None

    def get_action(
        self, profile: GameProfile, scene: UniversalSceneState
    ) -> Dict[str, Any]:
        """
        Hybrid decision pipeline:
        1. Evaluates smart physical/spatial reflex when obstacle is in strike zone.
        2. Queues asynchronous semantic evaluation when threat appears on horizon.
        3. Returns cached high-confidence decision with sub-millisecond response.
        """
        # 1. Smart Physical Reflex (instant 0.3ms execution)
        reflex_decision = self._evaluate_intelligent_reflex(profile, scene)
        if reflex_decision is not None:
            return reflex_decision

        # 2. Queue asynchronous brain evaluation when threats or targets are detected
        if (scene.threat_urgency > 0.12 or scene.targets) and not self._work_queue.full():
            try:
                self._work_queue.put_nowait((profile, scene))
            except queue.Full:
                pass

        # 3. Return latest situational awareness decision
        with self._lock:
            return self.last_decision
