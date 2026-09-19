"""
Universal Jev System One Brain for Jev-GamePilot.
Dynamically constructs TypeSafe System One schemas for ANY game profile,
evaluating tactical choices, threat severity, and reflex triggers.
"""

import os
import json
import urllib.request
import threading
import time
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from profile_manager import GameAction, GameProfile
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient
from universal_vision import UniversalSceneState

load_dotenv()


class UniversalBrain:
    def __init__(self, model_name: str = "jev-latest"):
        self.client = TypeSafeClient()
        self.model_name = model_name
        self.last_decision: Dict[str, Any] = {
            "action": "wait",
            "threat_score": 0.0,
            "confidence": 1.0,
            "latency_ms": 0.0,
            "source": "init",
            "target_coords": None,
        }
        self.is_querying = False
        self._lock = threading.Lock()

    def query_jev_universal(
        self, profile: GameProfile, scene: UniversalSceneState
    ) -> Optional[Dict[str, Any]]:
        """
        Dynamically queries Jev System One using the active game profile's actions and rules.
        """
        t0 = time.perf_counter()
        try:
            # Build dynamic Choice criteria from profile actions
            criteria = {
                action.name: action.description for action in profile.actions
            }
            if not criteria:
                criteria = {"wait": "Do nothing"}

            # Build semantic state
            state_dict = {
                "game_title": profile.name,
                "game_category": profile.category,
                "player_telemetry": (
                    {
                        "x": scene.player.x,
                        "y": scene.player.y,
                        "vx": round(scene.player.vx, 1),
                        "vy": round(scene.player.vy, 1),
                    }
                    if scene.player
                    else None
                ),
                "threats_in_view": len(scene.threats),
                "nearest_threat": (
                    {
                        "distance_px": round(
                            scene.nearest_threat.distance_to_player, 1
                        ),
                        "x": scene.nearest_threat.x,
                        "y": scene.nearest_threat.y,
                        "w": scene.nearest_threat.w,
                        "h": scene.nearest_threat.h,
                    }
                    if scene.nearest_threat
                    else None
                ),
                "targets_in_view": len(scene.targets),
                "best_target": (
                    {
                        "distance_px": round(
                            scene.best_target.distance_to_player, 1
                        ),
                        "click_x": scene.best_target.click_x,
                        "click_y": scene.best_target.click_y,
                    }
                    if scene.best_target
                    else None
                ),
            }

            questions = {
                "tactical_action": Choice(
                    instructions=f"Select the optimal immediate action for the player in '{profile.name}' based on the oncoming threats and tactical objectives.",
                    criteria=criteria,
                ),
                "threat_severity": Score(
                    instructions="Rate the immediate collision risk, threat urgency, or action necessity on a scale from 0 to 4",
                    criteria=[
                        "0: Safe - Horizon completely clear, no action needed",
                        "1: Low - Obstacle or target distant",
                        "2: Moderate - Approaching action perimeter",
                        "3: High - Critical reaction threshold reached",
                        "4: Critical - Imminent impact or immediate action required",
                    ],
                ),
                "is_urgent_reflex": Noul(
                    instructions="Should the player trigger a rapid emergency reflex immediately?"
                ),
            }

            resp = self.client.system_one(
                state=state_dict, model=self.model_name, questions=questions
            )

            latency = (time.perf_counter() - t0) * 1000.0

            ans = resp.answers
            action_choice = getattr(
                ans["tactical_action"], "choice", profile.actions[0].name
            )
            action_conf = getattr(ans["tactical_action"], "confidence", 0.95)
            danger_raw = float(getattr(ans["threat_severity"], "score", 2.0))
            danger_normalized = min(1.0, danger_raw / 4.0)

            # Target click coordinates if applicable
            target_coords = None
            if scene.best_target:
                target_coords = (
                    scene.best_target.click_x,
                    scene.best_target.click_y,
                )

            decision = {
                "action": action_choice,
                "threat_score": danger_normalized,
                "confidence": round(action_conf, 3),
                "latency_ms": round(latency, 1),
                "source": "jev_system_one",
                "target_coords": target_coords,
            }
            with self._lock:
                self.last_decision = decision
            return decision

        except Exception as e:
            # Fallback to keyless Jev decision model via classifier.dev
            return self.query_classifier_dev(profile, scene)

    def query_classifier_dev(
        self, profile: GameProfile, scene: UniversalSceneState
    ) -> Optional[Dict[str, Any]]:
        """
        Keyless fallback directly using classifier.dev (Jev System One).
        Ensures zero-config instant play even without TYPESAFE_API_KEY.
        """
        t0 = time.perf_counter()
        try:
            labels = [action.name for action in profile.actions]
            if "wait" not in labels:
                labels.append("wait")

            threat_desc = "none in view"
            threat_score = 0.0
            if scene.nearest_threat:
                dist = round(scene.nearest_threat.distance_to_player, 1)
                threat_desc = f"obstacle at distance {dist}px (w={scene.nearest_threat.w}, h={scene.nearest_threat.h})"
                threat_score = max(0.0, min(1.0, 1.0 - (dist / 300.0)))

            player_desc = "active"
            if scene.player:
                player_desc = f"pos=({scene.player.x}, {scene.player.y}), vy={round(scene.player.vy, 1)}"

            situation = (
                f"Game: {profile.name} ({profile.category}). Player: {player_desc}. "
                f"Oncoming: {threat_desc}. Actions: "
                + "; ".join([f"{a.name} ({a.description})" for a in profile.actions])
            )

            payload = {
                "labels": labels,
                "input": situation,
                "instructions": (
                    f"Select the single best immediate action for {profile.name}. "
                    "Avoid obstacles, jump or duck hazards, or wait if safe."
                )
            }

            req = urllib.request.Request(
                "https://classifier.dev",
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "jev-gamepilot/2.0 (TypeSafe Jev System One)"
                }
            )

            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.load(resp)

            action = data.get("label", "wait")
            conf = data.get("confidence", 0.95)
            latency = (time.perf_counter() - t0) * 1000

            target_coords = None
            if scene.best_target:
                target_coords = (
                    scene.best_target.click_x,
                    scene.best_target.click_y,
                )

            decision = {
                "action": action,
                "threat_score": round(threat_score, 2),
                "confidence": round(conf, 3),
                "latency_ms": round(latency, 1),
                "source": "jev_classifier_dev",
                "target_coords": target_coords,
            }
            with self._lock:
                self.last_decision = decision
            return decision

        except Exception as err:
            print(f"[UniversalBrain] Keyless fallback error: {err}")
            return None

    def query_jev_async(self, profile: GameProfile, scene: UniversalSceneState):
        """Dispatches Jev query in background thread."""
        if self.is_querying:
            return

        def worker():
            self.is_querying = True
            try:
                self.query_jev_universal(profile, scene)
            finally:
                self.is_querying = False

        threading.Thread(target=worker, daemon=True).start()

    def get_action(
        self, profile: GameProfile, scene: UniversalSceneState
    ) -> Dict[str, Any]:
        """
        Hybrid decision engine:
        Evaluates immediate reflex action when obstacle is in strike zone,
        while maintaining asynchronous Jev System One situational awareness.
        """
        # If aim clicker game, target nearest clickable
        if profile.category == "clicker" and scene.best_target:
            return {
                "action": "click_target",
                "threat_score": 0.90,
                "confidence": 0.99,
                "latency_ms": 0.8,
                "source": "aim_reflex",
                "target_coords": (
                    scene.best_target.click_x,
                    scene.best_target.click_y,
                ),
            }

        # Imminent collision reflex for runners/platformers
        if scene.threat_urgency >= 0.78:
            # Pick first active avoidance action from profile
            avoid_action = profile.actions[0].name
            return {
                "action": avoid_action,
                "threat_score": scene.threat_urgency,
                "confidence": 0.98,
                "latency_ms": 0.5,
                "source": "reflex_actuator",
                "target_coords": None,
            }

        # Trigger Jev System One evaluation if threat spotted
        if scene.threat_urgency > 0.15 and not self.is_querying:
            self.query_jev_async(profile, scene)

        with self._lock:
            return self.last_decision
