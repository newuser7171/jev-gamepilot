"""
Jev System One Brain for Jev-GamePilot.
Translates real-time visual game state into semantic decisions using TypeSafe Jev System One.
"""

import os
import threading
import time
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

# Load environment
load_dotenv()


class JevBrain:
    def __init__(self, model_name: str = "jev-latest"):
        self.client = TypeSafeClient()
        self.model_name = model_name
        self.last_decision: Dict[str, Any] = {
            "action": "run_normal",
            "threat_score": 0.0,
            "fast_fall": False,
            "confidence": 1.0,
            "latency_ms": 0.0,
            "source": "init",
        }
        self.is_querying = False
        self._lock = threading.Lock()

    def query_jev_dino(
        self, state_dict: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Queries TypeSafe Jev System One for tactical decision in Dino Runner.
        """
        t0 = time.perf_counter()
        try:
            questions = {
                "tactical_action": Choice(
                    instructions="Choose the optimal immediate movement action for the Chrome Dino runner based on the oncoming obstacle type and distance",
                    criteria={
                        "jump": "Obstacle is a low cactus, cactus cluster, or low flying bird within jump proximity that must be vaulted over",
                        "duck": "Obstacle is a mid-height flying bird that requires ducking/crawling underneath",
                        "run_normal": "Horizon is clear, obstacle is still far away, or obstacle is a high flying bird that requires no avoidance maneuver",
                    },
                ),
                "threat_severity": Score(
                    instructions="Rate the immediate collision risk and urgency level from 0 to 4",
                    criteria=[
                        "0: Safe - Horizon is completely clear",
                        "1: Low - Obstacle detected at distant horizon",
                        "2: Moderate - Obstacle approaching, prepare maneuver",
                        "3: High - Obstacle within critical reaction perimeter",
                        "4: Critical - Imminent impact, execute avoidance maneuver immediately",
                    ],
                ),
                "is_fast_fall_recommended": Noul(
                    instructions="Should the runner fast-fall (duck mid-air) to return quickly to ground to prepare for an immediate subsequent obstacle?"
                ),
            }

            resp = self.client.system_one(
                state=state_dict, model=self.model_name, questions=questions
            )

            latency = (time.perf_counter() - t0) * 1000.0

            ans = resp.answers
            action_choice = getattr(ans["tactical_action"], "choice", "jump")
            action_conf = getattr(ans["tactical_action"], "confidence", 0.95)
            danger_raw = float(getattr(ans["threat_severity"], "score", 3.0))
            danger_normalized = min(1.0, danger_raw / 4.0)
            fast_fall = bool(getattr(ans["is_fast_fall_recommended"], "noul", False))

            decision = {
                "action": action_choice,
                "threat_score": danger_normalized,
                "fast_fall": fast_fall,
                "confidence": round(action_conf, 3),
                "latency_ms": round(latency, 1),
                "source": "jev_system_one",
            }
            with self._lock:
                self.last_decision = decision
            return decision

        except Exception as e:
            print(f"[JevBrain] Query error: {e}")
            return None

    def query_jev_async(self, state_dict: Dict[str, Any]):
        """
        Dispatches Jev System One query in background thread.
        """
        if self.is_querying:
            return

        def worker():
            self.is_querying = True
            try:
                self.query_jev_dino(state_dict)
            finally:
                self.is_querying = False

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

    def get_action(
        self,
        vision_action: str,
        vision_urgency: float,
        state_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Hybrid decision engine:
        Fires immediate microsecond computer vision reflexes when obstacle is in the critical strike zone,
        and uses Jev System One for tactical situational awareness.
        """
        # Trigger Jev async query if obstacle spotted
        if vision_urgency > 0.15 and not self.is_querying:
            self.query_jev_async(state_dict)

        # In critical proximity (<= 120-150px), execute reaction immediately
        if vision_urgency >= 0.75:
            return {
                "action": vision_action,
                "threat_score": vision_urgency,
                "fast_fall": False,
                "confidence": 0.98,
                "latency_ms": 0.8,
                "source": "reflex_actuator",
            }

        with self._lock:
            # If Jev has evaluated and thinks we should act
            if self.last_decision.get("threat_score", 0.0) >= 0.75:
                return self.last_decision

            # Default to vision recommendation
            return {
                "action": vision_action,
                "threat_score": vision_urgency,
                "fast_fall": False,
                "confidence": 0.92,
                "latency_ms": 1.0,
                "source": "vision_horizon",
            }
