"""
Subway Surfers & 3-Lane Runner Adapter for Jev-GamePilot.
Divides the viewport into 3 dynamic perspective lanes (Left, Center, Right),
tracks obstacles, trains, and coins, and evaluates maneuvers.
"""

from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional
import cv2
import numpy as np


@dataclass
class LaneThreat:
    lane: str  # "left", "center", "right"
    threat_type: str  # "barrier_high", "barrier_low", "train", "clear"
    distance_px: int
    urgency: float


@dataclass
class RunnerGameState:
    current_lane: str = "center"
    player_state: str = "running"  # running, jumping, sliding
    left_threat: Optional[LaneThreat] = None
    center_threat: Optional[LaneThreat] = None
    right_threat: Optional[LaneThreat] = None
    recommended_action: str = (
        "maintain_course"  # "maintain_course", "dodge_left", "dodge_right", "jump", "slide"
    )
    threat_score: float = 0.0


class RunnerAdapter:
    def __init__(self):
        self.current_lane = "center"

    def analyze_frame(self, frame_bgr: np.ndarray) -> RunnerGameState:
        h, w, _ = frame_bgr.shape
        state = RunnerGameState(current_lane=self.current_lane)

        # Segment into 3 vertical corridors
        lane_w = w // 3
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # Inspect threat zone (middle third of screen depth)
        zone_top = int(h * 0.45)
        zone_bottom = int(h * 0.85)

        # Sample density/edges in each lane
        threats: Dict[str, LaneThreat] = {}
        for i, lane_name in enumerate(["left", "center", "right"]):
            x1 = i * lane_w
            x2 = (i + 1) * lane_w
            roi = gray[zone_top:zone_bottom, x1:x2]
            edges = cv2.Canny(roi, 50, 150)
            edge_density = float(np.mean(edges))

            if edge_density > 25.0:
                threat_type = "train" if edge_density > 45.0 else "barrier_low"
                urgency = min(1.0, edge_density / 50.0)
            else:
                threat_type = "clear"
                urgency = 0.0

            threats[lane_name] = LaneThreat(
                lane=lane_name,
                threat_type=threat_type,
                distance_px=150,
                urgency=urgency,
            )

        state.left_threat = threats["left"]
        state.center_threat = threats["center"]
        state.right_threat = threats["right"]

        # Basic heuristic fallback
        cur = threats[self.current_lane]
        if cur.urgency > 0.4:
            if cur.threat_type == "barrier_low":
                state.recommended_action = "jump"
            elif cur.threat_type == "barrier_high":
                state.recommended_action = "slide"
            else:
                # Train ahead: try dodging
                if (
                    self.current_lane == "center"
                    and threats["left"].urgency < 0.2
                ):
                    state.recommended_action = "dodge_left"
                    self.current_lane = "left"
                elif (
                    self.current_lane == "center"
                    and threats["right"].urgency < 0.2
                ):
                    state.recommended_action = "dodge_right"
                    self.current_lane = "right"
                elif (
                    self.current_lane != "center"
                    and threats["center"].urgency < 0.2
                ):
                    state.recommended_action = (
                        "dodge_right"
                        if self.current_lane == "left"
                        else "dodge_left"
                    )
                    self.current_lane = "center"
                else:
                    state.recommended_action = "jump"
            state.threat_score = cur.urgency
        else:
            state.recommended_action = "maintain_course"
            state.threat_score = 0.0

        return state
