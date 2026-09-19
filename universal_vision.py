"""
Universal Vision Engine for Jev-GamePilot.
Perceives any game on screen: tracks player avatars via template matching or contours,
identifies oncoming threat vectors, and pinpoints clickable targets.
"""

from dataclasses import dataclass, field
import math
import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
from profile_manager import GameProfile


@dataclass
class UniversalEntity:
    x: int
    y: int
    w: int
    h: int
    entity_type: str = "unknown"  # "player", "threat", "target", "item"
    confidence: float = 1.0
    vx: float = 0.0
    vy: float = 0.0
    distance_to_player: float = 0.0
    click_x: int = 0
    click_y: int = 0


@dataclass
class UniversalSceneState:
    player: Optional[UniversalEntity] = None
    threats: List[UniversalEntity] = field(default_factory=list)
    targets: List[UniversalEntity] = field(default_factory=list)
    nearest_threat: Optional[UniversalEntity] = None
    best_target: Optional[UniversalEntity] = None
    threat_urgency: float = 0.0
    recommended_action: str = "wait"
    is_game_over: bool = False
    game_speed: float = 0.0


class UniversalVision:
    def __init__(self):
        self.player_template: Optional[np.ndarray] = None
        self.player_template_size: Tuple[int, int] = (0, 0)
        self.last_player_pos: Optional[Tuple[int, int]] = None
        self.last_frame_gray: Optional[np.ndarray] = None
        self.last_time: float = time.time()

    def set_player_template(self, template_bgr: np.ndarray):
        """Sets user-calibrated avatar sprite template for exact tracking."""
        gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
        self.player_template = gray
        h, w = gray.shape
        self.player_template_size = (w, h)
        print(f"[UniversalVision] Player template set ({w}x{h} px)")

    def clear_player_template(self):
        self.player_template = None
        self.player_template_size = (0, 0)

    def analyze_frame(
        self, frame_bgr: np.ndarray, profile: GameProfile
    ) -> UniversalSceneState:
        now = time.time()
        dt = max(0.001, now - self.last_time)
        self.last_time = now

        h, w, _ = frame_bgr.shape
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        scene = UniversalSceneState()

        # 1. Track Player Avatar
        player_box = self._find_player(gray, w, h)
        if player_box:
            px, py, pw, ph = player_box
            vx, vy = 0.0, 0.0
            if self.last_player_pos:
                vx = (px - self.last_player_pos[0]) / dt
                vy = (py - self.last_player_pos[1]) / dt
            self.last_player_pos = (px, py)

            scene.player = UniversalEntity(
                x=px,
                y=py,
                w=pw,
                h=ph,
                entity_type="player",
                confidence=0.92,
                vx=vx,
                vy=vy,
                click_x=px + pw // 2,
                click_y=py + ph // 2,
            )
        else:
            # Fallback player location (left-center for runners, center for others)
            px = int(w * 0.15) if profile.category == "runner" else int(w * 0.45)
            py = int(h * 0.55)
            scene.player = UniversalEntity(
                x=px,
                y=py,
                w=40,
                h=40,
                entity_type="player",
                confidence=0.4,
                click_x=px + 20,
                click_y=py + 20,
            )

        p_center_x = scene.player.x + scene.player.w // 2
        p_center_y = scene.player.y + scene.player.h // 2

        # 2. Contour / Entity Detection
        # Adaptive edge & difference detection
        edges = cv2.Canny(gray, 40, 130)
        cnts, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        all_threats: List[UniversalEntity] = []
        all_targets: List[UniversalEntity] = []

        for c in cnts:
            bx, by, bw, bh = cv2.boundingRect(c)
            area = bw * bh

            # Filter out tiny noise and gigantic outer frame borders
            if area < 80 or (bw > w * 0.9 and bh > h * 0.9):
                continue

            # Ignore player's own bounding box
            if scene.player:
                overlap_x = max(
                    0,
                    min(bx + bw, scene.player.x + scene.player.w)
                    - max(bx, scene.player.x),
                )
                overlap_y = max(
                    0,
                    min(by + bh, scene.player.y + scene.player.h)
                    - max(by, scene.player.y),
                )
                if overlap_x * overlap_y > 0.4 * area:
                    continue

            center_x = bx + bw // 2
            center_y = by + bh // 2
            dist = math.hypot(center_x - p_center_x, center_y - p_center_y)

            entity = UniversalEntity(
                x=bx,
                y=by,
                w=bw,
                h=bh,
                distance_to_player=dist,
                click_x=center_x,
                click_y=center_y,
            )

            # Categorize based on game profile
            if profile.category == "clicker":
                # Targets in clicker games: compact shapes
                if 15 <= bw <= 120 and 15 <= bh <= 120:
                    entity.entity_type = "target"
                    all_targets.append(entity)
            else:
                # Platformer / Runner / Arcade: obstacles in path
                if profile.scan_direction == "right":
                    # Threats ahead of player horizontally
                    if bx > scene.player.x:
                        entity.entity_type = "threat"
                        all_threats.append(entity)
                else:
                    entity.entity_type = "threat"
                    all_threats.append(entity)

        # 3. Threat Assessment & Targeting
        if all_threats:
            all_threats.sort(key=lambda t: t.distance_to_player)
            scene.threats = all_threats
            nearest = all_threats[0]
            scene.nearest_threat = nearest

            # Threat urgency calculation (closer distance = higher urgency)
            critical_dist = (
                140.0 if profile.category == "runner" else 100.0
            )
            scene.threat_urgency = max(
                0.0, min(1.0, 1.0 - (nearest.distance_to_player / (critical_dist * 2.2)))
            )

        if all_targets:
            all_targets.sort(key=lambda t: t.distance_to_player)
            scene.targets = all_targets
            scene.best_target = all_targets[0]

        self.last_frame_gray = gray
        return scene

    def _find_player(
        self, gray_frame: np.ndarray, w: int, h: int
    ) -> Optional[Tuple[int, int, int, int]]:
        """Finds player position using template matching or focal contour."""
        # 1. Template matching if calibrated
        if self.player_template is not None:
            res = cv2.matchTemplate(
                gray_frame, self.player_template, cv2.TM_CCOEFF_NORMED
            )
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            if max_val > 0.60:
                tw, th = self.player_template_size
                return (max_loc[0], max_loc[1], tw, th)

        # 2. Heuristic fallback: player is in left-third for runners
        left_zone = gray_frame[int(h * 0.3) : int(h * 0.85), int(w * 0.05) : int(w * 0.35)]
        edges = cv2.Canny(left_zone, 50, 150)
        cnts, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        candidates = []
        for c in cnts:
            bx, by, bw, bh = cv2.boundingRect(c)
            area = bw * bh
            if 300 <= area <= 6000 and bh >= 20:
                candidates.append((area, (int(w * 0.05) + bx, int(h * 0.3) + by, bw, bh)))

        if candidates:
            candidates.sort(key=lambda item: item[0], reverse=True)
            return candidates[0][1]

        return None

    def render_debug_overlay(
        self,
        frame_bgr: np.ndarray,
        scene: UniversalSceneState,
        profile: GameProfile,
    ) -> np.ndarray:
        """Renders bounding boxes and tactical markers on any game frame."""
        annotated = frame_bgr.copy()
        h, w, _ = annotated.shape

        # Draw Player (Cyan)
        if scene.player:
            p = scene.player
            cv2.rectangle(
                annotated, (p.x, p.y), (p.x + p.w, p.y + p.h), (255, 255, 0), 2
            )
            cv2.putText(
                annotated,
                "PLAYER",
                (p.x, max(14, p.y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 0),
                1,
            )

        # Draw Threats (Red)
        for i, t in enumerate(scene.threats[:8]):
            is_nearest = i == 0
            color = (0, 0, 255) if is_nearest else (0, 165, 255)
            cv2.rectangle(
                annotated,
                (t.x, t.y),
                (t.x + t.w, t.y + t.h),
                color,
                2 if is_nearest else 1,
            )
            label = f"THREAT {int(t.distance_to_player)}px"
            cv2.putText(
                annotated,
                label,
                (t.x, max(12, t.y - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                color,
                1,
            )

        # Draw Clickable Targets (Green Crosshairs)
        for t in scene.targets[:10]:
            cv2.drawMarker(
                annotated,
                (t.click_x, t.click_y),
                (0, 255, 0),
                cv2.MARKER_CROSS,
                18,
                2,
            )
            cv2.circle(
                annotated, (t.click_x, t.click_y), max(10, t.w // 2), (0, 255, 0), 1
            )

        # Top Banner
        action_text = f"PROFILE: {profile.name} // ACTION: {scene.recommended_action.upper()}"
        cv2.putText(
            annotated,
            action_text,
            (15, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 204),
            2,
        )

        return annotated
