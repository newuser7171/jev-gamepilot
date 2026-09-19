"""
Chrome Dino & Edge Surf Vision Adapter for Jev-GamePilot.
Perceives the T-Rex runner, ground plane, oncoming cacti, pterodactyl birds, and game status.
"""

from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np


@dataclass
class DinoEntity:
    x: int
    y: int
    w: int
    h: int
    state: str = "running"  # running, jumping, ducking
    confidence: float = 1.0


@dataclass
class ObstacleEntity:
    x: int  # Absolute X on screen
    y: int  # Absolute Y on screen
    w: int
    h: int
    distance_from_dino: int  # Pixel distance from Dino's front
    obstacle_type: str  # cactus_small, cactus_large, cactus_cluster, bird_low, bird_mid, bird_high
    time_to_impact_ms: float = 0.0


@dataclass
class DinoGameState:
    dino: Optional[DinoEntity] = None
    obstacles: List[ObstacleEntity] = field(default_factory=list)
    nearest_obstacle: Optional[ObstacleEntity] = None
    ground_y: int = 0
    is_game_over: bool = False
    is_night_mode: bool = False
    game_speed_px_sec: float = 300.0
    threat_score: float = 0.0
    recommended_action: str = "run_normal"  # run_normal, jump, duck, restart
    action_urgency: float = 0.0


class DinoAdapter:
    def __init__(self):
        self.last_nearest_x: Optional[int] = None
        self.last_frame_time: float = time.time()
        self.ground_y: int = 0
        self.base_dino_w: int = 44
        self.base_dino_h: int = 47
        self.speed_history: List[float] = []

    def analyze_frame(self, frame_bgr: np.ndarray) -> DinoGameState:
        """
        Takes raw captured BGR frame and extracts full game telemetry.
        """
        now = time.time()
        dt = max(0.001, now - self.last_frame_time)
        self.last_frame_time = now

        h, w, _ = frame_bgr.shape
        state = DinoGameState()

        # 1. Detect Night Mode vs Day Mode
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        top_sample = gray[: int(h * 0.2), :]
        mean_brightness = float(np.mean(top_sample))
        is_night_mode = mean_brightness < 120
        state.is_night_mode = is_night_mode

        # 2. Binary Thresholding: foreground = 255, background = 0
        if is_night_mode:
            _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        else:
            _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

        # 3. Detect Ground Plane
        # The ground plane is typically in the lower 40-80% region
        horizontal_proj = np.sum(thresh[int(h * 0.4) : int(h * 0.9), :], axis=1)
        if len(horizontal_proj) > 0 and np.max(horizontal_proj) > w * 0.2 * 255:
            ground_rel_idx = int(np.argmax(horizontal_proj))
            self.ground_y = int(h * 0.4) + ground_rel_idx
        elif self.ground_y == 0:
            self.ground_y = int(h * 0.70)
        state.ground_y = self.ground_y

        # 4. Check for Game Over (Restart icon in center)
        center_zone = thresh[
            int(h * 0.25) : int(h * 0.65), int(w * 0.35) : int(w * 0.65)
        ]
        cnts_center, _ = cv2.findContours(
            center_zone, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for c in cnts_center:
            cw, ch = cv2.boundingRect(c)[2:]
            # Circular restart button is roughly 30x30 to 45x45
            if 25 <= cw <= 60 and 25 <= ch <= 60 and abs(cw - ch) < 15:
                state.is_game_over = True
                state.recommended_action = "restart"
                state.threat_score = 0.0
                return state

        # 5. Detect Dino
        # Dino is in the left 5% - 35% of the screen, standing near ground
        dino_zone_left = int(w * 0.02)
        dino_zone_right = int(w * 0.35)
        dino_zone_top = max(0, self.ground_y - int(h * 0.5))
        dino_zone_bottom = min(h, self.ground_y + 15)

        dino_zone = thresh[
            dino_zone_top:dino_zone_bottom, dino_zone_left:dino_zone_right
        ]
        cnts_dino, _ = cv2.findContours(
            dino_zone, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        dino_box: Optional[Tuple[int, int, int, int]] = None
        max_area = 0
        for c in cnts_dino:
            bx, by, bw, bh = cv2.boundingRect(c)
            area = bw * bh
            # Filter noise; Dino should have substantial area (> 300 px)
            if area > 350 and area > max_area and bh >= 20:
                max_area = area
                dino_box = (
                    dino_zone_left + bx,
                    dino_zone_top + by,
                    bw,
                    bh,
                )

        if dino_box:
            dx, dy, dw, dh = dino_box
            dino_bottom = dy + dh
            # Determine jumping/ducking status
            if dino_bottom < self.ground_y - 18:
                dino_state = "jumping"
            elif dh < 30 and dw > 50:
                dino_state = "ducking"
            else:
                dino_state = "running"
            state.dino = DinoEntity(
                x=dx, y=dy, w=dw, h=dh, state=dino_state, confidence=0.95
            )
        else:
            # Fallback estimation if Dino is momentarily occluded
            fallback_x = int(w * 0.12)
            fallback_w = 44
            fallback_h = 47
            fallback_y = self.ground_y - fallback_h
            state.dino = DinoEntity(
                x=fallback_x,
                y=fallback_y,
                w=fallback_w,
                h=fallback_h,
                state="running",
                confidence=0.5,
            )

        # 6. Scan Threat Horizon (Obstacles)
        # Search from front of Dino to ~70% screen width
        dino_front = state.dino.x + state.dino.w
        scan_left = dino_front + 2
        scan_right = min(w, dino_front + int(w * 0.65))
        scan_top = max(0, self.ground_y - int(h * 0.45))
        scan_bottom = min(h, self.ground_y + 10)

        scan_roi = thresh[scan_top:scan_bottom, scan_left:scan_right]
        cnts_obs, _ = cv2.findContours(
            scan_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        detected_obstacles: List[ObstacleEntity] = []
        for c in cnts_obs:
            ox, oy, ow, oh = cv2.boundingRect(c)
            # Filter out tiny ground specks and clouds
            if ow < 8 or oh < 12:
                continue

            abs_x = scan_left + ox
            abs_y = scan_top + oy
            distance = abs_x - dino_front

            # Classify obstacle type
            obs_bottom = abs_y + oh
            if obs_bottom >= self.ground_y - 15:
                # Ground level -> Cactus
                if ow < 25:
                    obs_type = "cactus_small"
                elif ow < 55:
                    obs_type = "cactus_large"
                else:
                    obs_type = "cactus_cluster"
            else:
                # Floating above ground -> Pterodactyl Bird
                rel_height_above_ground = self.ground_y - obs_bottom
                if rel_height_above_ground > 38:
                    obs_type = "bird_high"  # Safe to run under
                elif rel_height_above_ground > 14:
                    obs_type = "bird_mid"  # Must duck!
                else:
                    obs_type = "bird_low"  # Must jump!

            detected_obstacles.append(
                ObstacleEntity(
                    x=abs_x,
                    y=abs_y,
                    w=ow,
                    h=oh,
                    distance_from_dino=distance,
                    obstacle_type=obs_type,
                )
            )

        # Sort obstacles by distance
        detected_obstacles.sort(key=lambda o: o.distance_from_dino)
        state.obstacles = detected_obstacles

        # 7. Speed and Threat Urgency Evaluation
        if detected_obstacles:
            nearest = detected_obstacles[0]
            state.nearest_obstacle = nearest

            # Calculate speed if we tracked the same obstacle
            if self.last_nearest_x is not None:
                dx = self.last_nearest_x - nearest.x
                if 2 <= dx <= 40:  # Reasonable delta per frame
                    instant_speed = dx / dt
                    self.speed_history.append(instant_speed)
                    if len(self.speed_history) > 10:
                        self.speed_history.pop(0)
            self.last_nearest_x = nearest.x

            if self.speed_history:
                state.game_speed_px_sec = float(np.median(self.speed_history))
            else:
                state.game_speed_px_sec = 400.0

            # Time to impact
            speed = max(150.0, state.game_speed_px_sec)
            time_to_impact = (nearest.distance_from_dino / speed) * 1000.0
            nearest.time_to_impact_ms = time_to_impact

            # Tactical Threat Decision
            dist = nearest.distance_from_dino
            # Calibrated jump window: trigger jump when obstacle is within ~120 to 190 pixels depending on speed
            jump_threshold = max(115, int(speed * 0.28))
            duck_threshold = max(130, int(speed * 0.32))

            if nearest.obstacle_type == "bird_high":
                # High bird: no action needed
                state.recommended_action = "run_normal"
                state.threat_score = 0.1
                state.action_urgency = 0.0
            elif nearest.obstacle_type == "bird_mid":
                # Mid bird: duck!
                if dist <= duck_threshold:
                    state.recommended_action = "duck"
                    state.threat_score = 0.95
                    state.action_urgency = 1.0
                else:
                    state.recommended_action = "run_normal"
                    state.threat_score = 0.4
                    state.action_urgency = max(0.0, 1.0 - (dist / 350.0))
            else:
                # Cactus or low bird: jump!
                if dist <= jump_threshold:
                    state.recommended_action = "jump"
                    state.threat_score = 0.98
                    state.action_urgency = 1.0
                else:
                    state.recommended_action = "run_normal"
                    state.threat_score = 0.3
                    state.action_urgency = max(0.0, 1.0 - (dist / 300.0))
        else:
            self.last_nearest_x = None
            state.recommended_action = "run_normal"
            state.threat_score = 0.0
            state.action_urgency = 0.0

        return state

    def render_debug_overlay(
        self, frame_bgr: np.ndarray, state: DinoGameState
    ) -> np.ndarray:
        """
        Draws visual perception telemetry directly onto the frame for HUD preview.
        """
        annotated = frame_bgr.copy()
        h, w, _ = annotated.shape

        # Draw Ground Plane (Cyan)
        if state.ground_y > 0:
            cv2.line(
                annotated,
                (0, state.ground_y),
                (w, state.ground_y),
                (255, 255, 0),
                2,
            )

        # Draw Dino (Green if running/ducking, Yellow if airborne)
        if state.dino:
            d = state.dino
            d_color = (0, 255, 255) if d.state == "jumping" else (0, 255, 0)
            cv2.rectangle(
                annotated, (d.x, d.y), (d.x + d.w, d.y + d.h), d_color, 2
            )
            cv2.putText(
                annotated,
                f"DINO: {d.state.upper()}",
                (d.x, max(15, d.y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                d_color,
                1,
            )

        # Draw Scan Horizon (Dashed line or bounding region)
        if state.dino:
            front = state.dino.x + state.dino.w
            cv2.rectangle(
                annotated,
                (front + 2, max(0, state.ground_y - int(h * 0.45))),
                (min(w, front + int(w * 0.65)), min(h, state.ground_y + 10)),
                (80, 80, 80),
                1,
            )

        # Draw Obstacles (Red / Magenta)
        for i, obs in enumerate(state.obstacles):
            is_nearest = i == 0
            color = (0, 0, 255) if is_nearest else (255, 0, 255)
            cv2.rectangle(
                annotated,
                (obs.x, obs.y),
                (obs.x + obs.w, obs.y + obs.h),
                color,
                2 if is_nearest else 1,
            )
            label = f"{obs.obstacle_type} ({obs.distance_from_dino}px)"
            cv2.putText(
                annotated,
                label,
                (obs.x, max(12, obs.y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
            )

        # Draw Action Banner
        action_text = f"ACTION: {state.recommended_action.upper()}"
        act_color = (
            (0, 255, 0)
            if state.recommended_action == "run_normal"
            else (0, 165, 255)
        )
        if state.recommended_action in ["jump", "duck"]:
            act_color = (0, 0, 255)
        cv2.putText(
            annotated,
            action_text,
            (15, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            act_color,
            2,
        )

        return annotated
