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
    game_phase: str = "active"  # "main_menu", "in_battle", "game_over", "active"
    elixir: int = 5
    raw_frame: Optional[np.ndarray] = None
    table_detected: bool = False
    cue_ball: Optional[Tuple[int, int]] = None
    pockets: List[Tuple[int, int]] = field(default_factory=list)
    cue_ready: bool = False
    balls_moving: bool = False
    target_ball: Optional[Tuple[int, int]] = None
    ghost_ball: Optional[Tuple[int, int]] = None
    target_pocket: Optional[Tuple[int, int]] = None
    target_coords: Optional[Tuple[int, int]] = None
    cut_angle_deg: float = 0.0
    shot_power: float = 0.65


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
        is_portrait = h > w
        scene = UniversalSceneState(raw_frame=frame_bgr)

        # Compute optimal downscale factor for sub-4ms analysis (target max dim ~640)
        max_dim = max(w, h)
        scale = min(1.0, 640.0 / max(1, max_dim))
        inv_scale = 1.0 / scale if scale > 0 else 1.0

        if scale < 0.98:
            small_bgr = cv2.resize(frame_bgr, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        else:
            small_bgr = frame_bgr
        sw, sh = small_bgr.shape[1], small_bgr.shape[0]
        gray = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2GRAY)

        # 1. Track Player Avatar (in scaled space)
        player_box = self._find_player(gray, sw, sh, is_portrait, profile)
        if player_box:
            spx, spy, spw, sph = player_box
            px = int(spx * inv_scale)
            py = int(spy * inv_scale)
            pw = int(spw * inv_scale)
            ph = int(sph * inv_scale)

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
            # Fallback player location:
            # Portrait mobile runner: bottom-center (Subway Surfers, Temple Run)
            # Landscape runner: left-center (Dino, Earn to Die, Flappy)
            if is_portrait:
                px = int(w * 0.50) - 40
                py = int(h * 0.72)
            elif profile.category == "runner":
                px = int(w * 0.18)
                py = int(h * 0.55)
            else:
                px = int(w * 0.50) - 30
                py = int(h * 0.50) - 30

            scene.player = UniversalEntity(
                x=px,
                y=py,
                w=60,
                h=60,
                entity_type="player",
                confidence=0.5,
                click_x=px + 30,
                click_y=py + 30,
            )

        p_center_x = scene.player.x + scene.player.w // 2
        p_center_y = scene.player.y + scene.player.h // 2

        # 2. Contour / Entity Detection on scaled frame
        edges = cv2.Canny(gray, 40, 130)
        cnts, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        all_threats: List[UniversalEntity] = []
        all_targets: List[UniversalEntity] = []

        for c in cnts:
            sbx, sby, sbw, sbh = cv2.boundingRect(c)
            s_area = sbw * sbh

            # Filter out tiny noise and gigantic outer frame borders
            if s_area < 45 or (sbw > sw * 0.92 and sbh > sh * 0.92):
                continue

            # Project back to physical screen space
            bx = int(sbx * inv_scale)
            by = int(sby * inv_scale)
            bw = int(sbw * inv_scale)
            bh = int(sbh * inv_scale)
            area = bw * bh

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
                if overlap_x * overlap_y > 0.35 * area:
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
            if profile.category == "clicker" or "fruit" in profile.id or "solar" in profile.id:
                # Targets: compact flying or floating shapes
                if 25 <= bw <= int(w * 0.35) and 25 <= bh <= int(h * 0.35):
                    entity.entity_type = "target"
                    all_targets.append(entity)
            elif is_portrait and (profile.category == "runner" or profile.id == "runner_3lane" or profile.id == "mobile_universal"):
                # Portrait runner: obstacles are in front of player (vertically above player in screen space)
                if center_y < scene.player.y + 60:
                    entity.entity_type = "threat"
                    all_threats.append(entity)
            else:
                # Landscape runner / platformer: obstacles are horizontally ahead of player
                if profile.scan_direction == "right":
                    if center_x > scene.player.x:
                        entity.entity_type = "threat"
                        all_threats.append(entity)
                else:
                    entity.entity_type = "threat"
                    all_threats.append(entity)

        # 3. Threat Assessment & Targeting (Proportional to screen dimension)
        if is_portrait:
            critical_dist = max(250.0, h * 0.28)
        else:
            critical_dist = max(200.0, w * 0.28)

        if all_threats:
            all_threats.sort(key=lambda t: t.distance_to_player)
            scene.threats = all_threats
            nearest = all_threats[0]
            scene.nearest_threat = nearest

            # Threat urgency: 1.0 when touching critical perimeter, decaying to 0.0 at horizon
            scene.threat_urgency = max(
                0.0, min(1.0, 1.0 - (nearest.distance_to_player / (critical_dist * 1.8)))
            )

        if all_targets:
            all_targets.sort(key=lambda t: t.distance_to_player)
            scene.targets = all_targets
            scene.best_target = all_targets[0]

        # 4. Mobile Game Phase Classification (Auto Start & Post-Game Dismissal)
        if profile.id == "mobile_clash_royale":
            btn_region = frame_bgr[int(h * 0.78) : int(h * 0.86), int(w * 0.35) : int(w * 0.65)]
            yellow_pixels = (btn_region[:, :, 2] > 200) & (btn_region[:, :, 1] > 160) & (btn_region[:, :, 0] < 120)
            red_cancel = (btn_region[:, :, 2] > 170) & (btn_region[:, :, 1] < 100) & (btn_region[:, :, 0] < 100)

            # Check elixir bar in bottom HUD (active in battle)
            elixir_region = frame_bgr[int(h * 0.95) : int(h * 0.995), int(w * 0.15) : int(w * 0.98)]
            magenta = (elixir_region[:, :, 2] > 160) & (elixir_region[:, :, 0] > 160) & (elixir_region[:, :, 1] < 120)

            # Four-card hand at the bottom is a stronger in-battle signal than the
            # yellow-button heuristic, which false-fires on gold arena/card art.
            hand_y1, hand_y2 = int(h * 0.87), int(h * 0.99)
            hand_region = frame_bgr[hand_y1:hand_y2, int(w * 0.15) : int(w * 0.92)]
            hand_gray = cv2.cvtColor(hand_region, cv2.COLOR_BGR2GRAY)
            hand_edges = cv2.Canny(hand_gray, 60, 160)
            card_hand_present = (
                hand_region.size > 0
                and float(hand_edges.mean()) > 8.0
                and hand_region.shape[1] > 200
            )

            if red_cancel.sum() > 800:
                scene.game_phase = "matchmaking"
            elif magenta.sum() > 250 or card_hand_present:
                scene.game_phase = "in_battle"
                # Exact elixir calculation: measure magenta bar pixel width
                cols = np.where(magenta.sum(axis=0) > 2)[0]
                if len(cols) > 0:
                    filled_w = len(cols)
                    scene.elixir = max(1, min(10, int((filled_w - 55) / 73.0 + 0.5)))
                else:
                    scene.elixir = 4  # Default to playable elixir so it never starves
                if not scene.recommended_action or scene.recommended_action == "wait":
                    scene.recommended_action = "deploy_card_left"
            elif yellow_pixels.sum() > 2500:
                scene.game_phase = "main_menu"
                scene.recommended_action = "start_battle"
            else:
                ok_region = frame_bgr[int(h * 0.80) : int(h * 0.90), int(w * 0.35) : int(w * 0.65)]
                blue_pixels = (ok_region[:, :, 0] > 180) & (ok_region[:, :, 2] < 100)
                if blue_pixels.sum() > 800:
                    scene.game_phase = "game_over"
                    scene.recommended_action = "confirm_ok"
                else:
                    # Generic modal or reward screen — still play it like a battle
                    # so the clash idle path never dead-ends on "active".
                    scene.game_phase = "in_battle"
                    scene.recommended_action = "deploy_card_left"

            # Specialized Clash Royale RTS Perception (Enemy Unit Health Bar Detection)
            if scene.game_phase == "in_battle":
                friendly_y1 = int(h * 0.45)
                friendly_y2 = int(h * 0.77)
                friendly_zone = frame_bgr[friendly_y1:friendly_y2, :]

                # Enemy units carry crimson red health bars in our territory
                enemy_red_mask = (friendly_zone[:, :, 2] > 180) & (friendly_zone[:, :, 1] < 75) & (friendly_zone[:, :, 0] < 75)
                if enemy_red_mask.sum() > 35:
                    ry, rx = np.where(enemy_red_mask)
                    invader_x = int(np.median(rx))
                    invader_y = int(np.median(ry)) + friendly_y1
                    real_threat = UniversalEntity(
                        x=invader_x - 30,
                        y=invader_y - 20,
                        w=60,
                        h=40,
                        entity_type="threat",
                        confidence=0.96,
                        distance_to_player=math.hypot(invader_x - 540, invader_y - 1720),
                        click_x=invader_x,
                        click_y=invader_y,
                    )
                    scene.threats = [real_threat]
                    scene.nearest_threat = real_threat
                    scene.threat_urgency = 0.85
                else:
                    scene.threats = []
                    scene.nearest_threat = None
                    scene.threat_urgency = 0.0
            else:
                scene.threats = []
                scene.nearest_threat = None
                scene.threat_urgency = 0.0

        elif profile.id == "mobile_solitaire":
            # Specialized Solitaire Perception: detect Auto-Complete / Finish banner in lower region
            btn_region = frame_bgr[int(h * 0.78) : int(h * 0.90), int(w * 0.20) : int(w * 0.80)]
            green_auto = (btn_region[:, :, 1] > 175) & (btn_region[:, :, 0] < 125) & (btn_region[:, :, 2] < 125)
            blue_auto = (btn_region[:, :, 0] > 175) & (btn_region[:, :, 2] < 125)
            if green_auto.sum() > 350 or blue_auto.sum() > 350:
                scene.game_phase = "solvable"
                scene.recommended_action = "auto_complete"
                auto_target = UniversalEntity(
                    x=int(w * 0.50) - 80,
                    y=int(h * 0.84) - 30,
                    w=160,
                    h=60,
                    entity_type="target",
                    confidence=0.98,
                    click_x=int(w * 0.50),
                    click_y=int(h * 0.84),
                )
                scene.best_target = auto_target
                scene.targets = [auto_target]
        elif "8ball" in profile.id or "pool" in profile.id:
            # 8 Ball Pool Autonomous Perception & Shot Physics:
            pockets = [
                (int(w * 0.109), int(h * 0.171)),  # Top-Left
                (int(w * 0.500), int(h * 0.139)),  # Top-Mid
                (int(w * 0.891), int(h * 0.171)),  # Top-Right
                (int(w * 0.109), int(h * 0.829)),  # Bot-Left
                (int(w * 0.500), int(h * 0.861)),  # Bot-Mid
                (int(w * 0.891), int(h * 0.829)),  # Bot-Right
            ]
            scene.pockets = pockets
            scene.table_detected = True

            # 1. Table Motion Check (Are balls still rolling across the felt?)
            table_y1, table_y2 = int(h * 0.19), int(h * 0.81)
            table_x1, table_x2 = int(w * 0.15), int(w * 0.85)
            if self.last_frame_gray is not None and self.last_frame_gray.shape == gray.shape:
                cur_t = gray[table_y1:table_y2, table_x1:table_x2]
                prev_t = self.last_frame_gray[table_y1:table_y2, table_x1:table_x2]
                diff = cv2.absdiff(cur_t, prev_t)
                motion_pixels = int((diff > 25).sum())
                if motion_pixels > 1200:
                    scene.balls_moving = True
                    scene.game_phase = "balls_moving"
                    scene.recommended_action = "wait"
                    scene.threat_urgency = 0.10
                    self.last_frame_gray = gray
                    return scene

            # 2. Left Cue Stick Power Meter Check & Active Player Turn Detection
            x1, x2 = int(w * 0.070), int(w * 0.095)
            y1, y2 = int(h * 0.35), int(h * 0.85)
            crop_cue = frame_bgr[y1:y2, x1:x2]
            wood = (crop_cue[:, :, 2] > 180) & (crop_cue[:, :, 1] > 130) & (crop_cue[:, :, 0] < 150)
            cue_ready = bool(wood.sum() > 300)
            scene.cue_ready = cue_ready

            # Check player avatar timer borders
            p1_box = frame_bgr[int(h * 0.035) : int(h * 0.155), int(w * 0.395) : int(w * 0.445)]
            p2_box = frame_bgr[int(h * 0.035) : int(h * 0.155), int(w * 0.545) : int(w * 0.595)]
            p1_green = int(((p1_box[:, :, 2] < 120) & (p1_box[:, :, 1] > 180) & (p1_box[:, :, 0] < 120)).sum()) if p1_box.size > 0 else 0
            p2_green = int(((p2_box[:, :, 2] < 120) & (p2_box[:, :, 1] > 180) & (p2_box[:, :, 0] < 120)).sum()) if p2_box.size > 0 else 0
            turn_active = bool(cue_ready or p1_green > 500 or p2_green > 500)

            # Check for Ball-in-Hand notification at bottom
            bot_crop = frame_bgr[int(h * 0.94) :, int(w * 0.40) : int(w * 0.85)]
            is_ball_in_hand = False
            if bot_crop.size > 0:
                dark = (bot_crop[:, :, 0] < 50) & (bot_crop[:, :, 1] < 50) & (bot_crop[:, :, 2] < 50)
                white = (bot_crop[:, :, 0] > 200) & (bot_crop[:, :, 1] > 200) & (bot_crop[:, :, 2] > 200)
                if dark.sum() > 3500 and white.sum() > 250:
                    is_ball_in_hand = True

            if not turn_active:
                scene.game_phase = "waiting_for_turn"
                scene.recommended_action = "wait"
                scene.threat_urgency = 0.05
                self.last_frame_gray = gray
                return scene

            # 3. Detect Balls on Table Inner Felt
            table_bounds = frame_bgr[table_y1:table_y2, table_x1:table_x2]
            gray_bounds = gray[table_y1:table_y2, table_x1:table_x2]
            blurred = cv2.GaussianBlur(gray_bounds, (9, 9), 2)
            circles = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                dp=1.2,
                minDist=30,
                param1=50,
                param2=22,
                minRadius=16,
                maxRadius=28,
            )

            balls = []
            cue_ball_pos = None

            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                for cx, cy, r in circles:
                    rx, ry = cx + table_x1, cy + table_y1
                    patch = frame_bgr[max(0, ry - 16) : min(h, ry + 16), max(0, rx - 16) : min(w, rx + 16)]
                    white_px = int(((patch[:, :, 0] > 180) & (patch[:, :, 1] > 180) & (patch[:, :, 2] > 180)).sum()) if patch.size > 0 else 0
                    balls.append({
                        "pos": (rx, ry),
                        "radius": r,
                        "white_px": white_px,
                    })

            # Sort balls by white pixel density to isolate cue ball
            if balls:
                sorted_balls = sorted(balls, key=lambda b: b["white_px"], reverse=True)
                cue_ball_pos = sorted_balls[0]["pos"]
            else:
                cue_ball_pos = (int(w * 0.50), int(h * 0.50))

            scene.cue_ball = cue_ball_pos

            # If ball in hand and cue not yet placed, place it immediately
            if is_ball_in_hand and not cue_ready:
                scene.game_phase = "ball_in_hand"
                scene.recommended_action = "place_cue_ball"
                scene.target_coords = cue_ball_pos
                scene.threat_urgency = 0.85
                self.last_frame_gray = gray
                return scene

            scene.player = UniversalEntity(
                x=cue_ball_pos[0] - 16,
                y=cue_ball_pos[1] - 16,
                w=32,
                h=32,
                entity_type="cue_ball",
                click_x=cue_ball_pos[0],
                click_y=cue_ball_pos[1],
            )

            # 4. Filter Object Balls & Calculate Optimal Ghost Ball Cut Angle
            object_balls = [
                b for b in balls
                if b["pos"] != cue_ball_pos and math.hypot(b["pos"][0] - cue_ball_pos[0], b["pos"][1] - cue_ball_pos[1]) > 40
            ]

            best_shot = None
            best_score = -9999.0
            cx, cy = cue_ball_pos
            ball_r = 24

            for ob in object_balls:
                bx, by = ob["pos"]
                dist_cue_to_ball = math.hypot(bx - cx, by - cy)
                if dist_cue_to_ball < 35:
                    continue

                for px, py in pockets:
                    dist_ball_to_pocket = math.hypot(px - bx, py - by)
                    if dist_ball_to_pocket < 30:
                        continue

                    ux = (px - bx) / dist_ball_to_pocket
                    uy = (py - by) / dist_ball_to_pocket
                    gx = int(bx - ux * (ball_r * 2))
                    gy = int(by - uy * (ball_r * 2))

                    aim_dx = gx - cx
                    aim_dy = gy - cy
                    aim_dist = math.hypot(aim_dx, aim_dy)
                    if aim_dist < 10:
                        continue

                    dot = (aim_dx * ux + aim_dy * uy) / aim_dist
                    dot = max(-1.0, min(1.0, dot))
                    cut_angle_deg = math.degrees(math.acos(dot))

                    if cut_angle_deg > 75:
                        continue

                    score = 100.0 - (cut_angle_deg * 1.2) - (dist_ball_to_pocket * 0.03) - (dist_cue_to_ball * 0.02)
                    if score > best_score:
                        best_score = score
                        tot_dist = dist_cue_to_ball + dist_ball_to_pocket
                        power = 0.45 if tot_dist < 600 else (0.65 if tot_dist < 1100 else 0.85)
                        best_shot = {
                            "target_ball": (bx, by),
                            "ghost_ball": (gx, gy),
                            "pocket": (px, py),
                            "cut_angle_deg": round(cut_angle_deg, 1),
                            "power": power,
                        }

            if best_shot:
                scene.target_ball = best_shot["target_ball"]
                scene.ghost_ball = best_shot["ghost_ball"]
                scene.target_pocket = best_shot["pocket"]
                scene.target_coords = best_shot["ghost_ball"]
                scene.cut_angle_deg = best_shot["cut_angle_deg"]
                scene.shot_power = best_shot["power"]
                scene.game_phase = "aiming"
                scene.recommended_action = "execute_shot"

                aim_target = UniversalEntity(
                    x=best_shot["ghost_ball"][0] - 22,
                    y=best_shot["ghost_ball"][1] - 22,
                    w=44,
                    h=44,
                    entity_type="ghost_ball",
                    confidence=0.96,
                    click_x=best_shot["ghost_ball"][0],
                    click_y=best_shot["ghost_ball"][1],
                )
                scene.best_target = aim_target
                scene.targets = [aim_target]
                scene.threat_urgency = 0.70
            else:
                scene.game_phase = "break_ready"
                scene.recommended_action = "break_shot"
                scene.target_coords = (int(w * 0.75), int(h * 0.50))
                scene.shot_power = 1.0
                scene.threat_urgency = 0.50

        self.last_frame_gray = gray
        return scene

    def _find_player(
        self,
        gray_frame: np.ndarray,
        w: int,
        h: int,
        is_portrait: bool = False,
        profile: Optional[GameProfile] = None,
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

        # 2. Heuristic fallback based on orientation
        if is_portrait:
            # Mobile portrait runner: player is in bottom-center zone
            search_zone = gray_frame[int(h * 0.58) : int(h * 0.88), int(w * 0.25) : int(w * 0.75)]
            offset_x = int(w * 0.25)
            offset_y = int(h * 0.58)
        else:
            # Landscape runner / platformer: player is in left-third zone
            search_zone = gray_frame[int(h * 0.30) : int(h * 0.85), int(w * 0.05) : int(w * 0.35)]
            offset_x = int(w * 0.05)
            offset_y = int(h * 0.30)

        edges = cv2.Canny(search_zone, 50, 150)
        cnts, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        candidates = []
        for c in cnts:
            bx, by, bw, bh = cv2.boundingRect(c)
            area = bw * bh
            if 80 <= area <= 1500 and bh >= 12:
                candidates.append((area, (offset_x + bx, offset_y + by, bw, bh)))

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

        # 8 Ball Pool Laser Aim Guideline Overlay
        if "8ball" in profile.id or "pool" in profile.id:
            # Draw 6 pockets with cyan rings
            for px, py in scene.pockets:
                cv2.circle(annotated, (px, py), 26, (255, 200, 0), 2, cv2.LINE_AA)
                cv2.circle(annotated, (px, py), 6, (0, 255, 255), -1, cv2.LINE_AA)

            if scene.cue_ball and scene.ghost_ball:
                cb = scene.cue_ball
                gb = scene.ghost_ball
                # 1. Cue to ghost ball laser trajectory
                cv2.line(annotated, cb, gb, (0, 255, 100), 2, cv2.LINE_AA)
                # 2. Ghost ball indicator circle (ball diameter = 48)
                cv2.circle(annotated, gb, 24, (0, 255, 255), 2, cv2.LINE_AA)
                # 3. Target ball to pocket trajectory
                if scene.target_pocket and scene.target_ball:
                    tb = scene.target_ball
                    tp = scene.target_pocket
                    cv2.line(annotated, tb, tp, (0, 165, 255), 2, cv2.LINE_AA)
                    cv2.circle(annotated, tp, 18, (0, 120, 255), -1, cv2.LINE_AA)

                # Overlay status metrics
                metrics = f"AIM LOCK: {scene.cut_angle_deg}deg CUT | POWER: {int(scene.shot_power * 100)}%"
                cv2.putText(
                    annotated,
                    metrics,
                    (gb[0] - 60, max(40, gb[1] - 30)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            elif scene.balls_moving:
                cv2.putText(
                    annotated,
                    "BALLS IN MOTION // OBSERVING TABLE",
                    (int(annotated.shape[1] * 0.35), int(annotated.shape[0] * 0.50)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 165, 255),
                    2,
                    cv2.LINE_AA,
                )
            elif not scene.cue_ready:
                cv2.putText(
                    annotated,
                    "OPPONENT TURN / AWAITING CUE READY",
                    (int(annotated.shape[1] * 0.35), int(annotated.shape[0] * 0.50)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (70, 70, 220),
                    2,
                    cv2.LINE_AA,
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
