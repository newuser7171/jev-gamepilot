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
        # Clash phase hysteresis: once in_battle, only leave on sustained non-battle evidence.
        self._clash_phase: str = ""
        self._clash_phase_streak: int = 0
        self._CLASH_MENU_STREAK = 4  # consecutive non-battle frames before main_menu

    def apply_clash_phase_hysteresis(self, raw_phase: str) -> str:
        """Stick in_battle until non-battle is confirmed for N consecutive frames.

        A single-frame false main_menu / matchmaking / game_over must not exit
        battle — each of those resets the match clock and lane counter mid-fight
        (false game_over zeroed _push_counter, pinning every cycle to right_bridge).
        All non-battle labels streak the same way when sticky phase is in_battle.
        """
        if raw_phase == "in_battle":
            self._clash_phase = "in_battle"
            self._clash_phase_streak = 0
            return raw_phase
        # Non-battle candidate (main_menu, matchmaking, game_over, other)
        if self._clash_phase == "in_battle":
            self._clash_phase_streak += 1
            if self._clash_phase_streak < self._CLASH_MENU_STREAK:
                return "in_battle"
            self._clash_phase = raw_phase
            self._clash_phase_streak = 0
            return raw_phase
        self._clash_phase = raw_phase
        self._clash_phase_streak = 0
        return raw_phase

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

            # Wide horizontal magenta elixir bar (battle HUD), not scattered
            # magenta chrome on the menu nav (that was only ~24 columns).
            col_act = magenta.sum(axis=0) > 2
            elixir_run = 0
            _run = 0
            for _a in col_act:
                _run = _run + 1 if _a else 0
                elixir_run = max(elixir_run, _run)
            elixir_bar = elixir_run > 100

            # Dense solid green Battle button (menu). Arena turf is textured
            # and fails the strict G-dominance mask used here.
            menu_box = frame_bgr[int(h * 0.76) : int(h * 0.88), int(w * 0.25) : int(w * 0.75)]
            green_btn = (
                (menu_box[:, :, 1] > 160)
                & (menu_box[:, :, 1] > menu_box[:, :, 0] + 60)
                & (menu_box[:, :, 1] > menu_box[:, :, 2] + 60)
            )
            battle_button = int(green_btn.sum()) > 5000 or int(yellow_pixels.sum()) > 2500

            # Four card slots with dark gaps between them — main-menu nav icons
            # light the gaps, so require >=2 dark gaps to call it a hand.
            hand_y1, hand_y2 = int(h * 0.895), int(h * 0.975)
            card_centers = (0.287, 0.463, 0.634, 0.806)
            gap_centers = (0.375, 0.548, 0.720)
            cards_ok = 0
            for xc in card_centers:
                reg = frame_bgr[hand_y1:hand_y2, int(w * (xc - 0.07)) : int(w * (xc + 0.07))]
                if reg.size == 0:
                    continue
                if float(cv2.cvtColor(reg, cv2.COLOR_BGR2GRAY).std()) > 25:
                    cards_ok += 1
            dark_gaps = 0
            for xc in gap_centers:
                reg = frame_bgr[hand_y1:hand_y2, int(w * (xc - 0.03)) : int(w * (xc + 0.03))]
                if reg.size == 0:
                    continue
                if float(cv2.cvtColor(reg, cv2.COLOR_BGR2GRAY).mean()) < 90:
                    dark_gaps += 1
            card_hand_present = cards_ok >= 3 and dark_gaps >= 2

            # Winner screen: gold Play Again (left) + blue OK (right) sit at
            # y≈0.83–0.88 as two large siblings (~24k/30k px). Bottom-nav icons
            # light card_hand_present there, so this pair must outrank the HUD.
            # Deck chrome / menu Battle button only produce one large blob (or
            # two small ones) — require both CCs >= 12k and on the same row.
            post_band = frame_bgr[int(h * 0.80) : int(h * 0.92), :]
            gold_left = post_band[:, : int(w * 0.50)]
            blue_right = post_band[:, int(w * 0.50) :]
            post_gold = (
                (gold_left[:, :, 2] > 160)
                & (gold_left[:, :, 1] > 130)
                & (gold_left[:, :, 0] < 140)
                & (gold_left[:, :, 2] > gold_left[:, :, 0] + 40)
            ).astype(np.uint8)
            post_blue = (
                (blue_right[:, :, 0] > 160)
                & (blue_right[:, :, 0] > blue_right[:, :, 2] + 50)
                & (blue_right[:, :, 1] > 90)
                & (blue_right[:, :, 1] < 210)
            ).astype(np.uint8)

            def _post_btn_cc(mask, x_off):
                n, _, st, _ = cv2.connectedComponentsWithStats(mask, 8)
                if n <= 1:
                    return None
                i = 1 + int(np.argmax(st[1:, 4]))
                x, y, bw, bh, area = st[i]
                if int(area) < 12000 or int(bw) < 160 or int(bh) < 70:
                    return None
                return (x_off + x + bw * 0.5, y + bh * 0.5, int(area))

            gold_cc = _post_btn_cc(post_gold, 0)
            blue_cc = _post_btn_cc(post_blue, int(w * 0.50))
            post_ok_pair = bool(
                gold_cc
                and blue_cc
                and gold_cc[0] < blue_cc[0]
                and abs(gold_cc[1] - blue_cc[1]) < 0.035 * h
            )

            # Phase priority with hysteresis: matchmaking > game_over > battle HUD > menu.
            # Mid-battle false menu (turf/button) was resetting the match clock every few frames.
            raw_phase = ""
            if red_cancel.sum() > 800:
                raw_phase = "matchmaking"
            elif post_ok_pair:
                raw_phase = "game_over"
            elif elixir_bar or card_hand_present:
                raw_phase = "in_battle"
            else:
                # Legacy fallback: broad blue band above the buttons (older layouts).
                ok_region = frame_bgr[int(h * 0.60) : int(h * 0.78), int(w * 0.28) : int(w * 0.72)]
                blue_pixels = (ok_region[:, :, 0] > 180) & (ok_region[:, :, 2] < 100) & (ok_region[:, :, 1] < 160)
                blue_n = int(blue_pixels.sum())
                blue_area = ok_region.shape[0] * ok_region.shape[1]
                if blue_n > 3500 and blue_n > 0.12 * blue_area:
                    raw_phase = "game_over"
                elif battle_button:
                    raw_phase = "main_menu"
                else:
                    # Ambiguous mid-transition — treat as battle evidence (sticky).
                    raw_phase = "in_battle"

            # Hysteresis: leaving in_battle requires N consecutive non-battle frames
            # (except matchmaking / game_over which are high-confidence UI).
            scene.game_phase = self.apply_clash_phase_hysteresis(raw_phase)
            if scene.game_phase == "in_battle":
                if elixir_bar:
                    cols = np.where(col_act)[0]
                else:
                    cols = np.where(magenta.sum(axis=0) > 2)[0]
                if len(cols) > 0:
                    scene.elixir = max(1, min(10, int((len(cols) - 55) / 73.0 + 0.5)))
                elif getattr(scene, "elixir", 0) <= 0:
                    scene.elixir = 4
            if scene.game_phase == "game_over":
                scene.recommended_action = "confirm_ok"
            elif scene.game_phase == "main_menu":
                scene.recommended_action = "start_battle"

        # 4b. Clash of Clans phase classification
        elif profile.id == "mobile_coc":
            from adapters.coc_adapter import detect_coc_phase

            raw = detect_coc_phase(frame_bgr)
            # Reuse CR hysteresis: stick in_raid until sustained non-raid evidence.
            phase_map = {
                "home_village": "main_menu",
                "attack_search": "matchmaking",
                "results": "game_over",
                "in_raid": "in_battle",
                "unknown": "",
            }
            scene.game_phase = self.apply_clash_phase_hysteresis(phase_map.get(raw, ""))
            # Expose the fine-grained CoC label for the adapter (phase_map collapses it).
            scene.coc_phase = raw
            if scene.game_phase == "game_over":
                scene.recommended_action = "confirm_ok"
            elif scene.game_phase == "main_menu":
                scene.recommended_action = "find_match"

        # 4c. Brawl Stars phase classification
        elif profile.id == "mobile_brawlstars":
            from adapters.brawl_adapter import detect_brawl_phase

            raw = detect_brawl_phase(frame_bgr)
            phase_map = {
                "menu": "main_menu",
                "matchmaking": "matchmaking",
                "results": "game_over",
                "in_match": "in_battle",
                "unknown": "",
            }
            scene.game_phase = self.apply_clash_phase_hysteresis(phase_map.get(raw, ""))
            scene.brawl_phase = raw
            if scene.game_phase == "game_over":
                scene.recommended_action = "confirm_ok"
            elif scene.game_phase == "main_menu":
                scene.recommended_action = "start_battle"

        # Specialized Clash Royale RTS Perception (Enemy Unit Health Bar Detection)
        if profile.id == "mobile_clash_royale" and scene.game_phase == "in_battle":
            # Bridge band through our half — y grows downward on portrait.
            # Starting at 0.55 missed units still crossing the river (0.47-0.55).
            friendly_y1 = int(h * 0.47)
            friendly_y2 = int(h * 0.77)
            friendly_zone = frame_bgr[friendly_y1:friendly_y2, :]

            # Enemy invader health bars: saturated crimson, wide enough to be a bar
            # not a stray pixel from effects/tower trim.
            enemy_red_mask = (friendly_zone[:, :, 2] > 180) & (friendly_zone[:, :, 1] < 75) & (friendly_zone[:, :, 0] < 75)
            red_count = int(enemy_red_mask.sum())
            if red_count > 120:
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
