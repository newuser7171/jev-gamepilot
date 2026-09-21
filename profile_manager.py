"""
Profile Manager for Universal Jev-GamePilot.
Manages game configurations, keybindings, vision parameters, and Jev System One objectives.
"""

from dataclasses import asdict, dataclass, field
import json
import os
from typing import Any, Dict, List, Optional


@dataclass
class GameAction:
    name: str  # e.g. "jump", "dodge_left", "click_target"
    key: str  # e.g. "space", "left", "a", "mouse_left"
    description: str  # Semantic description fed to Jev Choice criteria
    duration_sec: float = 0.10  # Key hold duration
    is_mouse_click: bool = False


@dataclass
class GameProfile:
    id: str
    name: str
    category: str  # "runner", "platformer", "clicker", "arcade", "custom"
    description: str
    actions: List[GameAction] = field(default_factory=list)
    default_window_keyword: str = ""
    scan_direction: str = "right"  # "right", "left", "omnidirectional", "click_targets"
    auto_restart_key: str = "space"
    is_builtin: bool = False


DEFAULT_PROFILES: List[GameProfile] = [
    GameProfile(
        id="runner_dino",
        name="🦖 Chrome Dino & Obstacle Runner",
        category="runner",
        description="Endless runner dodging ground cacti and aerial birds",
        default_window_keyword="dino",
        scan_direction="right",
        auto_restart_key="space",
        is_builtin=True,
        actions=[
            GameAction(
                name="jump",
                key="space",
                description="Vault over oncoming low cacti, clusters, or low-flying birds",
                duration_sec=0.14,
            ),
            GameAction(
                name="duck",
                key="down",
                description="Duck/slide under mid-altitude flying pterodactyl birds",
                duration_sec=0.35,
            ),
            GameAction(
                name="run_normal",
                key="none",
                description="Horizon is clear or obstacle is harmless high bird",
            ),
        ],
    ),
    GameProfile(
        id="runner_3lane",
        name="🏄 3-Lane Runner (Subway Surfers / Mobile)",
        category="runner",
        description="3-lane runner with barriers, trains, and coins",
        default_window_keyword="subway",
        scan_direction="omnidirectional",
        auto_restart_key="space",
        is_builtin=True,
        actions=[
            GameAction(
                name="dodge_left",
                key="left",
                description="Shift one lane to the left to avoid a train or obstacle",
                duration_sec=0.08,
            ),
            GameAction(
                name="dodge_right",
                key="right",
                description="Shift one lane to the right to avoid a train or obstacle",
                duration_sec=0.08,
            ),
            GameAction(
                name="jump",
                key="up",
                description="Leap over low hurdles, roadblocks, or jump on train roofs",
                duration_sec=0.12,
            ),
            GameAction(
                name="slide",
                key="down",
                description="Slide under high overhead tunnel signs or barriers",
                duration_sec=0.30,
            ),
            GameAction(
                name="hoverboard",
                key="space",
                description="Double-tap screen to activate invincibility hoverboard shield",
                duration_sec=0.08,
            ),
            GameAction(
                name="fast_fall",
                key="down",
                description="Cancel jump arc in mid-air to land immediately and slide under barrier",
                duration_sec=0.15,
            ),
            GameAction(
                name="maintain_course",
                key="none",
                description="Current lane is open and safe",
            ),
        ],
    ),
    GameProfile(
        id="flappy_tap",
        name="🐦 Flappy Bird & One-Tap Jumper",
        category="arcade",
        description="Single-button rhythm and obstacle navigation jumper",
        default_window_keyword="flappy",
        scan_direction="right",
        auto_restart_key="space",
        is_builtin=True,
        actions=[
            GameAction(
                name="flap",
                key="space",
                description="Flap wings to gain altitude and clear pipes or gaps",
                duration_sec=0.06,
            ),
            GameAction(
                name="glide",
                key="none",
                description="Descend naturally through gap without flapping",
            ),
        ],
    ),
    GameProfile(
        id="chess_copilot",
        name="♟️ Chess.com & Lichess Screen Copilot",
        category="strategy",
        description="Analyzes 8x8 chessboard on screen, evaluates tactical moves, and plays or highlights squares",
        default_window_keyword="chess",
        scan_direction="omnidirectional",
        auto_restart_key="none",
        is_builtin=True,
        actions=[
            GameAction(
                name="play_best_move",
                key="mouse_left",
                description="Execute Jev recommended move by clicking start and target squares",
                is_mouse_click=True,
            ),
            GameAction(
                name="analyze_only",
                key="none",
                description="Calculate and display optimal move overlay without sending clicks",
            ),
        ],
    ),
    GameProfile(
        id="aim_clicker",
        name="🎯 Target Clicker & Aim Trainer (Osu / Aim Labs)",
        category="clicker",
        description="Fast reflex targeting clicking active targets and objects",
        default_window_keyword="aim",
        scan_direction="click_targets",
        auto_restart_key="space",
        is_builtin=True,
        actions=[
            GameAction(
                name="click_target",
                key="mouse_left",
                description="Immediately click the highest priority target on screen",
                is_mouse_click=True,
            ),
            GameAction(
                name="wait",
                key="none",
                description="No active targets in view",
            ),
        ],
    ),
    GameProfile(
        id="retro_platformer",
        name="🕹️ 2D Platformer / Retro Arcade",
        category="platformer",
        description="Side-scrolling platformer with running, jumping, and attacking",
        default_window_keyword="game",
        scan_direction="right",
        auto_restart_key="enter",
        is_builtin=True,
        actions=[
            GameAction(
                name="move_right",
                key="d",
                description="Advance forward towards the right",
                duration_sec=0.25,
            ),
            GameAction(
                name="move_left",
                key="a",
                description="Retreat left away from an approaching hazard",
                duration_sec=0.15,
            ),
            GameAction(
                name="jump",
                key="space",
                description="Jump over pit, platform, or enemy",
                duration_sec=0.18,
            ),
            GameAction(
                name="attack",
                key="j",
                description="Attack or shoot nearby enemy",
                duration_sec=0.08,
            ),
            GameAction(
                name="stand_idle",
                key="none",
                description="Wait for moving platform or hazard to pass",
            ),
        ],
    ),
    GameProfile(
        id="mobile_fruit_ninja",
        name="🍉 Fruit Ninja & Slice Arcades",
        category="clicker",
        description="Tracks flying fruit and slices across centroids while evading bombs",
        default_window_keyword="fruit",
        scan_direction="click_targets",
        auto_restart_key="space",
        is_builtin=True,
        actions=[
            GameAction(
                name="slice_target",
                key="mouse_left",
                description="Perform rapid slice across flying target centroid",
                is_mouse_click=True,
            ),
            GameAction(
                name="wait",
                key="none",
                description="No fruit on screen or hazard bomb in trajectory",
            ),
        ],
    ),
    GameProfile(
        id="mobile_earntodie2",
        name="🚗 Earn to Die 2 / 2D Physics Drivers",
        category="platformer",
        description="2D vehicle runner with gas throttle, nitro boost, and tilt balance",
        default_window_keyword="earntodie",
        scan_direction="right",
        auto_restart_key="space",
        is_builtin=True,
        actions=[
            GameAction(
                name="accelerate",
                key="d",
                description="Apply main throttle pedal to drive vehicle forward",
                duration_sec=0.25,
            ),
            GameAction(
                name="boost",
                key="w",
                description="Fire nitro thruster boost through large zombie hordes or steep inclines",
                duration_sec=0.15,
            ),
            GameAction(
                name="tilt_forward",
                key="right",
                description="Tilt vehicle nose downward to maintain ground traction",
                duration_sec=0.10,
            ),
            GameAction(
                name="tilt_back",
                key="left",
                description="Tilt vehicle nose upward to prevent catastrophic rollover",
                duration_sec=0.10,
            ),
        ],
    ),
    GameProfile(
        id="mobile_universal",
        name="📱 Universal Mobile AI (Auto-Adapts to ANY Game)",
        category="runner",
        description="Universal autonomous agent: dynamically executes 4-way swipes, reflex jumps, timed taps, or target clicks based on active screen state",
        default_window_keyword="",
        scan_direction="omnidirectional",
        auto_restart_key="space",
        is_builtin=True,
        actions=[
            GameAction(
                name="dodge_left",
                key="left",
                description="Swipe left to evade obstacles or shift to open left path",
                duration_sec=0.08,
            ),
            GameAction(
                name="dodge_right",
                key="right",
                description="Swipe right to evade obstacles or shift to open right path",
                duration_sec=0.08,
            ),
            GameAction(
                name="jump",
                key="up",
                description="Swipe up or tap to jump over oncoming ground hurdles or pits",
                duration_sec=0.12,
            ),
            GameAction(
                name="slide",
                key="down",
                description="Swipe down to duck or slide underneath overhead barriers",
                duration_sec=0.30,
            ),
            GameAction(
                name="tap",
                key="space",
                description="Tap screen or target to trigger rhythm timing or click object",
                duration_sec=0.05,
            ),
            GameAction(
                name="maintain_course",
                key="none",
                description="Current path is clear, maintain forward momentum",
            ),
        ],
    ),
    GameProfile(
        id="mobile_clash_royale",
        name="👑 Clash Royale & Tower RTS",
        category="strategy",
        description="Analyzes arena state, deploys defense troops and win-conditions at bridges, launches spells against enemy towers",
        default_window_keyword="clashroyale",
        scan_direction="omnidirectional",
        auto_restart_key="space",
        is_builtin=True,
        actions=[
            GameAction(
                name="deploy_card_left",
                key="mouse_left",
                description="Select active card from bottom deck and deploy into left lane to attack or defend",
                is_mouse_click=True,
            ),
            GameAction(
                name="deploy_card_right",
                key="mouse_left",
                description="Select active card from bottom deck and deploy into right lane to attack or defend",
                is_mouse_click=True,
            ),
            GameAction(
                name="deploy_spell_center",
                key="mouse_left",
                description="Cast direct damage spell at enemy tower or cluster",
                is_mouse_click=True,
            ),
            GameAction(
                name="deploy_defense_center",
                key="mouse_left",
                description="Plant defensive building or troops in golden central pocket to draw both lanes",
                is_mouse_click=True,
            ),
            GameAction(
                name="wait",
                key="none",
                description="Preserve and generate elixir for high-cost counter-push",
            ),
        ],
    ),
    GameProfile(
        id="mobile_fifa",
        name="⚽ EA Sports FC / FIFA Mobile",
        category="platformer",
        description="Executes ball control: sprint tackle, passes to open wings, and power shots on goal",
        default_window_keyword="fifamobile",
        scan_direction="right",
        auto_restart_key="space",
        is_builtin=True,
        actions=[
            GameAction(
                name="sprint_tackle",
                key="d",
                description="Press and hold sprint / tackle button to close down opposing attacker",
                duration_sec=0.20,
            ),
            GameAction(
                name="pass",
                key="s",
                description="Execute ground pass to open midfield teammate",
                duration_sec=0.10,
            ),
            GameAction(
                name="through_pass",
                key="w",
                description="Send through ball into space behind enemy defensive line",
                duration_sec=0.10,
            ),
            GameAction(
                name="shoot_goal",
                key="space",
                description="Calibrated power strike (~65% power bar charge)",
                duration_sec=0.18,
            ),
            GameAction(
                name="finesse_shot",
                key="down",
                description="Flick down on shoot button for curling far-post finesse strike",
                duration_sec=0.15,
            ),
            GameAction(
                name="chip_shot",
                key="up",
                description="Flick up on shoot button to lob the oncoming goalkeeper",
                duration_sec=0.15,
            ),
            GameAction(
                name="power_shot",
                key="right",
                description="Flick right across shoot button for high velocity rocket strike",
                duration_sec=0.20,
            ),
            GameAction(
                name="skill_move",
                key="up",
                description="Flick upward on sprint & skill button to execute 5-star skill move",
                duration_sec=0.15,
            ),
            GameAction(
                name="dribble_forward",
                key="right",
                description="Engage virtual joystick forward to advance down the wing towards enemy goal",
                duration_sec=0.25,
            ),
            GameAction(
                name="dribble_cut_inside",
                key="up",
                description="Diagonal joystick cut inside towards the penalty box",
                duration_sec=0.25,
            ),
            GameAction(
                name="maintain_course",
                key="none",
                description="Dribble forward in open space",
            ),
        ],
    ),
    GameProfile(
        id="mobile_solarsmash",
        name="🪐 Solar Smash Planetary Sandbox",
        category="clicker",
        description="Fires orbital lasers, celestial weapons, and kinetic projectiles at targeted planet regions",
        default_window_keyword="solarsmash",
        scan_direction="click_targets",
        auto_restart_key="space",
        is_builtin=True,
        actions=[
            GameAction(
                name="fire_laser",
                key="mouse_left",
                description="Engage continuous superheated core piercing laser",
                is_mouse_click=True,
            ),
            GameAction(
                name="orbital_strike",
                key="mouse_left",
                description="Deploy satellite kinetic bombardments on planetary crust",
                is_mouse_click=True,
            ),
            GameAction(
                name="launch_meteor",
                key="mouse_left",
                description="Direct asteroid cluster impact at targeted coordinates",
                is_mouse_click=True,
            ),
            GameAction(
                name="rotate_planet",
                key="left",
                description="Rotate planet coordinate sphere to target intact continental sectors",
                duration_sec=0.15,
            ),
        ],
    ),
    GameProfile(
        id="mobile_bitlife",
        name="📖 BitLife & Choice Simulators",
        category="strategy",
        description="Reads life events, hits age progression button, and selects optimal choice answers",
        default_window_keyword="bitlife",
        scan_direction="omnidirectional",
        auto_restart_key="space",
        is_builtin=True,
        actions=[
            GameAction(
                name="age_up",
                key="mouse_left",
                description="Tap large green Age progression button to advance life year",
                is_mouse_click=True,
            ),
            GameAction(
                name="primary_choice",
                key="mouse_left",
                description="Select favorable primary choice button in scenario popup",
                is_mouse_click=True,
            ),
            GameAction(
                name="secondary_choice",
                key="mouse_left",
                description="Select secondary alternative choice in scenario popup",
                is_mouse_click=True,
            ),
            GameAction(
                name="dismiss_popup",
                key="mouse_left",
                description="Dismiss modal or milestone notification",
                is_mouse_click=True,
            ),
        ],
    ),
]


class ProfileManager:
    def __init__(self, profiles_path: str = "profiles.json"):
        self.profiles_path = profiles_path
        self.profiles: Dict[str, GameProfile] = {}
        self.load_profiles()

    def load_profiles(self):
        """Loads default presets and any user custom profiles."""
        self.profiles = {p.id: p for p in DEFAULT_PROFILES}

        if os.path.exists(self.profiles_path):
            try:
                with open(self.profiles_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for item in data:
                        actions = [GameAction(**a) for a in item.get("actions", [])]
                        item["actions"] = actions
                        profile = GameProfile(**item)
                        self.profiles[profile.id] = profile
            except Exception as e:
                print(f"[ProfileManager] Error loading profiles.json: {e}")

    def save_custom_profiles(self):
        """Persists custom profiles to JSON."""
        custom_list = [
            asdict(p) for p in self.profiles.values() if not p.is_builtin
        ]
        try:
            with open(self.profiles_path, "w", encoding="utf-8") as f:
                json.dump(custom_list, f, indent=2)
        except Exception as e:
            print(f"[ProfileManager] Error saving profiles: {e}")

    def list_profiles(self) -> List[GameProfile]:
        return list(self.profiles.values())

    def get_profile(self, profile_id: str) -> Optional[GameProfile]:
        return self.profiles.get(profile_id)

    def add_custom_profile(
        self,
        profile_id: str,
        name: str,
        category: str,
        description: str,
        actions: List[GameAction],
        window_keyword: str = "",
        scan_direction: str = "right",
    ) -> GameProfile:
        prof = GameProfile(
            id=profile_id,
            name=name,
            category=category,
            description=description,
            actions=actions,
            default_window_keyword=window_keyword,
            scan_direction=scan_direction,
            is_builtin=False,
        )
        self.profiles[profile_id] = prof
        self.save_custom_profiles()
        return prof

    def delete_profile(self, profile_id: str) -> bool:
        if profile_id in self.profiles and not self.profiles[profile_id].is_builtin:
            del self.profiles[profile_id]
            self.save_custom_profiles()
            return True
        return False
