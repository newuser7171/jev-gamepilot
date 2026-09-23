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
    icon: str = ""


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
                name="deploy_clash_card",
                key="mouse_left",
                description="Precise two-step deploy: tap card slot then target square from strategy (4-tuple coords)",
                is_mouse_click=True,
            ),
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
                name="start_battle",
                key="mouse_left",
                description="Tap the yellow Battle button on the main menu to queue a match",
                is_mouse_click=True,
            ),
            GameAction(
                name="confirm_ok",
                key="mouse_left",
                description="Tap OK / center to dismiss post-game screens and collect rewards",
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
        id="mobile_coc",
        name="🏰 Clash of Clans Raids",
        category="strategy",
        description="Autonomous CoC raider: find a match, spread-deploy the full army around the enemy base, end for loot",
        default_window_keyword="clashofclans",
        scan_direction="omnidirectional",
        auto_restart_key="space",
        is_builtin=True,
        actions=[
            GameAction(
                name="find_match",
                key="mouse_left",
                description="Tap the Attack / Find a Match button on the home village",
                is_mouse_click=True,
            ),
            GameAction(
                name="deploy_troop",
                key="mouse_left",
                description="Select army-bar slot then tap a perimeter point around the enemy base (4-tuple coords)",
                is_mouse_click=True,
            ),
            GameAction(
                name="end_battle",
                key="mouse_left",
                description="End the raid after the army has engaged to bank percentage loot",
                is_mouse_click=True,
            ),
            GameAction(
                name="confirm_ok",
                key="mouse_left",
                description="Dismiss loot / victory screens and return to the home village",
                is_mouse_click=True,
            ),
            GameAction(
                name="collect_resources",
                key="mouse_left",
                description="Tap gold mines and elixir collectors on the home village",
                is_mouse_click=True,
            ),
            GameAction(
                name="wait",
                key="none",
                description="Hold position — matchmaking, troop march, or results transition",
            ),
        ],
    ),
    GameProfile(
        id="mobile_brawlstars",
        name="⭐ Brawl Stars",
        category="arcade",
        description="Landscape arena fighter: joystick roam on objectives, aimed attacks on red enemies, super when charged",
        default_window_keyword="brawlstars",
        scan_direction="omnidirectional",
        auto_restart_key="space",
        is_builtin=True,
        actions=[
            GameAction(
                name="move_to",
                key="mouse_left",
                description="Drag the left virtual joystick toward a normalized roam point (4-tuple base→target)",
                is_mouse_click=True,
            ),
            GameAction(
                name="attack",
                key="mouse_left",
                description="Aim joystick at enemy centroid then tap the right-hand attack button (5-tuple)",
                is_mouse_click=True,
            ),
            GameAction(
                name="use_super",
                key="mouse_left",
                description="Tap the super button when the gold charge ring is full",
                is_mouse_click=True,
            ),
            GameAction(
                name="start_battle",
                key="mouse_left",
                description="Tap Play / event button on the main menu to queue a match",
                is_mouse_click=True,
            ),
            GameAction(
                name="confirm_ok",
                key="mouse_left",
                description="Dismiss victory / defeat results and return to the menu",
                is_mouse_click=True,
            ),
            GameAction(
                name="wait",
                key="none",
                description="Matchmaking or attack cadence cooldown — no touch",
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
    GameProfile(
        id="mobile_card_battler",
        name="🃏 Card Games, TCGs & Deckbuilders",
        category="strategy",
        description="Autonomous card player for Marvel SNAP, Pokémon TCG, Hearthstone, MTG Arena, Yu-Gi-Oh, Balatro, Slay the Spire, and Solitaire with drag-to-play, minion attacks, and turn passing",
        default_window_keyword="card",
        scan_direction="omnidirectional",
        auto_restart_key="space",
        is_builtin=True,
        actions=[
            GameAction(
                name="play_card_center",
                key="mouse_left",
                description="Drag active card from bottom hand tray into center battlefield or middle location",
                is_mouse_click=True,
            ),
            GameAction(
                name="play_card_left",
                key="mouse_left",
                description="Drag card from hand tray into left lane or drop zone",
                is_mouse_click=True,
            ),
            GameAction(
                name="play_card_right",
                key="mouse_left",
                description="Drag card from hand tray into right lane or drop zone",
                is_mouse_click=True,
            ),
            GameAction(
                name="attack_face",
                key="mouse_left",
                description="Direct friendly minion attack arrow directly at enemy hero / face",
                is_mouse_click=True,
            ),
            GameAction(
                name="attack_minion",
                key="mouse_left",
                description="Direct friendly minion attack arrow to trade into enemy frontline creature",
                is_mouse_click=True,
            ),
            GameAction(
                name="hero_power",
                key="mouse_left",
                description="Activate Leader ability, Hero Power, or trigger Cosmic SNAP Cube",
                is_mouse_click=True,
            ),
            GameAction(
                name="end_turn",
                key="space",
                description="Tap End Turn / Done / Pass button to pass priority to opponent",
            ),
            GameAction(
                name="balatro_play_hand",
                key="enter",
                description="Trigger Balatro Play Hand button to score played poker combination",
            ),
            GameAction(
                name="balatro_discard",
                key="backspace",
                description="Trigger Balatro Discard button to cycle unneeded cards",
            ),
            GameAction(
                name="select_card",
                key="mouse_left",
                description="Tap card in hand to inspect, select, or queue for play",
                is_mouse_click=True,
            ),
            GameAction(
                name="confirm_choice",
                key="enter",
                description="Confirm battle prompt, mulligan choice, or claim victory rewards",
            ),
            GameAction(
                name="wait",
                key="none",
                description="Observe opponent turn, evaluate board state, and accumulate energy/mana",
            ),
        ],
    ),
    GameProfile(
        id="mobile_solitaire",
        name="♠️ Solitaire & Classic Card Puzzles",
        category="strategy",
        description="Autonomous pilot for Klondike, Spider, and FreeCell Solitaire: scans tableau columns, auto-moves cards to foundations, draws from stock, and executes column transfers",
        default_window_keyword="solitaire",
        scan_direction="omnidirectional",
        auto_restart_key="space",
        is_builtin=True,
        actions=[
            GameAction(
                name="tap_tableau_column",
                key="mouse_left",
                description="Tap exposed card of tableau column to trigger auto-move to foundation or build sequence",
                is_mouse_click=True,
            ),
            GameAction(
                name="draw_stock",
                key="mouse_left",
                description="Tap stock deck in top corner to deal fresh cards to waste pile",
                is_mouse_click=True,
            ),
            GameAction(
                name="tap_waste_card",
                key="mouse_left",
                description="Tap waste pile card to auto-move to foundation or build onto tableau",
                is_mouse_click=True,
            ),
            GameAction(
                name="drag_column_transfer",
                key="mouse_left",
                description="Drag card sequence from one tableau column to another valid destination column",
                is_mouse_click=True,
            ),
            GameAction(
                name="auto_complete",
                key="mouse_left",
                description="Tap Auto-Finish / Auto-Complete button when all cards are face-up",
                is_mouse_click=True,
            ),
            GameAction(
                name="sweep_all_columns",
                key="mouse_left",
                description="Execute rapid multi-column sweep across all 7 tableau columns",
                is_mouse_click=True,
            ),
            GameAction(
                name="tap_foundation",
                key="mouse_left",
                description="Tap foundation piles (Aces to Kings) to retrieve or verify sequence",
                is_mouse_click=True,
            ),
            GameAction(
                name="new_deal",
                key="mouse_left",
                description="Start a new deal or restart when game ends",
                is_mouse_click=True,
            ),
            GameAction(
                name="wait",
                key="none",
                description="Pause for card animation or deal completion",
            ),
        ],
    ),
    GameProfile(
        id="mobile_snake",
        name="🐍 Snake & Grid Runners",
        category="arcade",
        description="Autonomous pilot for Snake arcades: evaluates 4-way heading, food trajectory, body collisions, and cornering maneuvers",
        default_window_keyword="snake",
        scan_direction="omnidirectional",
        auto_restart_key="space",
        is_builtin=True,
        actions=[
            GameAction(
                name="turn_up",
                key="up",
                description="Turn heading upward towards food or open vertical corridor",
                duration_sec=0.04,
            ),
            GameAction(
                name="turn_down",
                key="down",
                description="Turn heading downward towards food or away from ceiling collision",
                duration_sec=0.04,
            ),
            GameAction(
                name="turn_left",
                key="left",
                description="Turn heading left towards food or open lateral lane",
                duration_sec=0.04,
            ),
            GameAction(
                name="turn_right",
                key="right",
                description="Turn heading right towards food or open lateral lane",
                duration_sec=0.04,
            ),
            GameAction(
                name="maintain_heading",
                key="none",
                description="Current heading is clear and direct to target",
            ),
        ],
    ),
    GameProfile(
        id="pc_balatro",
        name="🃏 Balatro Poker Roguelike (Steam / PC)",
        category="strategy",
        description="Autonomous pilot for Balatro: card selection, scoring poker hands, discarding unneeded cards, and rerolling/cashing out",
        default_window_keyword="balatro",
        scan_direction="omnidirectional",
        auto_restart_key="enter",
        is_builtin=True,
        actions=[
            GameAction(
                name="balatro_play_hand",
                key="enter",
                description="Play selected poker hand combination to score chips against the blind",
            ),
            GameAction(
                name="balatro_discard",
                key="backspace",
                description="Discard selected unneeded cards to draw fresh cards from deck",
            ),
            GameAction(
                name="select_card",
                key="mouse_left",
                description="Tap/click card in hand to toggle selection for play or discard",
                is_mouse_click=True,
            ),
            GameAction(
                name="cash_out",
                key="mouse_left",
                description="Cash out blind victory, collect dollars, and advance to next round",
                is_mouse_click=True,
            ),
            GameAction(
                name="wait",
                key="none",
                description="Wait for chips scoring tally or card deal animation",
            ),
        ],
    ),
    GameProfile(
        id="pc_slaythespire",
        name="🗡️ Slay the Spire Roguelike (Steam / PC)",
        category="strategy",
        description="Autonomous pilot for Slay the Spire: evaluates energy, drags attack cards onto enemies, applies defense shields, and ends turn",
        default_window_keyword="slay the spire",
        scan_direction="omnidirectional",
        auto_restart_key="space",
        is_builtin=True,
        actions=[
            GameAction(
                name="play_card_enemy",
                key="mouse_left",
                description="Drag offensive combat card from hand onto enemy target",
                is_mouse_click=True,
            ),
            GameAction(
                name="play_card_self",
                key="mouse_left",
                description="Drag defense shield or power card onto hero character",
                is_mouse_click=True,
            ),
            GameAction(
                name="end_turn",
                key="e",
                description="Press E or click End Turn when energy is depleted",
            ),
            GameAction(
                name="select_card",
                key="mouse_left",
                description="Inspect or select card in hand",
                is_mouse_click=True,
            ),
            GameAction(
                name="wait",
                key="none",
                description="Observe enemy intent, attack animations, or relic triggers",
            ),
        ],
    ),
    GameProfile(
        id="pc_hearthstone",
        name="🛡️ Hearthstone / MTG / Master Duel (PC)",
        category="strategy",
        description="Autonomous pilot for PC card battlers: plays cards into board lanes, directs minion attacks at face or enemy minions, uses hero power, passes turn",
        default_window_keyword="hearthstone",
        scan_direction="omnidirectional",
        auto_restart_key="space",
        is_builtin=True,
        actions=[
            GameAction(
                name="play_card_center",
                key="mouse_left",
                description="Drag card from bottom hand onto battlefield",
                is_mouse_click=True,
            ),
            GameAction(
                name="attack_face",
                key="mouse_left",
                description="Drag friendly minion attack arrow directly at enemy hero",
                is_mouse_click=True,
            ),
            GameAction(
                name="attack_minion",
                key="mouse_left",
                description="Drag friendly minion to trade into opposing creature",
                is_mouse_click=True,
            ),
            GameAction(
                name="hero_power",
                key="mouse_left",
                description="Activate Hero Power or Leader ability",
                is_mouse_click=True,
            ),
            GameAction(
                name="end_turn",
                key="space",
                description="Click End Turn or press Space to pass priority",
            ),
            GameAction(
                name="wait",
                key="none",
                description="Wait for opponent rope or attack animations",
            ),
        ],
    ),
    GameProfile(
        id="pc_solitaire",
        name="♠️ Windows Solitaire Collection (PC)",
        category="strategy",
        description="Autonomous pilot for PC Solitaire / Klondike / FreeCell / Spider: sweeps columns, auto-moves cards to foundations, draws fresh cards",
        default_window_keyword="solitaire",
        scan_direction="omnidirectional",
        auto_restart_key="space",
        is_builtin=True,
        actions=[
            GameAction(
                name="sweep_all_columns",
                key="mouse_left",
                description="Execute rapid cascade click sweep across all 7 tableau columns",
                is_mouse_click=True,
            ),
            GameAction(
                name="tap_tableau_column",
                key="mouse_left",
                description="Click exposed card in tableau column to trigger auto-move",
                is_mouse_click=True,
            ),
            GameAction(
                name="draw_stock",
                key="mouse_left",
                description="Click stock deck to deal fresh cards to waste pile",
                is_mouse_click=True,
            ),
            GameAction(
                name="tap_waste_card",
                key="mouse_left",
                description="Click waste pile card to auto-place onto foundation or tableau",
                is_mouse_click=True,
            ),
            GameAction(
                name="auto_complete",
                key="mouse_left",
                description="Click Auto-Complete button when all cards are face-up",
                is_mouse_click=True,
            ),
        ],
    ),
    GameProfile(
        id="pc_aimlab",
        name="🎯 Aim Lab & Target Clickers (Steam / PC)",
        category="clicker",
        description="Precision flick and click targeting for Aim Lab, KovaaK's, and target clickers",
        default_window_keyword="aim",
        scan_direction="click_targets",
        auto_restart_key="space",
        is_builtin=True,
        actions=[
            GameAction(
                name="click_target",
                key="mouse_left",
                description="Click active orb/target centroid on screen",
                is_mouse_click=True,
            ),
            GameAction(
                name="wait",
                key="none",
                description="Scanning for active target spawns",
            ),
        ],
    ),
    GameProfile(
        id="pc_roblox",
        name="🧱 Roblox Action & Obby Parkour (PC)",
        category="platformer",
        description="Autonomous pilot for Roblox: WASD navigation, precision Space jumps across lava/pits, camera adjustments",
        default_window_keyword="roblox",
        scan_direction="omnidirectional",
        auto_restart_key="space",
        is_builtin=True,
        actions=[
            GameAction(
                name="move_forward",
                key="w",
                description="Run forward along course or platform",
                duration_sec=0.25,
            ),
            GameAction(
                name="jump",
                key="space",
                description="Jump over hazard block, gap, or obstacle",
                duration_sec=0.12,
            ),
            GameAction(
                name="move_left",
                key="a",
                description="Strafe left away from hazard",
                duration_sec=0.15,
            ),
            GameAction(
                name="move_right",
                key="d",
                description="Strafe right away from hazard",
                duration_sec=0.15,
            ),
            GameAction(
                name="interact",
                key="e",
                description="Press E to interact with prompt or door",
                duration_sec=0.10,
            ),
        ],
    ),
    GameProfile(
        id="pc_minecraft",
        name="⛏️ Minecraft Survival & Mining (PC)",
        category="platformer",
        description="Autonomous pilot for Minecraft: WASD navigation, sprint jumping, mining blocks, placing items",
        default_window_keyword="minecraft",
        scan_direction="omnidirectional",
        auto_restart_key="space",
        is_builtin=True,
        actions=[
            GameAction(
                name="move_forward",
                key="w",
                description="Walk forward",
                duration_sec=0.25,
            ),
            GameAction(
                name="jump",
                key="space",
                description="Jump over 1-block elevation or obstacle",
                duration_sec=0.12,
            ),
            GameAction(
                name="attack_mine",
                key="mouse_left",
                description="Hold left click to attack hostile mob or mine targeted block",
                is_mouse_click=True,
            ),
            GameAction(
                name="use_place",
                key="mouse_right",
                description="Right click to place block or eat food",
            ),
            GameAction(
                name="move_left",
                key="a",
                description="Strafe left",
                duration_sec=0.15,
            ),
            GameAction(
                name="move_right",
                key="d",
                description="Strafe right",
                duration_sec=0.15,
            ),
        ],
    ),
    GameProfile(
        id="pc_trackmania",
        name="🏎️ Trackmania & PC Racing",
        category="platformer",
        description="Autonomous vehicle racing pilot: throttle acceleration, braking, precision steering",
        default_window_keyword="trackmania",
        scan_direction="omnidirectional",
        auto_restart_key="backspace",
        is_builtin=True,
        actions=[
            GameAction(
                name="accelerate",
                key="up",
                description="Hold accelerator throttle down straightaways",
                duration_sec=0.30,
            ),
            GameAction(
                name="turn_left",
                key="left",
                description="Steer into left apex",
                duration_sec=0.12,
            ),
            GameAction(
                name="turn_right",
                key="right",
                description="Steer into right apex",
                duration_sec=0.12,
            ),
            GameAction(
                name="brake",
                key="down",
                description="Tap brake to initiate controlled drift",
                duration_sec=0.08,
            ),
        ],
    ),
    GameProfile(
        id="pc_universal",
        name="💻 Universal PC Game AI (Auto-Adapts to ANY PC Game)",
        category="runner",
        description="Autonomous universal PC agent: dynamically detects hazards, platforms, targets, and objectives, routing optimal controls through Laya + Jev",
        default_window_keyword="",
        scan_direction="omnidirectional",
        auto_restart_key="space",
        is_builtin=True,
        actions=[
            GameAction(
                name="jump",
                key="space",
                description="Jump or vault over oncoming obstacles or hazards",
                duration_sec=0.12,
            ),
            GameAction(
                name="duck",
                key="down",
                description="Duck or slide under overhead obstacles",
                duration_sec=0.25,
            ),
            GameAction(
                name="move_left",
                key="a",
                description="Move left",
                duration_sec=0.15,
            ),
            GameAction(
                name="move_right",
                key="d",
                description="Move right",
                duration_sec=0.15,
            ),
            GameAction(
                name="click_target",
                key="mouse_left",
                description="Click priority target or objective on screen",
                is_mouse_click=True,
            ),
            GameAction(
                name="maintain_course",
                key="none",
                description="Maintain current position or trajectory",
            ),
        ],
    ),
    GameProfile(
        id="mobile_8ball_pool",
        name="🎱 8 Ball Pool & Billiards (Miniclip)",
        category="sports",
        description="Autonomous pilot and guideline computer vision assistant for Miniclip 8 Ball Pool: calculates cue ball cut angles, ghost ball trajectory to pockets, power meter pull back, and break shots",
        default_window_keyword="pool",
        scan_direction="omnidirectional",
        auto_restart_key="space",
        is_builtin=True,
        actions=[
            GameAction(
                name="break_shot",
                key="mouse_left",
                description="Execute maximum power break shot into the center ball of the opening rack",
                duration_sec=0.25,
                is_mouse_click=True,
            ),
            GameAction(
                name="aim_target_ball",
                key="mouse_left",
                description="Align cue stick along the calculated ghost ball cut angle toward the optimal pocket",
                duration_sec=0.15,
                is_mouse_click=True,
            ),
            GameAction(
                name="shoot_power",
                key="mouse_left",
                description="Pull back the power meter cue slider on the left to calculated strength and release",
                duration_sec=0.25,
                is_mouse_click=True,
            ),
            GameAction(
                name="fine_tune_aim_left",
                key="left",
                description="Micro-adjust cue stick angle counter-clockwise for pixel-perfect pocket alignment",
                duration_sec=0.05,
            ),
            GameAction(
                name="fine_tune_aim_right",
                key="right",
                description="Micro-adjust cue stick angle clockwise for pixel-perfect pocket alignment",
                duration_sec=0.05,
            ),
            GameAction(
                name="spin_top",
                key="up",
                description="Apply follow-through topspin to cue ball",
                duration_sec=0.08,
            ),
            GameAction(
                name="spin_back",
                key="down",
                description="Apply draw backspin to cue ball for position play",
                duration_sec=0.08,
            ),
            GameAction(
                name="spin_center",
                key="space",
                description="Reset spin to neutral center ball",
                duration_sec=0.05,
            ),
            GameAction(
                name="call_pocket_tl",
                key="1",
                description="Call Top-Left Corner Pocket on 8-ball",
            ),
            GameAction(
                name="call_pocket_tm",
                key="2",
                description="Call Top-Middle Side Pocket on 8-ball",
            ),
            GameAction(
                name="call_pocket_tr",
                key="3",
                description="Call Top-Right Corner Pocket on 8-ball",
            ),
            GameAction(
                name="call_pocket_bl",
                key="4",
                description="Call Bottom-Left Corner Pocket on 8-ball",
            ),
            GameAction(
                name="call_pocket_bm",
                key="5",
                description="Call Bottom-Middle Side Pocket on 8-ball",
            ),
            GameAction(
                name="call_pocket_br",
                key="6",
                description="Call Bottom-Right Corner Pocket on 8-ball",
            ),
            GameAction(
                name="auto_rematch",
                key="enter",
                description="Tap Rematch / Play Again / Claim Coins button after match concludes",
            ),
            GameAction(
                name="wait",
                key="none",
                description="Wait while balls are rolling, opponent is taking shot, or table is settling",
            ),
        ],
    ),
    GameProfile(
        id="pc_8ball_pool",
        name="🎱 8 Ball Pool (Web / PC / Steam)",
        category="sports",
        description="Autonomous pilot for 8 Ball Pool on browser, Facebook, and PC desktop: mouse drag aiming, pocket line calculation, and power bar meter release",
        default_window_keyword="pool",
        scan_direction="omnidirectional",
        auto_restart_key="enter",
        is_builtin=True,
        actions=[
            GameAction(
                name="break_shot",
                key="mouse_left",
                description="Full power break shot into the center rack",
                duration_sec=0.25,
                is_mouse_click=True,
            ),
            GameAction(
                name="aim_target_ball",
                key="mouse_left",
                description="Rotate cue stick to align laser guide line with target ball and pocket",
                duration_sec=0.15,
                is_mouse_click=True,
            ),
            GameAction(
                name="shoot_power",
                key="mouse_left",
                description="Drag power meter slider on left and release",
                duration_sec=0.25,
                is_mouse_click=True,
            ),
            GameAction(
                name="fine_tune_aim_left",
                key="left",
                description="Micro-step cue angle counter-clockwise",
                duration_sec=0.05,
            ),
            GameAction(
                name="fine_tune_aim_right",
                key="right",
                description="Micro-step cue angle clockwise",
                duration_sec=0.05,
            ),
            GameAction(
                name="auto_rematch",
                key="enter",
                description="Accept rematch or queue next match",
            ),
            GameAction(
                name="wait",
                key="none",
                description="Wait for balls to stop moving",
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
