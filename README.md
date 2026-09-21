# ⚡ Jev-GamePilot: Dual-Tier Autonomous AI for PC & Mobile Games

**Jev-GamePilot** is a universal autonomous AI gaming agent powered by **Laya (local sub-30ms System One inference)** and **TypeSafe's Jev System One** (`Choice`, `Score`, `Noul`). It captures real-time gameplay at 60+ FPS, fuses instant local reflexes with high-level strategic reasoning, and executes physical hardware inputs across Windows PC games and connected Android phones.

---

## 🚀 Dual Platforms: PC & Phone

### 1. 💻 PC Game Pilot (`pc_pilot.py` / `run-pc-pilot.bat`)
Locks onto any running Windows game window or full desktop (Steam, Epic, Battle.net, Browser, Standalone) with microsecond hardware scan codes (`SendInput`), precision mouse clicks, and smooth card drag gestures.

| PC Game Profile | Genre | Inputs | What AI Evaluates |
| :--- | :--- | :--- | :--- |
| **🃏 Balatro** | Roguelike Poker | `Select`, `Play Hand [Enter]`, `Discard [Backspace]`, `Cash Out` | Hand cards, poker hand score valuation, discard cycling |
| **🗡️ Slay the Spire** | Deckbuilder | `Drag Card to Enemy`, `Drag Shield to Self`, `End Turn [E]` | Energy balance, enemy intent, lethal attacks vs defense |
| **🛡️ Hearthstone / MTG / Master Duel** | Card Battler | `Drag Card to Board`, `Attack Face`, `Attack Minion`, `Hero Power` | Mana curves, board presence, minion trades, lethal face damage |
| **♠️ Windows Solitaire Collection** | Card Puzzle | `Cascade Sweep`, `Column Taps 1-7`, `Draw Stock`, `Auto-Finish` | Available auto-moves, card foundations, empty column builds |
| **🎯 Aim Lab & KovaaK's** | FPS Aim Trainer | `Precision Centroid Flick Click` | Target orb spawns, distance, microsecond flick accuracy |
| **🧱 Roblox** | 3D Action / Obby | `WASD Movement`, `Space Jump`, `E Interact` | Gap obstacles, lava pits, moving hazards |
| **⛏️ Minecraft** | Sandbox / Survival | `WASD Movement`, `Space Jump`, `Attack/Mine`, `Use/Place` | Hostile mobs, elevation changes, resource mining |
| **🏎️ Trackmania** | Racing | `Up (Gas)`, `Left/Right (Steer)`, `Down (Drift Brake)` | Track apex, obstacle dodging, drift control |
| **🦖 Chrome Dino** | Runner | `Space (Jump)`, `Down (Duck)`, `Space (Restart)` | Cacti clusters, 3-altitude pterodactyls |
| **💻 Universal PC AI** | Universal | Dynamic 4-way movement, jumps, target clicks | Automatically detects hazards, platforms, and objectives |

### 2. 📱 Phone Game Pilot (`phone_pilot.py` / `run-phone-pilot.bat`)
Connects directly to an Android phone over USB or Wi-Fi via ADB with zero-latency screen capture and atomic hardware touch injection.

* **Supported Games**: Solar Smash, Subway Surfers, Fruit Ninja, Clash Royale, Earn to Die 2, EA Sports FC / FIFA Mobile, BitLife, Marvel SNAP, Pokémon TCG Pocket, Solitaire, Snake, and Universal Mobile.
* **Orientation Awareness**: Dynamically detects Portrait vs Landscape and adjusts bottom drawers, side trays, and target coordinates.

---

## ⚡ Quick Start

Keep the `.bat` launchers in the repository folder. They resolve their working
directory relative to their own location, so the checkout can live anywhere.

### Run PC Game Pilot
Double-click `run-pc-pilot.bat` in the repository folder or run:
```powershell
# Auto-detects active PC game window
python pc_pilot.py --profile auto

# Or run a specific game profile:
python pc_pilot.py --profile pc_balatro
python pc_pilot.py --profile pc_slaythespire
python pc_pilot.py --profile pc_hearthstone
python pc_pilot.py --profile pc_solitaire
python pc_pilot.py --profile pc_aimlab
```

### Run Phone Game Pilot
Double-click `run-phone-pilot.bat` in the repository folder or run:
```powershell
# Auto-detects active foreground game on phone
python phone_pilot.py --profile auto

# Or run specific game:
python phone_pilot.py --profile mobile_solarsmash
python phone_pilot.py --profile mobile_solitaire
python phone_pilot.py --profile mobile_card_battler
```

---

## 🧠 Architecture

```
jev-gamepilot/
├── pc_pilot.py                # Autonomous PC Game Pilot with window tracking & input injection
├── run-pc-pilot.bat           # Desktop batch launcher for PC games
├── phone_pilot.py             # Autonomous Android Phone Game Pilot via ADB
├── run-phone-pilot.bat        # Desktop batch launcher for Android phone games
├── adapters/
│   ├── phone_adapter.py       # High-speed ADB frame stream & atomic touch dispatcher
│   ├── dino_adapter.py        # Specialized Dino runner perception
│   └── screen_chess_adapter.py# Screen Chess 8x8 detection & board evaluation
├── universal_brain.py         # Dual-Tier Consensus: Local Laya (sub-30ms) + TypeSafe Jev
├── universal_vision.py        # Universal entity tracking, threat vectors & target centroids
├── input_controller.py        # Windows hardware scan codes, mouse drags & clicks
├── profile_manager.py         # PC and Mobile game profiles repository
├── vision_engine.py           # High-FPS desktop screen grabber with desktop attachment
├── game_hud.py                # Cyber HUD with Native Arena & Floating Mini-Bar
└── cli.py                     # Universal CLI interface
```

---

## 🛡️ Safety & Fail-Safes

* **ESC / Q**: Instant emergency stop and shutdown.
* **Space / F8**: Toggle inputs between **ARMED** (live controls) and **DISARMED** (monitor only).
* **Failsafe Corner**: Flick mouse to top-left corner (0, 0) for immediate PyAutoGUI hardware interrupt.

## Offline regression tests

Run `python -m unittest discover -s tests -v` from the repository folder.
These tests mock desktop APIs: they do not send keyboard/mouse events, contact
AI services, or require a running game. The separate `test_classifier_pilot.py`
script is a live network smoke test and is not part of this offline suite.
