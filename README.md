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

* **Supported Games**: Clash of Clans, Brawl Stars, Solar Smash, Subway Surfers, Fruit Ninja, Clash Royale, Earn to Die 2, EA Sports FC / FIFA Mobile, BitLife, Marvel SNAP, Pokémon TCG Pocket, Solitaire, Snake, and Universal Mobile.
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
python phone_pilot.py --profile mobile_coc
python phone_pilot.py --profile mobile_brawlstars
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
│   ├── clash_adapter.py       # Clash Royale tactical RTS controller
│   ├── coc_adapter.py         # Clash of Clans raid spread-deploy controller
│   ├── brawl_adapter.py       # Brawl Stars joystick/attack match controller
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

Also: `run_tests.py` (offline suite runner), `pyproject.toml` (ruff/pytest
config), `.github/workflows/ci.yml` (Windows lint + offline suite).

---

## 🛡️ Safety & Fail-Safes

* **ESC / Q**: Instant emergency stop and shutdown.
* **Space / F8**: Toggle inputs between **ARMED** (live controls) and **DISARMED** (monitor only).
* **Failsafe Corner**: Flick mouse to top-left corner (0, 0) for immediate PyAutoGUI hardware interrupt.
* **Crash-safe teardown**: whatever ends the loop (ESC, Ctrl+C, exception), held keys are released and the capture handle is closed in a `finally` block.

## 🧠 Brain tiers and env switches

The fusion ladder runs fastest-first: Laya → openjev → hosted Fastino →
local GLiNER → LLM2Jev → Bev → TypeSafe → Simple Jev → classifier.dev.
Optional backends are *located* with `importlib.util.find_spec` at import time
and imported inside their loader threads, so `--help` and the HUD open in
seconds even on a machine with no torch stack.

| Variable | Default | Effect |
| :--- | :--- | :--- |
| `JEV_OFFLINE` | `0` | `1` starts no weight loaders and downloads nothing; heuristic/cloud tiers still run. The test runner sets this. |
| `JEV_ENABLE_GLINER` | `0` | `1` lets a weak openjev tick background-load the local 340M GLiNER weights (~1.9 GB RAM, multi-second warm-up). |
| `LLM2JEV_MODEL` | unset | Local path or HF id; enables the LLM2Jev loader. |
| `OPENJEV_PATH` | `models/openjev` | Local openjev weights directory. |
| `LAYA_FAST_PATH_CONF` | `0.88` | Laya confidence needed to skip escalation. |
| `LOCAL_ACCEPT_CONF` | `0.72` | Local-tier confidence needed to skip deeper tiers. |
| `FAST_PATH_THREAT_GATE` | `0.60` | Threat level that blocks a passive fast-path commit. |
| `TYPESAFE_API_KEY` | unset | Enables the TypeSafe cloud tier. |
| `FASTINO_API_KEY` | unset | Enables the hosted GLiNER tier. |
| `FEATHERLESS_API_KEY` | unset | Simple Jev tier (keyless demo works without it). |
| `BEV_URL` | `http://127.0.0.1:18781` | Local Bev SystemOne endpoint. |

## Offline regression tests

```powershell
python run_tests.py          # 350+ tests, ~20s, no weights, no network
python run_tests.py -v
python run_tests.py tests.test_input_controller
```

`run_tests.py` sets `JEV_OFFLINE=1` before discovery and imports `tests` as a
package, so the guard applies before any test pulls in `universal_brain`.
The suite mocks desktop APIs: it never sends keyboard/mouse events, contacts AI
services, or requires a running game. `test_classifier_pilot.py` is a live
network smoke test and is not part of this suite.

Lint gate (defects only: undefined names, unused imports, syntax):

```powershell
python -m ruff check .
```
