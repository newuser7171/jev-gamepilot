# ⚡ Jev-GamePilot: Autonomous AI Gaming Agent

**Jev-GamePilot** is a computer vision and real-time gaming agent powered by **TypeSafe's Jev System One** model (`Choice`, `Score`, `Noul`). It captures games running live on your Windows screen (browser games, emulators like BlueStacks, or PC titles), extracts visual game state at 60+ FPS, queries Jev for semantic tactical maneuvers, and executes microsecond inputs with emergency fail-safes.

---

## 🎮 Features

* **Real-Time Perception**: Ultra-low latency 60+ FPS frame grabber using `mss` and OpenCV contour tracking.
* **TypeSafe Jev System One Brain**:
  * `Choice`: Selects optimal tactical movement (`jump`, `duck`, `run_normal`, `dodge_left`, `dodge_right`).
  * `Score`: Evaluates immediate collision risk from 0 to 4 ($0.0 \rightarrow 1.0$).
  * `Noul`: Reflex triggers (`is_fast_fall_recommended`, `should_use_powerup`).
* **Game Adapters**:
  * 🦖 **Chrome Dino & Edge Surf**: Ground line detection, obstacle distance tracking, small/large/cluster cactus classification, and 3-altitude pterodactyl recognition (high, mid, low).
  * 🏄 **Subway Surfers**: 3-lane depth corridor segmentation, high barrier ducking, low barrier jumping, and train avoidance.
* **Cyber Desktop HUD (`game_hud.py`)**:
  * Live visual feed with bounding boxes drawn over the player and obstacles.
  * Real-time brain telemetry: active decision, urgency gauge, impact time (ms), speed (px/s), and maneuver counters.
  * Window snappper (auto-locks to Edge, Chrome, or BlueStacks).
  * Built-in offline HTML5 Dino runner launcher.
* **Safety Fail-Safes**:
  * Global emergency kill hotkey (`ESC`).
  * PyAutoGUI mouse corner abort (`FAILSAFE = True`).
  * Non-blocking virtual key simulation via Windows API.

---

## 🚀 Quick Start

### 1. Launch the Cyber Desktop HUD
Double-click [jev-gamepilot.bat](file:///C:/Users/newuser/Downloads/jev-gamepilot.bat) or run:
```powershell
python cli.py hud
```

### 2. Open the Dino Game
Click **🌐 Open Offline Dino Game** in the HUD (or run `python cli.py open-dino`). This opens the authentic bundled HTML5 Dino runner in your default browser.

### 3. Engage GamePilot
1. Select the browser window from the **Viewport Calibration** dropdown and click **🎯 Snap to Selected Window**.
2. Click **🚀 START GAMEPILOT (Click to Play)**.
   * *A 3-second countdown gives you time to click into your game window.*
   * *You can also configure an optional hotkey (`Enter`, `F2`, `Tab`) or use purely on-screen clicks.*
3. Watch Jev autonomously detect obstacles, calculate jump distances, duck mid-air birds, and recover after game-overs!
4. Press **ESC** or click **🛑 STOP GAMEPILOT** at any time to immediately disarm all controls.

---

## 💻 CLI Commands

```powershell
# Open Cyber HUD
python cli.py hud

# Run Jev System One diagnostics & decision table
python cli.py test-brain

# Launch built-in offline Dino game
python cli.py open-dino

# Run headless terminal pilot
python cli.py play --snap dino
python cli.py play --snap edge --monitor   # Monitor only, no inputs
```

---

## 📁 Architecture

```
jev-gamepilot/
├── adapters/
│   ├── dino_adapter.py        # Chrome Dino / Edge Surf perception
│   └── runner_adapter.py      # Subway Surfers 3-lane perception
├── dino_game.html             # Offline authentic HTML5 Dino runner
├── game_hud.py                # CustomTkinter Dark Cyber HUD
├── input_controller.py        # Windows virtual key events & failsafes
├── jev_brain.py               # TypeSafe Jev System One engine
├── pilot_core.py              # 60 FPS perception-decision-action loop
├── vision_engine.py           # mss screen capture & window calibration
├── cli.py                     # Command-line interface
└── jev-gamepilot.bat          # Desktop launcher
```
