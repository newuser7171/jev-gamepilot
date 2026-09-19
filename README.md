# ⚡ Universal Jev-GamePilot: Autonomous AI for ANY Game

**Universal Jev-GamePilot** is an autonomous AI gaming agent powered by **TypeSafe's Jev System One** model (`Choice`, `Score`, `Noul`). It captures any game running live on your Windows desktop (browser games, Steam titles, PC games, or Android emulators like BlueStacks), extracts visual game state at 60+ FPS, dynamically queries Jev for tactical decisions matching the game's custom keybindings, and executes microsecond keyboard and mouse inputs with multi-tier emergency stop fail-safes.

---

## 🎮 Built-in Game Profiles & Custom Creator

| Profile | Genre | Key Mappings | What Jev Evaluates |
| :--- | :--- | :--- | :--- |
| **🦖 Chrome Dino & Obstacle Runner** | Runner | `Jump (Space)`, `Duck (Down)` | Cacti distance, 3-altitude birds, impact velocity |
| **🏄 3-Lane Mobile Runner (Subway Surfers)** | Runner | `Left (Left)`, `Right (Right)`, `Jump (Up)`, `Slide (Down)` | 3-lane depth corridor, oncoming trains, barriers |
| **🐦 Flappy Bird & One-Tap Jumper** | Arcade | `Flap (Space)`, `Glide (None)` | Gap altitude, vertical fall velocity, gap timing |
| **🎯 Target Clicker & Aim Trainer (Osu / Aim Labs)** | Clicker | `Click Target (Mouse Left)` | Screen target crosshairs, distance, click precision |
| **🕹️ 2D Platformer / Retro Arcade** | Platformer | `Left (A)`, `Right (D)`, `Jump (Space)`, `Attack (J)` | Enemy proximity, pit gaps, platform navigation |
| **➕ Custom Game Profile Creator** | Custom | Any keys (`A-Z`, Arrows, Space, Mouse Click) | User-defined objective prompt and allowed actions |

---

## 🚀 Key Features

* **Universal Vision Engine (`universal_vision.py`)**:
  * Tracks player avatars across any visual style using contour dynamics or user-calibrated template matching (`📌 Calibrate Avatar`).
  * Computes threat velocity vectors heading toward the player.
  * Pinpoints high-contrast clickable targets for clicker and aim games.
* **Dynamic TypeSafe Jev System One Brain (`universal_brain.py`)**:
  * Dynamically generates `Choice` criteria from the active profile's action descriptions.
  * Dynamically scales `Score` threat urgency and evaluates `Noul` reflex triggers.
* **Dual Game Arenas**:
  * **🎮 Native Dino Arena**: Embedded 60 FPS canvas game inside the HUD for instant zero-setup play.
  * **🖥️ Universal Screen Grabber**: Locks onto any game window (Roblox, Minecraft, BlueStacks, Chrome, Steam) using hardware virtual keys (`MapVirtualKeyW`) and coordinate clicks.
* **🪟 Floating Mini-Bar Mode**:
  * Shrinks the HUD into a sleek, translucent 380x105 overlay that stays on top of any game without obstructing the screen.
* **Emergency Fail-Safes**:
  * Press **ESC** at any time to immediately disarm controls and release all keys.
  * Flick mouse to top-left corner for PyAutoGUI panic abort.

---

## 💻 CLI Commands

```powershell
# Open Universal Cyber HUD
python cli.py hud

# List all available game profiles
python cli.py profiles

# Run Jev System One benchmark across profiles
python cli.py test-brain

# Run headless pilot on a specific profile
python cli.py play --profile runner_3lane --snap bluestacks
python cli.py play --profile flappy_tap --snap chrome
python cli.py play --profile aim_clicker --monitor
```

---

## 📁 Architecture

```
jev-gamepilot/
├── custom_profile_dialog.py   # Modal dialog for creating new game profiles
├── profile_manager.py         # Dynamic game profile repository (profiles.json)
├── universal_brain.py         # Dynamic TypeSafe Jev System One engine
├── universal_vision.py        # Universal avatar tracking & threat vectors
├── input_controller.py        # Universal hardware key & mouse click dispatcher
├── pilot_core.py              # 60 FPS orchestrator connecting perception & brain
├── game_hud.py                # Cyber HUD with Native Arena & Floating Mini-Bar
├── embedded_dino.py           # Native embedded 60 FPS canvas game
├── dino_game.html             # Offline HTML5 Dino runner
├── cli.py                     # Universal CLI runner
└── jev-gamepilot.bat          # Desktop batch launcher
```
