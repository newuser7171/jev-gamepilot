@echo off
title Laya + Jev PC Game Pilot // Autonomous PC Gaming AI
cd /d "%~dp0"

echo =======================================================
echo       LAYA + JEV UNIVERSAL PC GAME PILOT
echo       Dual-Tier Autonomous AI Driver for PC Games
echo =======================================================
echo.
echo Mode:
echo  1. [AUTO] Universal PC AI (Auto-detects active game window on screen)
echo  2. Balatro / Poker Roguelike (Play Hand, Discard, Cash Out)
echo  3. Slay the Spire / Roguelike Deckbuilder (Drag Attacks, Shields, End Turn)
echo  4. Hearthstone / MTG Arena / Master Duel (Card Play, Minion Trades, Face)
echo  5. Windows Solitaire Collection (Tableau Cascade, Stock, Waste, Auto-Finish)
echo  6. Chrome Dino ^& Browser Runners (Jump, Duck, Auto-Restart)
echo  7. Aim Lab ^& Target Clickers (Precision Centroid Flick Click)
echo  8. Roblox / Obby Parkour (WASD Movement, Space Jump, Camera)
echo  9. Minecraft / Survival ^& Mining (WASD, Jump, Mine, Place)
echo 10. Trackmania ^& PC Racing (Throttle, Steer, Drift Brake)
echo 11. Retro 2D Platformer / Arcade (D-pad Movement, Jump, Attack)
echo 12. Chess.com ^& Lichess Screen Copilot (Best Move Clicker)
echo.
set /p choice="Select mode [1-12] (default: 1): "

if "%choice%"=="2" (
    python pc_pilot.py --profile pc_balatro
) else if "%choice%"=="3" (
    python pc_pilot.py --profile pc_slaythespire
) else if "%choice%"=="4" (
    python pc_pilot.py --profile pc_hearthstone
) else if "%choice%"=="5" (
    python pc_pilot.py --profile pc_solitaire
) else if "%choice%"=="6" (
    python pc_pilot.py --profile runner_dino
) else if "%choice%"=="7" (
    python pc_pilot.py --profile pc_aimlab
) else if "%choice%"=="8" (
    python pc_pilot.py --profile pc_roblox
) else if "%choice%"=="9" (
    python pc_pilot.py --profile pc_minecraft
) else if "%choice%"=="10" (
    python pc_pilot.py --profile pc_trackmania
) else if "%choice%"=="11" (
    python pc_pilot.py --profile retro_platformer
) else if "%choice%"=="12" (
    python pc_pilot.py --profile chess_copilot
) else (
    python pc_pilot.py --profile auto
)

pause
