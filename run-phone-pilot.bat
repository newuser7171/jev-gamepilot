@echo off
title Laya + Jev Phone Game Pilot // Autonomous Mobile Gaming AI
cd /d "C:\Users\newuser\.gemini\antigravity\scratch\jev-gamepilot"

echo =======================================================
echo     ⚡ LAYA + JEV UNIVERSAL PHONE GAME PILOT
echo       Dual-Tier Autonomous AI Driver for ANY Game
echo =======================================================
echo.
echo Mode:
echo  1. [AUTO] Universal AI (Auto-detects ANY game opened on phone)
echo  2. Subway Surfers / 3-Lane Runners (Swipes Up/Down/Left/Right)
echo  3. Fruit Ninja ^& Slicers (Diagonal Target Slices)
echo  4. Clash Royale ^& RTS (Lane Deployment, Spells, Deck Timing)
echo  5. Earn to Die 2 / 2D Drivers (Gas, Boost, Tilt)
echo  6. EA Sports FC / FIFA Mobile (Sprint, Pass, Shoot)
echo  7. Solar Smash ^& Sandboxes (Laser, Asteroids, Superweapons)
echo  8. BitLife ^& Choice Sims (Age Progression, Scenario Decisions)
echo  9. Flappy Bird / One-Tap Arcades (Precision Taps)
echo 10. Card Games ^& TCGs (Marvel SNAP, Hearthstone, Pokemon, Balatro)
echo 11. Solitaire ^& Classic Card Puzzles (Klondike, Spider, FreeCell)
echo 12. Snake ^& Grid Arcades (4-Way Turn Reflexes)
echo.
set /p choice="Select mode [1-12] (default: 1): "

if "%choice%"=="2" (
    python phone_pilot.py --profile runner_3lane
) else if "%choice%"=="3" (
    python phone_pilot.py --profile mobile_fruit_ninja
) else if "%choice%"=="4" (
    python phone_pilot.py --profile mobile_clash_royale
) else if "%choice%"=="5" (
    python phone_pilot.py --profile mobile_earntodie2
) else if "%choice%"=="6" (
    python phone_pilot.py --profile mobile_fifa
) else if "%choice%"=="7" (
    python phone_pilot.py --profile mobile_solarsmash
) else if "%choice%"=="8" (
    python phone_pilot.py --profile mobile_bitlife
) else if "%choice%"=="9" (
    python phone_pilot.py --profile flappy_tap
) else if "%choice%"=="10" (
    python phone_pilot.py --profile mobile_card_battler
) else if "%choice%"=="11" (
    python phone_pilot.py --profile mobile_solitaire
) else if "%choice%"=="12" (
    python phone_pilot.py --profile mobile_snake
) else (
    python phone_pilot.py --profile auto
)

pause
