"""
CLI interface for Jev-GamePilot.
Provides command-line launching, headless monitoring, terminal telemetry, and diagnostics.
"""

import argparse
import os
import sys
import time
import webbrowser

# Ensure Windows cp1252 handles Unicode emojis safely
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def run_hud():
    """Launch CustomTkinter Cyber HUD."""
    from game_hud import GamePilotHUD

    console.print(
        "[bold cyan]⚡ Launching Jev-GamePilot Cyber Desktop HUD...[/bold cyan]"
    )
    app = GamePilotHUD()
    app.mainloop()


def open_dino():
    """Open offline Dino runner."""
    path = os.path.abspath("dino_game.html")
    console.print(
        f"[bold green]🌐 Opening offline Dino Runner at: {path}[/bold green]"
    )
    webbrowser.open(f"file:///{path}")


def test_brain():
    """Test Jev System One gaming decisions."""
    from jev_brain import JevBrain

    console.print(
        "[bold cyan]🧠 Querying TypeSafe Jev System One on mock game states...[/bold cyan]"
    )
    brain = JevBrain()

    scenarios = [
        (
            "Approaching Cactus",
            {
                "game": "Chrome Dino Runner",
                "dino_state": "running",
                "game_speed_px_sec": 420.0,
                "nearest_obstacle": {
                    "type": "cactus_large",
                    "distance_px": 85,
                    "time_to_impact_ms": 202.0,
                },
                "total_obstacles_in_view": 1,
            },
        ),
        (
            "Approaching Mid-Height Pterodactyl",
            {
                "game": "Chrome Dino Runner",
                "dino_state": "running",
                "game_speed_px_sec": 480.0,
                "nearest_obstacle": {
                    "type": "bird_mid",
                    "distance_px": 105,
                    "time_to_impact_ms": 218.0,
                },
                "total_obstacles_in_view": 1,
            },
        ),
        (
            "Clear Horizon",
            {
                "game": "Chrome Dino Runner",
                "dino_state": "running",
                "game_speed_px_sec": 420.0,
                "nearest_obstacle": None,
                "total_obstacles_in_view": 0,
            },
        ),
    ]

    table = Table(
        title="Jev System One Gaming Policy Verification",
        header_style="bold magenta",
    )
    table.add_column("Scenario", style="bold white")
    table.add_column("Jev Action", style="bold cyan")
    table.add_column("Threat Score", style="bold yellow")
    table.add_column("Confidence", style="bold green")
    table.add_column("Fast-Fall", style="bold blue")
    table.add_column("Latency", style="dim white")

    for name, state in scenarios:
        with console.status(f"[bold green]Evaluating {name}...[/]"):
            res = brain.query_jev_dino(state)
            if res:
                table.add_row(
                    name,
                    res["action"].upper(),
                    f"{round(res['threat_score'] * 100, 1)}%",
                    f"{round(res['confidence'] * 100, 1)}%",
                    "YES" if res["fast_fall"] else "NO",
                    f"{res['latency_ms']} ms",
                )
            else:
                table.add_row(name, "[red]FAILED[/red]", "-", "-", "-", "-")

    console.print(table)


def run_headless(snap_keyword: str = "dino", armed: bool = True):
    """Run autonomous pilot in terminal without GUI."""
    from pilot_core import PilotCore

    core = PilotCore()
    console.print(
        f"[bold cyan]🔍 Snapping vision to window matching '{snap_keyword}'...[/bold cyan]"
    )
    if core.vision.snap_to_window(snap_keyword):
        reg = core.vision.region
        console.print(
            f"[bold green]✓ Snapped: ({reg['left']}, {reg['top']}) {reg['width']}x{reg['height']}[/bold green]"
        )
    else:
        console.print(
            "[yellow]⚠️ Could not find matching window, using center screen.[/yellow]"
        )

    console.print(
        Panel.fit(
            f"[bold green]🚀 Jev-GamePilot Headless Runner[/bold green]\n"
            f"Mode: [bold cyan]Chrome Dino / Edge Surf[/bold cyan]\n"
            f"Inputs: [bold {'red' if armed else 'yellow'}]{'ARMED (Live Controls)' if armed else 'MONITOR ONLY'}[/]\n"
            f"Press [bold white on red] CTRL+C [/] to stop at any time.",
            border_style="cyan",
        )
    )

    core.start(arm_inputs=armed)
    try:
        while True:
            time.sleep(0.5)
            if core.last_state:
                st = core.last_state
                dec = core.last_decision
                act = dec.get("action", "run_normal").upper()
                speed = round(st.game_speed_px_sec, 1)
                jumps = core.total_jumps
                ducks = core.total_ducks
                obs = (
                    f"{st.nearest_obstacle.obstacle_type} ({st.nearest_obstacle.distance_from_dino}px)"
                    if st.nearest_obstacle
                    else "Clear"
                )
                console.print(
                    f"[dim]{round(core.fps, 1)} FPS[/dim] | Action: [bold cyan]{act:<10}[/] | Obstacle: [yellow]{obs:<22}[/] | Jumps: [green]{jumps}[/] Ducks: [blue]{ducks}[/] Speed: [magenta]{speed} px/s[/]"
                )
    except KeyboardInterrupt:
        console.print("\n[bold red]Stopping Jev-GamePilot...[/bold red]")
        core.stop()


def main():
    parser = argparse.ArgumentParser(
        description="Jev-GamePilot: Autonomous AI Gaming Agent powered by TypeSafe Jev System One"
    )
    sub = parser.add_subparsers(dest="command", help="Commands")

    sub.add_parser("hud", help="Launch Cyber Desktop GUI HUD")
    sub.add_parser("open-dino", help="Launch offline Dino Runner in browser")
    sub.add_parser("test-brain", help="Run Jev System One diagnostics")

    play_p = sub.add_parser("play", help="Run autonomous pilot in terminal")
    play_p.add_argument(
        "--snap",
        default="dino",
        help="Window title keyword to snap to (e.g. dino, edge, chrome)",
    )
    play_p.add_argument(
        "--monitor",
        action="store_true",
        help="Monitor only, do not send keypresses",
    )

    args = parser.parse_args()

    if args.command == "open-dino":
        open_dino()
    elif args.command == "test-brain":
        test_brain()
    elif args.command == "play":
        run_headless(snap_keyword=args.snap, armed=not args.monitor)
    else:
        # Default to HUD
        run_hud()


if __name__ == "__main__":
    main()
