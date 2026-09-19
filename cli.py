"""
Universal CLI interface for Jev-GamePilot.
Supports launching the Universal HUD, running headless on any game profile,
and testing Jev System One decisions.
"""

import argparse
import os
import sys
import time
import webbrowser

# Ensure Windows cp1252 handles Unicode safely
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from profile_manager import ProfileManager
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def run_hud():
    """Launch Universal Cyber Desktop HUD."""
    from game_hud import GamePilotHUD

    console.print(
        "[bold cyan]⚡ Launching Universal Jev-GamePilot HUD...[/bold cyan]"
    )
    app = GamePilotHUD()
    app.mainloop()


def list_profiles():
    """List all available game profiles."""
    pm = ProfileManager()
    table = Table(
        title="Universal GamePilot Profiles", header_style="bold magenta"
    )
    table.add_column("Profile ID", style="bold cyan")
    table.add_column("Game Title", style="bold white")
    table.add_column("Genre", style="bold yellow")
    table.add_column("Key Mappings", style="green")

    for p in pm.list_profiles():
        keys = ", ".join([f"{a.name}->{a.key.upper()}" for a in p.actions])
        table.add_row(p.id, p.name, p.category.upper(), keys)

    console.print(table)


def test_brain():
    """Test Jev System One gaming decisions across all profiles."""
    from profile_manager import ProfileManager
    from universal_brain import UniversalBrain
    from universal_vision import UniversalEntity, UniversalSceneState

    console.print(
        "[bold cyan]🧠 Querying TypeSafe Jev System One on Universal Profiles...[/bold cyan]"
    )
    pm = ProfileManager()
    brain = UniversalBrain()

    tests = [
        (
            "runner_dino",
            UniversalSceneState(
                player=UniversalEntity(
                    x=50, y=200, w=40, h=40, entity_type="player"
                ),
                threats=[
                    UniversalEntity(
                        x=120,
                        y=200,
                        w=25,
                        h=40,
                        entity_type="threat",
                        distance_to_player=70.0,
                    )
                ],
                nearest_threat=UniversalEntity(
                    x=120,
                    y=200,
                    w=25,
                    h=40,
                    entity_type="threat",
                    distance_to_player=70.0,
                ),
                threat_urgency=0.88,
            ),
        ),
        (
            "flappy_tap",
            UniversalSceneState(
                player=UniversalEntity(
                    x=60, y=250, w=30, h=30, entity_type="player"
                ),
                threats=[
                    UniversalEntity(
                        x=110,
                        y=280,
                        w=50,
                        h=180,
                        entity_type="threat",
                        distance_to_player=50.0,
                    )
                ],
                nearest_threat=UniversalEntity(
                    x=110,
                    y=280,
                    w=50,
                    h=180,
                    entity_type="threat",
                    distance_to_player=50.0,
                ),
                threat_urgency=0.82,
            ),
        ),
        (
            "aim_clicker",
            UniversalSceneState(
                player=UniversalEntity(
                    x=200, y=200, w=20, h=20, entity_type="player"
                ),
                targets=[
                    UniversalEntity(
                        x=400,
                        y=300,
                        w=40,
                        h=40,
                        entity_type="target",
                        click_x=420,
                        click_y=320,
                    )
                ],
                best_target=UniversalEntity(
                    x=400,
                    y=300,
                    w=40,
                    h=40,
                    entity_type="target",
                    click_x=420,
                    click_y=320,
                ),
            ),
        ),
    ]

    table = Table(
        title="Jev System One Universal Gaming Benchmark",
        header_style="bold magenta",
    )
    table.add_column("Profile ID", style="bold white")
    table.add_column("Jev Action", style="bold cyan")
    table.add_column("Threat / Urgency", style="bold yellow")
    table.add_column("Confidence", style="bold green")
    table.add_column("Latency", style="dim white")

    for pid, scene in tests:
        prof = pm.get_profile(pid)
        if not prof:
            continue
        with console.status(f"[bold green]Evaluating {prof.name}...[/]"):
            res = brain.query_jev_universal(prof, scene)
            if res:
                table.add_row(
                    prof.id,
                    res["action"].upper(),
                    f"{round(res['threat_score'] * 100, 1)}%",
                    f"{round(res['confidence'] * 100, 1)}%",
                    f"{res['latency_ms']} ms",
                )
            else:
                table.add_row(prof.id, "[red]FAILED[/red]", "-", "-", "-")

    console.print(table)


def run_headless(
    profile_id: str = "runner_dino",
    snap_keyword: str = "",
    armed: bool = True,
):
    """Run autonomous pilot in terminal without GUI."""
    from pilot_core import PilotCore

    core = PilotCore()
    core.set_profile(profile_id)
    target_kw = snap_keyword or core.current_profile.default_window_keyword

    if target_kw:
        console.print(
            f"[bold cyan]🔍 Snapping to window matching '{target_kw}'...[/bold cyan]"
        )
        if core.vision.snap_to_window(target_kw):
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
            f"[bold green]🚀 Universal Jev-GamePilot Headless Runner[/bold green]\n"
            f"Active Profile: [bold cyan]{core.current_profile.name}[/bold cyan]\n"
            f"Genre: [bold yellow]{core.current_profile.category.upper()}[/bold yellow]\n"
            f"Inputs: [bold {'red' if armed else 'yellow'}]{'ARMED (Live Controls)' if armed else 'MONITOR ONLY'}[/]\n"
            f"Press [bold white on red] CTRL+C [/] to stop at any time.",
            border_style="cyan",
        )
    )

    core.start(arm_inputs=armed)
    try:
        while True:
            time.sleep(0.5)
            dec = core.last_decision
            act = dec.get("action", "wait").upper()
            console.print(
                f"[dim]{round(core.fps, 1)} FPS[/dim] | Action: [bold cyan]{act:<14}[/] | Actions Executed: [green]{core.total_actions}[/]"
            )
    except KeyboardInterrupt:
        console.print("\n[bold red]Stopping Jev-GamePilot...[/bold red]")
        core.stop()


def main():
    parser = argparse.ArgumentParser(
        description="Universal Jev-GamePilot: Autonomous AI Gaming Agent powered by TypeSafe Jev System One"
    )
    sub = parser.add_subparsers(dest="command", help="Commands")

    sub.add_parser("hud", help="Launch Universal Cyber Desktop GUI HUD")
    sub.add_parser("profiles", help="List all available game profiles")
    sub.add_parser("test-brain", help="Run Universal Jev System One benchmark")

    play_p = sub.add_parser("play", help="Run autonomous pilot in terminal")
    play_p.add_argument(
        "--profile",
        default="runner_dino",
        help="Game profile ID (e.g. runner_dino, runner_3lane, flappy_tap, aim_clicker, retro_platformer)",
    )
    play_p.add_argument(
        "--snap",
        default="",
        help="Window title keyword to snap to (e.g. roblox, bluestacks, chrome, steam)",
    )
    play_p.add_argument(
        "--monitor",
        action="store_true",
        help="Monitor only, do not dispatch inputs",
    )

    args = parser.parse_args()

    if args.command == "profiles":
        list_profiles()
    elif args.command == "test-brain":
        test_brain()
    elif args.command == "play":
        run_headless(
            profile_id=args.profile, snap_keyword=args.snap, armed=not args.monitor
        )
    else:
        run_hud()


if __name__ == "__main__":
    main()
