"""
All Games Catalog & Browser Dialog for Jev-GamePilot.
Provides a visual library of all 28+ supported games across Phone (ADB) and PC,
with real-time keyword search, category filtering, and instant one-click activation.
"""

from typing import Callable, List, Optional
import customtkinter as ctk

from profile_manager import GameProfile, ProfileManager


class AllGamesDialog(ctk.CTkToplevel):
    def __init__(
        self,
        parent,
        profile_mgr: ProfileManager,
        on_game_selected: Callable[[GameProfile], None],
    ):
        super().__init__(parent)

        self.title("🎮 Universal Game Library // 28+ Autonomous AI Profiles")
        self.geometry("860x620")
        self.minsize(760, 520)
        self.configure(fg_color="#0d0e15")

        self.profile_mgr = profile_mgr
        self.on_game_selected = on_game_selected
        self.all_profiles: List[GameProfile] = self.profile_mgr.list_profiles()

        # Lift window to top
        self.attributes("-topmost", True)
        self.after(100, lambda: self.attributes("-topmost", False))
        self.focus()

        self._build_ui()
        self._refresh_list()

    def _build_ui(self):
        # Header
        header = ctk.CTkFrame(self, fg_color="#141622", corner_radius=0, height=65)
        header.pack(fill="x", side="top")

        title_lbl = ctk.CTkLabel(
            header,
            text="⚡ UNIVERSAL GAME CATALOG",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="#00ffcc",
        )
        title_lbl.pack(side="left", padx=20, pady=12)

        sub_lbl = ctk.CTkLabel(
            header,
            text=f"// {len(self.all_profiles)} AUTONOMOUS AI PROFILES READY",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#70758a",
        )
        sub_lbl.pack(side="left", padx=5)

        close_btn = ctk.CTkButton(
            header,
            text="✕ Close",
            font=ctk.CTkFont(size=11, weight="bold"),
            width=80,
            height=28,
            fg_color="#1a1c26",
            hover_color="#2b2f42",
            command=self.destroy,
        )
        close_btn.pack(side="right", padx=20)

        # Controls row: Search entry + Category Filter
        ctrl_bar = ctk.CTkFrame(self, fg_color="#10111a", corner_radius=8)
        ctrl_bar.pack(fill="x", padx=20, pady=(15, 10))

        # Search box
        self.search_var = ctk.StringVar()
        self.search_var.trace_add("write", lambda *args: self._refresh_list())

        self.search_entry = ctk.CTkEntry(
            ctrl_bar,
            textvariable=self.search_var,
            placeholder_text="🔍 Search games, genres, or keywords (e.g. pool, solitaire, clash, balatro)...",
            font=ctk.CTkFont(size=12),
            height=36,
            fg_color="#181a26",
            border_color="#282c3c",
            text_color="#ffffff",
        )
        self.search_entry.pack(fill="x", padx=12, pady=(10, 8))

        # Category Filter Pills
        self.cat_filter = ctk.CTkSegmentedButton(
            ctrl_bar,
            values=[
                "All Games",
                "📱 Mobile (ADB)",
                "🖥️ PC Desktop",
                "♠️ Cards & Strategy",
                "🎱 Sports & Billiards",
                "🕹️ Arcade & Action",
            ],
            command=lambda v: self._refresh_list(),
            fg_color="#141620",
            selected_color="#00d26a",
            selected_hover_color="#00b058",
            unselected_color="#1a1c26",
            unselected_hover_color="#2b2f42",
            text_color="#ffffff",
            font=ctk.CTkFont(size=11, weight="bold"),
        )
        self.cat_filter.set("All Games")
        self.cat_filter.pack(fill="x", padx=12, pady=(0, 10))

        # Scrollable Game Cards Container
        self.scroll_container = ctk.CTkScrollableFrame(
            self, fg_color="transparent", corner_radius=6
        )
        self.scroll_container.pack(fill="both", expand=True, padx=20, pady=(0, 15))

    def _refresh_list(self):
        # Clear existing cards
        for widget in self.scroll_container.winfo_children():
            widget.destroy()

        query = self.search_var.get().lower().strip()
        cat = self.cat_filter.get()

        matching = []
        for p in self.all_profiles:
            is_mobile = p.id.startswith("mobile_") or p.id in ["runner_3lane"]
            is_pc = p.id.startswith("pc_") or p.id in ["runner_dino", "chess_copilot", "flappy_tap", "aim_clicker", "retro_platformer"]

            # Category filter
            if cat == "📱 Mobile (ADB)" and not is_mobile:
                continue
            if cat == "🖥️ PC Desktop" and not is_pc:
                continue
            if cat == "♠️ Cards & Strategy" and p.category != "strategy" and "card" not in p.id and "solitaire" not in p.id:
                continue
            if cat == "🎱 Sports & Billiards" and p.category != "sports" and "pool" not in p.id and "fifa" not in p.id:
                continue
            if cat == "🕹️ Arcade & Action" and p.category not in ["arcade", "runner", "platformer", "clicker"]:
                continue

            # Query search
            blob = f"{p.name} {p.id} {p.description} {p.category}".lower()
            if query and query not in blob:
                continue

            matching.append(p)

        if not matching:
            empty_lbl = ctk.CTkLabel(
                self.scroll_container,
                text=f"No game profiles found matching '{query}'.\nTry searching for 'pool', 'solitaire', 'clash', or 'balatro'.",
                font=ctk.CTkFont(size=13),
                text_color="#70758a",
                justify="center",
            )
            empty_lbl.pack(pady=40)
            return

        # Render cards
        for p in matching:
            self._render_game_card(p)

    def _render_game_card(self, profile: GameProfile):
        is_mobile = profile.id.startswith("mobile_") or profile.id in ["runner_3lane"]
        card = ctk.CTkFrame(
            self.scroll_container, fg_color="#141622", corner_radius=8
        )
        card.pack(fill="x", pady=4, padx=2)

        inner = ctk.CTkFrame(card, fg_color="transparent")
        inner.pack(fill="x", padx=12, pady=10)

        # Left Info
        info_box = ctk.CTkFrame(inner, fg_color="transparent")
        info_box.pack(side="left", fill="both", expand=True)

        # Title Row
        title_row = ctk.CTkFrame(info_box, fg_color="transparent")
        title_row.pack(anchor="w")

        title_lbl = ctk.CTkLabel(
            title_row,
            text=profile.name,
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#ffffff",
        )
        title_lbl.pack(side="left", padx=(0, 10))

        # Platform Badge
        platform_txt = "📱 ANDROID (ADB)" if is_mobile else "🖥️ PC WINDOWS"
        platform_color = "#00d26a" if is_mobile else "#00bfff"
        plat_badge = ctk.CTkLabel(
            title_row,
            text=platform_txt,
            font=ctk.CTkFont(size=9, weight="bold"),
            text_color=platform_color,
            fg_color="#1a1c28",
            corner_radius=4,
            padx=6,
            pady=1,
        )
        plat_badge.pack(side="left", padx=(0, 6))

        # Genre Badge
        genre_badge = ctk.CTkLabel(
            title_row,
            text=profile.category.upper(),
            font=ctk.CTkFont(size=9, weight="bold"),
            text_color="#ffaa00",
            fg_color="#1a1c28",
            corner_radius=4,
            padx=6,
            pady=1,
        )
        genre_badge.pack(side="left")

        # Description
        desc_lbl = ctk.CTkLabel(
            info_box,
            text=profile.description[:110] + ("..." if len(profile.description) > 110 else ""),
            font=ctk.CTkFont(size=11),
            text_color="#888c9e",
            justify="left",
            wraplength=520,
        )
        desc_lbl.pack(anchor="w", pady=(3, 0))

        # Right Launch Button
        btn_launch = ctk.CTkButton(
            inner,
            text="🚀 Launch Pilot",
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color="#00d26a",
            hover_color="#00b058",
            text_color="#000000",
            width=110,
            height=34,
            command=lambda p=profile: self._select_and_close(p),
        )
        btn_launch.pack(side="right", padx=(10, 0))

    def _select_and_close(self, profile: GameProfile):
        self.on_game_selected(profile)
        self.destroy()
