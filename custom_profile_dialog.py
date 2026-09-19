"""
Custom Game Profile Creator Dialog for Universal Jev-GamePilot.
Allows users to create, configure, and save autonomous AI profiles for ANY game.
"""

import customtkinter as ctk
from profile_manager import GameAction, GameProfile, ProfileManager


class CustomProfileDialog(ctk.CTkToplevel):
    def __init__(self, master, profile_mgr: ProfileManager, on_created_cb=None):
        super().__init__(master)

        self.title("Create Custom Game Profile // Jev-GamePilot")
        self.geometry("540x620")
        self.minsize(500, 560)
        self.configure(fg_color="#0e1017")

        self.profile_mgr = profile_mgr
        self.on_created_cb = on_created_cb

        # Stay on top of parent HUD
        self.attributes("-topmost", True)

        self._build_ui()

    def _build_ui(self):
        # Header
        hdr = ctk.CTkFrame(self, fg_color="#141724", height=50, corner_radius=0)
        hdr.pack(fill="x", side="top")

        ctk.CTkLabel(
            hdr,
            text="➕ NEW GAME PROFILE CREATOR",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#00ffcc",
        ).pack(side="left", padx=15, pady=12)

        # Content Form
        form = ctk.CTkScrollableFrame(self, fg_color="transparent")
        form.pack(fill="both", expand=True, padx=20, pady=15)

        # 1. Profile Name
        ctk.CTkLabel(
            form,
            text="Game Title",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#c0c0c0",
        ).pack(anchor="w", pady=(0, 2))

        self.name_entry = ctk.CTkEntry(
            form,
            placeholder_text="e.g. Roblox Obby, Minecraft Jump, Cookie Clicker",
            fg_color="#1a1c28",
            border_color="#2c3044",
            height=32,
        )
        self.name_entry.pack(fill="x", pady=(0, 10))

        # 2. Category Dropdown
        ctk.CTkLabel(
            form,
            text="Game Genre",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#c0c0c0",
        ).pack(anchor="w", pady=(0, 2))

        self.category_opt = ctk.CTkOptionMenu(
            form,
            values=["runner", "platformer", "clicker", "arcade", "custom"],
            fg_color="#1a1c28",
            button_color="#2b2f42",
            dropdown_fg_color="#1a1c28",
            height=30,
        )
        self.category_opt.pack(fill="x", pady=(0, 10))

        # 3. Target Window Keyword
        ctk.CTkLabel(
            form,
            text="Target Window Keyword (for auto-snapping)",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#c0c0c0",
        ).pack(anchor="w", pady=(0, 2))

        self.win_entry = ctk.CTkEntry(
            form,
            placeholder_text="e.g. roblox, bluestacks, chrome, steam",
            fg_color="#1a1c28",
            border_color="#2c3044",
            height=32,
        )
        self.win_entry.pack(fill="x", pady=(0, 10))

        # 4. Action Keybindings
        ctk.CTkLabel(
            form,
            text="Game Actions & Keybindings (Evaluated by Jev)",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#00ffcc",
        ).pack(anchor="w", pady=(10, 4))

        ctk.CTkLabel(
            form,
            text="Format: [Action Name] | [Key to Press] | [Semantic Description for Jev]",
            font=ctk.CTkFont(size=10),
            text_color="#70758a",
        ).pack(anchor="w", pady=(0, 6))

        self.action_rows_frame = ctk.CTkFrame(form, fg_color="transparent")
        self.action_rows_frame.pack(fill="x", pady=(0, 8))

        self.action_entries = []
        # Pre-fill with 2 standard actions
        self._add_action_row("jump", "space", "Jump over oncoming hazards")
        self._add_action_row("duck", "down", "Duck or slide under overhead obstacles")
        self._add_action_row("wait", "none", "Safe horizon, hold current position")

        add_row_btn = ctk.CTkButton(
            form,
            text="+ Add Another Action",
            font=ctk.CTkFont(size=11),
            fg_color="#1f2232",
            hover_color="#2c3044",
            text_color="#00ffcc",
            height=26,
            command=lambda: self._add_action_row("", "", ""),
        )
        add_row_btn.pack(anchor="w", pady=(0, 15))

        # Bottom Buttons
        btn_box = ctk.CTkFrame(self, fg_color="#141724", height=55)
        btn_box.pack(fill="x", side="bottom")

        cancel_btn = ctk.CTkButton(
            btn_box,
            text="Cancel",
            fg_color="#1f2230",
            hover_color="#2c3044",
            command=self.destroy,
            width=100,
            height=36,
        )
        cancel_btn.pack(side="left", padx=15, pady=10)

        save_btn = ctk.CTkButton(
            btn_box,
            text="💾 Save & Activate Profile",
            font=ctk.CTkFont(weight="bold"),
            fg_color="#00d26a",
            hover_color="#00b058",
            text_color="#000000",
            command=self._save_profile,
            width=200,
            height=36,
        )
        save_btn.pack(side="right", padx=15, pady=10)

    def _add_action_row(self, name="", key="", desc=""):
        row = ctk.CTkFrame(self.action_rows_frame, fg_color="#141724", corner_radius=6)
        row.pack(fill="x", pady=3)

        name_e = ctk.CTkEntry(
            row,
            placeholder_text="action_name",
            width=90,
            height=28,
            fg_color="#1a1c28",
            border_color="#2c3044",
        )
        name_e.insert(0, name)
        name_e.pack(side="left", padx=(5, 3), pady=4)

        key_e = ctk.CTkEntry(
            row,
            placeholder_text="key (space/a/click)",
            width=90,
            height=28,
            fg_color="#1a1c28",
            border_color="#2c3044",
        )
        key_e.insert(0, key)
        key_e.pack(side="left", padx=3, pady=4)

        desc_e = ctk.CTkEntry(
            row,
            placeholder_text="Semantic description for Jev...",
            height=28,
            fg_color="#1a1c28",
            border_color="#2c3044",
        )
        desc_e.insert(0, desc)
        desc_e.pack(side="left", fill="x", expand=True, padx=3, pady=4)

        del_b = ctk.CTkButton(
            row,
            text="✕",
            width=24,
            height=24,
            fg_color="transparent",
            hover_color="#ff2a55",
            text_color="#ff2a55",
            command=lambda: self._remove_action_row(row),
        )
        del_b.pack(side="right", padx=4)

        self.action_entries.append((row, name_e, key_e, desc_e))

    def _remove_action_row(self, row_frame):
        self.action_entries = [
            item for item in self.action_entries if item[0] != row_frame
        ]
        row_frame.destroy()

    def _save_profile(self):
        title = self.name_entry.get().strip()
        if not title:
            return

        genre = self.category_opt.get()
        win_kw = self.win_entry.get().strip()
        prof_id = "custom_" + title.lower().replace(" ", "_")[:18]

        actions = []
        for _, name_e, key_e, desc_e in self.action_entries:
            a_name = name_e.get().strip()
            a_key = key_e.get().strip()
            a_desc = desc_e.get().strip()
            if a_name:
                is_click = "click" in a_key.lower() or "mouse" in a_key.lower()
                actions.append(
                    GameAction(
                        name=a_name,
                        key=a_key if a_key else "none",
                        description=a_desc if a_desc else f"Perform {a_name}",
                        is_mouse_click=is_click,
                    )
                )

        if not actions:
            actions.append(
                GameAction(
                    name="act",
                    key="space",
                    description="Perform primary game action",
                )
            )

        new_prof = self.profile_mgr.add_custom_profile(
            profile_id=prof_id,
            name=f"⭐ {title}",
            category=genre,
            description=f"Custom AI profile for {title}",
            actions=actions,
            window_keyword=win_kw,
        )

        if self.on_created_cb:
            self.on_created_cb(new_prof)

        self.destroy()
