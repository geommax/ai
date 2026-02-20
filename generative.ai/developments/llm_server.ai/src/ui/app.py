"""LLM Server.AI — Main Textual TUI Application (thin client).

The TUI is purely a *frontend*.  All heavy state (loaded model, API
server, downloads) lives in the background daemon process.  The TUI
communicates with it via ``DaemonClient`` over a Unix socket.
"""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, Container
from textual.widgets import (
    Header,
    Footer,
    Static,
    Button,
    ContentSwitcher,
)

from src.daemon.client import DaemonClient, DaemonDisconnected

from src.ui.screens.dashboard import DashboardScreen
from src.ui.screens.models import ModelsScreen
from src.ui.screens.keys import APIKeysScreen
from src.ui.screens.testing import TestingScreen
from src.ui.screens.tuning import TuningScreen
from src.ui.screens.settings import SettingsScreen


class LLMServerApp(App):
    """Terminal-based management UI for a local LLM server."""

    TITLE = "LLM Server.AI"
    SUB_TITLE = "Local LLM Inference Server"
    CSS_PATH = "app.tcss"

    BINDINGS = [
        Binding("d", "switch_screen('dashboard')", "Dashboard", show=True),
        Binding("m", "switch_screen('models')", "Models", show=True),
        Binding("k", "switch_screen('keys')", "API Keys", show=True),
        Binding("t", "switch_screen('testing')", "Testing", show=True),
        Binding("o", "switch_screen('tuning')", "Tuning", show=True),
        Binding("s", "switch_screen('settings')", "Settings", show=True),
        Binding("q", "quit", "Quit", show=True),
    ]

    # ── Shared state ───────────────────────────────────────────────
    def __init__(self) -> None:
        super().__init__()
        self.client = DaemonClient()
        self._current_nav: str = "dashboard"

    # ── Compose ────────────────────────────────────────────────────
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        with Horizontal(id="app-grid"):
            # Sidebar
            with Vertical(id="sidebar"):
                yield Static("🤖 LLM Server.AI", classes="nav-title")
                yield Static("v1.0 — Local Inference", classes="nav-subtitle")
                yield Static("─" * 26, classes="nav-sep")

                yield Button(
                    "📊  Dashboard", id="nav-dashboard", classes="nav-btn -active"
                )
                yield Button("🧠  Models", id="nav-models", classes="nav-btn")
                yield Button("🔑  API Keys", id="nav-keys", classes="nav-btn")
                yield Button("🧪  Testing", id="nav-testing", classes="nav-btn")
                yield Button("🎛️   Tuning", id="nav-tuning", classes="nav-btn")
                yield Button("⚙️   Settings", id="nav-settings", classes="nav-btn")

                yield Static("─" * 26, classes="nav-sep")

                # Quick status in sidebar
                with Container(classes="nav-status"):
                    yield Static("", id="nav-srv-status")
                    yield Static("", id="nav-model-status")

            # Main content area
            with ContentSwitcher(initial="dashboard", id="main-content"):
                yield DashboardScreen(id="dashboard")
                yield ModelsScreen(id="models")
                yield APIKeysScreen(id="keys")
                yield TestingScreen(id="testing")
                yield TuningScreen(id="tuning")
                yield SettingsScreen(id="settings")

        yield Footer()

    # ── Lifecycle ──────────────────────────────────────────────────
    def on_mount(self) -> None:
        self.update_sidebar_status()

    def on_unmount(self) -> None:
        """TUI exit — do NOT touch daemon state."""
        pass  # Daemon keeps running

    # ── Navigation ─────────────────────────────────────────────────
    def action_switch_screen(self, name: str) -> None:
        switcher = self.query_one("#main-content", ContentSwitcher)
        switcher.current = name
        self._current_nav = name

        for btn in self.query(".nav-btn"):
            btn.remove_class("-active")
        target = self.query_one(f"#nav-{name}", Button)
        target.add_class("-active")

        self._refresh_active_screen(name)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id or ""
        if bid.startswith("nav-"):
            screen_name = bid.replace("nav-", "")
            self.action_switch_screen(screen_name)
            self.set_focus(None)
            event.stop()

    def _refresh_active_screen(self, name: str) -> None:
        try:
            if name == "dashboard":
                self.query_one(DashboardScreen).refresh_info()
            elif name == "testing":
                self.query_one(TestingScreen)._refresh_model_label()
            elif name == "tuning":
                self.query_one(TuningScreen)._load_from_config()
            elif name == "settings":
                self.query_one(SettingsScreen)._load_settings()
        except Exception:
            pass

    # ── Sidebar status (called from anywhere) ──────────────────────
    def update_sidebar_status(self) -> None:
        try:
            status = self.client.get_status()
        except DaemonDisconnected:
            try:
                self.query_one("#nav-srv-status", Static).update(
                    "[red]● Daemon offline[/red]"
                )
                self.query_one("#nav-model-status", Static).update("")
            except Exception:
                pass
            return

        try:
            srv = self.query_one("#nav-srv-status", Static)
            mdl = self.query_one("#nav-model-status", Static)

            if status.get("server_running"):
                port = status.get("server_port", "?")
                srv.update(f"[green]● Server ON[/green]  :{port}")
            else:
                srv.update("[red]● Server OFF[/red]")

            if status.get("model_loaded"):
                mdl.update(f"[green]●[/green] {status.get('model_id', '?')}")
            elif status.get("loading_model"):
                mdl.update(f"[yellow]⏳[/yellow] Loading {status['loading_model']}…")
            else:
                mdl.update("[dim]No model[/dim]")
        except Exception:
            pass
