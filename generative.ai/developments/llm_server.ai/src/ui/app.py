"""LLM Server.AI — Main Textual TUI Application."""

from __future__ import annotations

from pathlib import Path

from textual import work
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

from src.config import ServerConfig, DB_FILE
from src.database import Database
from src.llms import ModelManager, InferenceEngine
from src.apis import create_api, ServerThread

from src.ui.screens.dashboard import DashboardScreen
from src.ui.screens.models import ModelsScreen
from src.ui.screens.keys import APIKeysScreen
from src.ui.screens.testing import TestingScreen
from src.ui.screens.tuning import TuningScreen


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
        Binding("q", "quit", "Quit", show=True),
    ]

    # ── Shared state ───────────────────────────────────────────────
    def __init__(self) -> None:
        super().__init__()
        self.config = ServerConfig.load()
        self.db = Database(str(DB_FILE))
        self.model_manager = ModelManager()
        self.inference_engine = InferenceEngine()
        self.server_thread: ServerThread | None = None
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
                yield Button("⚙️   Tuning", id="nav-tuning", classes="nav-btn")

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

        yield Footer()

    # ── Lifecycle ──────────────────────────────────────────────────
    def on_mount(self) -> None:
        self._update_sidebar_status()
        # Auto-restore previous session
        if self.config.auto_restore:
            self._auto_restore()

    # ── Navigation ─────────────────────────────────────────────────
    def action_switch_screen(self, name: str) -> None:
        """Switch the content area to the given screen name."""
        switcher = self.query_one("#main-content", ContentSwitcher)
        switcher.current = name
        self._current_nav = name

        # Highlight active button
        for btn in self.query(".nav-btn"):
            btn.remove_class("-active")
        target = self.query_one(f"#nav-{name}", Button)
        target.add_class("-active")

        # Refresh the target screen when switching
        self._refresh_active_screen(name)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle sidebar navigation buttons."""
        bid = event.button.id or ""
        if bid.startswith("nav-"):
            screen_name = bid.replace("nav-", "")
            self.action_switch_screen(screen_name)
            # Remove focus from the nav button so only -active highlight shows
            self.set_focus(None)
            event.stop()

    def _refresh_active_screen(self, name: str) -> None:
        """Call refresh on the currently visible screen."""
        try:
            if name == "dashboard":
                self.query_one(DashboardScreen).refresh_info()
            elif name == "testing":
                self.query_one(TestingScreen)._refresh_model_label()
        except Exception:
            pass

    # ── Server control ─────────────────────────────────────────────
    def start_server(self) -> None:
        api = create_api(self.inference_engine, self.db, self.config)
        self.server_thread = ServerThread(
            api, host=self.config.host, port=self.config.port
        )
        self.server_thread.start()
        self._update_sidebar_status()

    def stop_server(self) -> None:
        if self.server_thread:
            self.server_thread.stop()
            self.server_thread = None
        self._update_sidebar_status()

    # ── Auto-restore previous session ───────────────────────
    @work(thread=True, exclusive=True, group="restore")
    def _auto_restore(self) -> None:
        """Reload the last model and restart the server if they were active."""
        import time

        # 1) Restore model
        if self.config.active_model:
            model_id = self.config.active_model
            self.call_from_thread(
                self.notify, f"Restoring model: {model_id}..."
            )
            try:
                self.inference_engine.load_model(model_id)
                self.call_from_thread(
                    self.notify, f"Model restored: {model_id} ✓"
                )
            except Exception as exc:
                self.call_from_thread(
                    self.notify,
                    f"Could not restore model: {exc}",
                    severity="warning",
                )
                self.config.active_model = ""

        # 2) Restore server
        if self.config.server_was_running:
            self.call_from_thread(self.start_server)
            self.call_from_thread(
                self.notify, "Server auto-started ✓"
            )

        self.call_from_thread(self._update_sidebar_status)

    # ── Sidebar status ─────────────────────────────────────────────
    def _update_sidebar_status(self) -> None:
        try:
            srv = self.query_one("#nav-srv-status", Static)
            mdl = self.query_one("#nav-model-status", Static)

            if self.server_thread and self.server_thread.is_running:
                srv.update(f"[green]● Server ON[/green]  :{self.config.port}")
            else:
                srv.update("[red]● Server OFF[/red]")

            if self.inference_engine.is_loaded:
                mdl.update(f"[green]●[/green] {self.inference_engine.model_id}")
            else:
                mdl.update("[dim]No model[/dim]")
        except Exception:
            pass  # Widgets not yet mounted

    # ── Cleanup ────────────────────────────────────────────────────
    def on_unmount(self) -> None:
        # Save running state *before* tearing down
        self.config.server_was_running = (
            self.server_thread is not None and self.server_thread.is_running
        )
        self.config.save()

        self.stop_server()
