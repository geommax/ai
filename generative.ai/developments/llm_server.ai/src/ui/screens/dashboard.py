"""Dashboard screen — server status & quick overview."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal, Container
from textual.widgets import Static, Button, Label
from textual.reactive import reactive


class StatusCard(Static):
    """A styled info card."""

    DEFAULT_CSS = """
    StatusCard {
        height: auto;
        background: #24283b;
        border: solid #414868;
        padding: 1 2;
        margin: 0 0 1 0;
    }
    """


class DashboardScreen(Container):
    """Main dashboard showing server & system overview."""

    DEFAULT_CSS = """
    DashboardScreen {
        padding: 1 2;
        overflow-y: auto;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static("📊  Dashboard", classes="panel-title")

        # ── Server Status ──────────────────────────────────────────
        with StatusCard():
            yield Static("🖥️  [b]Server Status[/b]", markup=True)
            yield Static("", id="srv-status")
            yield Static("", id="srv-endpoint")
            yield Horizontal(
                Button("▶  Start Server", id="btn-start-server", variant="success"),
                Button("⏹  Stop Server", id="btn-stop-server", variant="error"),
                classes="btn-row",
            )

        # ── Model Info ─────────────────────────────────────────────
        with StatusCard():
            yield Static("🧠  [b]Active Model[/b]", markup=True)
            yield Static("", id="srv-model")

        # ── System Info ────────────────────────────────────────────
        with StatusCard():
            yield Static("💻  [b]System Information[/b]", markup=True)
            yield Static("", id="sys-device")
            yield Static("", id="sys-cache")
            yield Static("", id="sys-keys")

    # ── Lifecycle ──────────────────────────────────────────────────
    def on_mount(self) -> None:
        self.refresh_info()

    def refresh_info(self) -> None:
        app = self.app  # type: ignore[attr-defined]

        # Server
        if hasattr(app, "server_thread") and app.server_thread and app.server_thread.is_running:
            self.query_one("#srv-status", Static).update(
                "  Status:   [green]● Running[/green]"
            )
        else:
            self.query_one("#srv-status", Static).update(
                "  Status:   [red]● Stopped[/red]"
            )

        cfg = app.config
        self.query_one("#srv-endpoint", Static).update(
            f"  Endpoint: http://{cfg.host}:{cfg.port}"
        )

        # Model
        engine = app.inference_engine
        if engine.is_loaded:
            self.query_one("#srv-model", Static).update(
                f"  [green]✓[/green] {engine.model_id}"
            )
        else:
            self.query_one("#srv-model", Static).update(
                "  [dim]No model loaded[/dim]"
            )

        # System
        dev = engine.device_info()
        self.query_one("#sys-device", Static).update(
            f"  Device: {dev['type']}  |  {dev['name']}  |  VRAM {dev['memory']}"
        )
        self.query_one("#sys-cache", Static).update(
            f"  Cache:  {app.model_manager.total_cache_size()}"
        )
        self.query_one("#sys-keys", Static).update(
            f"  API Keys: {app.db.key_count()} active"
        )

    # ── Button handlers ────────────────────────────────────────────
    def on_button_pressed(self, event: Button.Pressed) -> None:
        app = self.app  # type: ignore[attr-defined]

        if event.button.id == "btn-start-server":
            if app.server_thread and app.server_thread.is_running:
                app.notify("Server is already running", severity="warning")
                return
            app.start_server()
            app.notify("Server started ✓")
            self.refresh_info()

        elif event.button.id == "btn-stop-server":
            if not app.server_thread or not app.server_thread.is_running:
                app.notify("Server is not running", severity="warning")
                return
            app.stop_server()
            app.notify("Server stopped")
            self.refresh_info()
