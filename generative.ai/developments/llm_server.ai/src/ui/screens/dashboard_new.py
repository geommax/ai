"""Dashboard screen — server status & quick overview (daemon client)."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal, Container
from textual.widgets import Static, Button

from src.daemon.client import DaemonDisconnected


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
            with Horizontal(
                classes="btn-row",
            ):
                yield Button("▶  Start Server", id="btn-start-server", variant="success")
                yield Button("⏹  Stop Server", id="btn-stop-server", variant="error")

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
        client = self.app.client  # type: ignore[attr-defined]
        try:
            status = client.get_status()
        except DaemonDisconnected:
            self.query_one("#srv-status", Static).update(
                "  Status:   [red]● Daemon offline[/red]"
            )
            return

        # Server
        if status.get("server_running"):
            self.query_one("#srv-status", Static).update(
                "  Status:   [green]● Running[/green]"
            )
        else:
            self.query_one("#srv-status", Static).update(
                "  Status:   [red]● Stopped[/red]"
            )

        host = status.get("server_host", "127.0.0.1")
        port = status.get("server_port", 8000)
        self.query_one("#srv-endpoint", Static).update(
            f"  Endpoint: http://{host}:{port}"
        )

        # Model
        if status.get("model_loaded"):
            self.query_one("#srv-model", Static).update(
                f"  [green]✓[/green] {status.get('model_id', '?')}"
            )
        elif status.get("loading_model"):
            self.query_one("#srv-model", Static).update(
                f"  [yellow]⏳[/yellow] Loading {status['loading_model']}…"
            )
        else:
            self.query_one("#srv-model", Static).update(
                "  [dim]No model loaded[/dim]"
            )

        # System
        try:
            dev = client.device_info()
            self.query_one("#sys-device", Static).update(
                f"  Device: {dev.get('type', '?')}  |  {dev.get('name', '?')}  |  VRAM {dev.get('memory', '?')}"
            )
        except Exception:
            self.query_one("#sys-device", Static).update("  Device: [dim]unavailable[/dim]")

        try:
            cache = client.cache_size()
            self.query_one("#sys-cache", Static).update(f"  Cache:  {cache}")
        except Exception:
            pass

        try:
            kc = client.key_count()
            self.query_one("#sys-keys", Static).update(f"  API Keys: {kc} active")
        except Exception:
            pass

    # ── Button handlers ────────────────────────────────────────────
    def on_button_pressed(self, event: Button.Pressed) -> None:
        app = self.app  # type: ignore[attr-defined]
        client = app.client

        if event.button.id == "btn-start-server":
            try:
                result = client.start_server()
                if result["ok"]:
                    app.notify("Server started ✓")
                else:
                    app.notify(result.get("error", "Failed"), severity="warning")
            except DaemonDisconnected:
                app.notify("Daemon offline", severity="error")
            self.refresh_info()
            app.update_sidebar_status()

        elif event.button.id == "btn-stop-server":
            try:
                result = client.stop_server()
                if result["ok"]:
                    app.notify("Server stopped")
                else:
                    app.notify(result.get("error", "Failed"), severity="warning")
            except DaemonDisconnected:
                app.notify("Daemon offline", severity="error")
            self.refresh_info()
            app.update_sidebar_status()
