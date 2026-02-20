"""App Settings screen — HuggingFace login, server config, preferences."""

from __future__ import annotations

from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Static, Button, Input, Switch, Label, Select

from src.daemon.client import DaemonDisconnected


# ── Reusable row widget ───────────────────────────────────────────────
class SettingRow(Horizontal):
    """A single setting row: label + control + hint."""

    DEFAULT_CSS = """
    SettingRow {
        height: 3;
        margin: 0 0 1 0;
        align: left middle;
    }
    SettingRow > Label {
        width: 26;
        padding: 0 1;
    }
    SettingRow > Input {
        width: 30;
    }
    SettingRow > .hint {
        width: 1fr;
        padding: 0 2;
        color: #565f89;
    }
    """


class SettingsScreen(Container):
    """App-wide settings: HuggingFace account, server, preferences."""

    DEFAULT_CSS = """
    SettingsScreen {
        padding: 1 2;
        overflow-y: auto;
    }
    .settings-section {
        background: #24283b;
        border: solid #414868;
        padding: 1 2;
        margin: 0 0 1 0;
        height: auto;
    }
    .settings-section-title {
        text-style: bold;
        color: #7aa2f7;
        margin: 0 0 1 0;
    }
    #hf-login-status {
        height: auto;
        margin: 1 0 0 0;
        padding: 0 1;
    }
    #settings-status {
        height: auto;
        margin: 1 0;
        color: #9ece6a;
    }
    .settings-btn-row {
        height: 3;
        margin: 1 0 0 0;
    }
    .switch-row {
        height: 3;
        margin: 0 0 1 0;
        align: left middle;
    }
    .switch-row > Label {
        width: 26;
        padding: 0 1;
    }
    .switch-row > .hint {
        width: 1fr;
        padding: 0 2;
        color: #565f89;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static("🛠️  App Settings", classes="panel-title")

        # ══════════════════════════════════════════════════════════
        #  Section 1: HuggingFace Account
        # ══════════════════════════════════════════════════════════
        with Container(classes="settings-section"):
            yield Static(
                "🤗  HuggingFace Account",
                classes="settings-section-title",
                markup=True,
            )
            yield Static(
                "[dim]Hugging Face Token ID is required for gated model downloads[/dim]",
                markup=True,
            )

            with SettingRow():
                yield Label("Access Token")
                yield Input(
                    "",
                    id="hf-token-input",
                    placeholder="hf_xxxxxxxxxxxxxxxxxx",
                    password=True,
                )
                yield Static("huggingface.co/settings/tokens", classes="hint")

            with Horizontal(classes="settings-btn-row"):
                yield Button(
                    "🔑  Login", id="btn-hf-login", variant="success"
                )
                yield Button(
                    "🚪  Logout", id="btn-hf-logout", variant="error"
                )
                yield Button(
                    "👁  Show/Hide Token",
                    id="btn-hf-toggle",
                    variant="default",
                )

            yield Static("", id="hf-login-status")

        # ══════════════════════════════════════════════════════════
        #  Section 2: Server Configuration
        # ══════════════════════════════════════════════════════════
        with Container(classes="settings-section"):
            yield Static(
                "🌐  Server Configuration",
                classes="settings-section-title",
                markup=True,
            )
            yield Static(
                "[dim]Set the host and port for the API server[/dim]",
                markup=True,
            )

            with SettingRow():
                yield Label("Host")
                yield Input(
                    "127.0.0.1",
                    id="setting-host",
                    placeholder="127.0.0.1",
                )
                yield Static("IP address (0.0.0.0 = all)", classes="hint")

            with SettingRow():
                yield Label("Port")
                yield Input(
                    "8000",
                    id="setting-port",
                    placeholder="8000",
                )
                yield Static("1024 – 65535  (default: 8000)", classes="hint")

            with Horizontal(classes="settings-btn-row"):
                yield Button(
                    "💾  Save Server Config",
                    id="btn-save-server",
                    variant="success",
                )

        # ══════════════════════════════════════════════════════════
        #  Section 3: Model Storage
        # ══════════════════════════════════════════════════════════
        with Container(classes="settings-section"):
            yield Static(
                "📁  Model Storage Directory",
                classes="settings-section-title",
                markup=True,
            )
            yield Static(
                "[dim]Directory where downloaded model files are stored[/dim]",
                markup=True,
            )

            with SettingRow():
                yield Label("Model Directory")
                yield Input(
                    "",
                    id="setting-model-dir",
                    placeholder="~/.cache/huggingface (default)",
                )
                yield Static("Absolute path or blank = default", classes="hint")

            yield Static("", id="model-dir-info")

            with Horizontal(classes="settings-btn-row"):
                yield Button(
                    "💾  Save Path",
                    id="btn-save-model-dir",
                    variant="success",
                )
                yield Button(
                    "🔄  Reset Default",
                    id="btn-reset-model-dir",
                    variant="warning",
                )

        # ══════════════════════════════════════════════════════════
        #  Section 4: App Preferences
        # ══════════════════════════════════════════════════════════
        with Container(classes="settings-section"):
            yield Static(
                "⚙️  Preferences",
                classes="settings-section-title",
                markup=True,
            )

            # Auto-restore
            with Horizontal(classes="switch-row"):
                yield Label("Auto Restore")
                yield Switch(value=True, id="setting-auto-restore")
                yield Static(
                    "Automatically restore model/server on daemon start",
                    classes="hint",
                )

            # Log level
            with SettingRow():
                yield Label("Log Level")
                yield Select(
                    [
                        ("DEBUG", "DEBUG"),
                        ("INFO", "INFO"),
                        ("WARNING", "WARNING"),
                        ("ERROR", "ERROR"),
                    ],
                    id="setting-log-level",
                    value="INFO",
                    allow_blank=False,
                )
                yield Static("Daemon log level", classes="hint")

            with Horizontal(classes="settings-btn-row"):
                yield Button(
                    "💾  Save Preferences",
                    id="btn-save-prefs",
                    variant="success",
                )

        # ══════════════════════════════════════════════════════════
        #  Section 5: App Info
        # ══════════════════════════════════════════════════════════
        with Container(classes="settings-section"):
            yield Static(
                "📋  App Information",
                classes="settings-section-title",
                markup=True,
            )
            yield Static("", id="info-config-path")
            yield Static("", id="info-log-path")
            yield Static("", id="info-cache-path")
            yield Static("", id="info-version")

        yield Static("", id="settings-status")

    # ── Lifecycle ──────────────────────────────────────────────────
    def on_mount(self) -> None:
        self._load_settings()

    def _load_settings(self) -> None:
        """Pull current settings from daemon and populate UI."""
        client = self.app.client  # type: ignore[attr-defined]
        try:
            cfg = client.get_config()
        except DaemonDisconnected:
            return

        # Server config
        self.query_one("#setting-host", Input).value = cfg.get("host", "127.0.0.1")
        self.query_one("#setting-port", Input).value = str(cfg.get("port", 8000))

        # Preferences
        self.query_one("#setting-auto-restore", Switch).value = cfg.get(
            "auto_restore", True
        )
        try:
            self.query_one("#setting-log-level", Select).value = cfg.get(
                "log_level", "INFO"
            )
        except Exception:
            pass

        # HF token — populate if saved (masked)
        hf_token = cfg.get("hf_token", "")
        if hf_token:
            self.query_one("#hf-token-input", Input).value = hf_token

        # Model directory
        model_dir = cfg.get("model_dir", "")
        self.query_one("#setting-model-dir", Input).value = model_dir
        default_dir = str(Path.home() / ".cache" / "huggingface")
        effective = model_dir or default_dir
        self.query_one("#model-dir-info", Static).update(
            f"  [dim]Current:[/dim] {effective}"
        )

        # Check HF login status
        self._check_hf_status()

        # App info
        from src.config import CONFIG_DIR, CACHE_DIR, CONFIG_FILE

        self.query_one("#info-config-path", Static).update(
            f"  Config:  {CONFIG_FILE}"
        )
        from src.daemon.process import LOG_FILE

        self.query_one("#info-log-path", Static).update(
            f"  Log:     {LOG_FILE}"
        )
        self.query_one("#info-cache-path", Static).update(
            f"  Cache:   {CACHE_DIR}"
        )
        self.query_one("#info-version", Static).update(
            "  Version: 1.0.0"
        )

    def _check_hf_status(self) -> None:
        """Check HuggingFace login status from daemon."""
        client = self.app.client  # type: ignore[attr-defined]
        status_label = self.query_one("#hf-login-status", Static)
        try:
            hf = client.hf_status()
            if hf.get("logged_in"):
                username = hf.get("username", "?")
                fullname = hf.get("fullname", "")
                display = f"{fullname} ({username})" if fullname else username
                status_label.update(
                    f"[green]✓ Logged in as[/green]  [b]{display}[/b]"
                )
            elif hf.get("error"):
                status_label.update(
                    "[yellow]⚠ Token saved but invalid — please re-login[/yellow]"
                )
            else:
                status_label.update("[dim]Not logged in[/dim]")
        except DaemonDisconnected:
            status_label.update("[red]● Daemon offline[/red]")

    # ── Button handlers ────────────────────────────────────────────
    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id or ""

        if bid == "btn-hf-login":
            self._do_hf_login()
        elif bid == "btn-hf-logout":
            self._do_hf_logout()
        elif bid == "btn-hf-toggle":
            self._toggle_token_visibility()
        elif bid == "btn-save-server":
            self._save_server_config()
        elif bid == "btn-save-model-dir":
            self._save_model_dir()
        elif bid == "btn-reset-model-dir":
            self._reset_model_dir()
        elif bid == "btn-save-prefs":
            self._save_preferences()

    # ── HuggingFace login / logout ─────────────────────────────────
    def _do_hf_login(self) -> None:
        app = self.app  # type: ignore[attr-defined]
        token = self.query_one("#hf-token-input", Input).value.strip()
        status_label = self.query_one("#hf-login-status", Static)

        if not token:
            status_label.update("[red]Please enter a token[/red]")
            app.notify("Please enter a token", severity="warning")
            return

        status_label.update("[yellow]⏳ Logging in…[/yellow]")
        app.notify("HuggingFace login in progress…")
        # Run in worker so TUI doesn't freeze
        self.run_worker(self._login_worker(token), exclusive=True)

    async def _login_worker(self, token: str) -> None:
        app = self.app  # type: ignore[attr-defined]
        status_label = self.query_one("#hf-login-status", Static)
        try:
            result = app.client.set_hf_token(token)
            if result.get("ok"):
                data = result.get("data", {})
                username = data.get("username", "?")
                fullname = data.get("fullname", "")
                display = f"{fullname} ({username})" if fullname else username
                status_label.update(
                    f"[green]✓ Logged in as[/green]  [b]{display}[/b]"
                )
                app.notify(f"HuggingFace login successful: {display} ✓")
            else:
                err = result.get("error", "Unknown error")
                status_label.update(f"[red]✗ {err}[/red]")
                app.notify(f"Login failed: {err}", severity="error")
        except DaemonDisconnected:
            status_label.update("[red]● Daemon offline[/red]")
            app.notify("Daemon offline", severity="error")

    def _do_hf_logout(self) -> None:
        app = self.app  # type: ignore[attr-defined]
        status_label = self.query_one("#hf-login-status", Static)
        try:
            result = app.client.set_hf_token("")  # empty = clear
            if result.get("ok"):
                self.query_one("#hf-token-input", Input).value = ""
                status_label.update("[dim]Logged out[/dim]")
                app.notify("HuggingFace logout ✓")
            else:
                app.notify("Logout failed", severity="error")
        except DaemonDisconnected:
            app.notify("Daemon offline", severity="error")

    def _toggle_token_visibility(self) -> None:
        inp = self.query_one("#hf-token-input", Input)
        inp.password = not inp.password

    # ── Model directory ──────────────────────────────────────────
    def _save_model_dir(self) -> None:
        app = self.app  # type: ignore[attr-defined]
        status_label = self.query_one("#settings-status", Static)
        path = self.query_one("#setting-model-dir", Input).value.strip()

        try:
            result = app.client.set_model_dir(path)
            if result.get("ok"):
                data = result.get("data", {})
                effective = data.get("effective", path)
                self.query_one("#model-dir-info", Static).update(
                    f"  [dim]Current:[/dim] {effective}"
                )
                status_label.update(
                    f"[green]✓ Model directory saved → {effective}[/green]"
                )
                app.notify(f"Model directory: {effective} ✓")
            else:
                err = result.get("error", "Failed")
                status_label.update(f"[red]✗ {err}[/red]")
                app.notify(err, severity="error")
        except DaemonDisconnected:
            app.notify("Daemon offline", severity="error")

    def _reset_model_dir(self) -> None:
        app = self.app  # type: ignore[attr-defined]
        status_label = self.query_one("#settings-status", Static)
        try:
            result = app.client.set_model_dir("")  # empty = default
            if result.get("ok"):
                self.query_one("#setting-model-dir", Input).value = ""
                effective = result.get("data", {}).get("effective", "")
                self.query_one("#model-dir-info", Static).update(
                    f"  [dim]Current:[/dim] {effective}"
                )
                status_label.update(
                    "[yellow]↺ Model directory reset to default[/yellow]"
                )
                app.notify("Model directory reset to default ✓")
        except DaemonDisconnected:
            app.notify("Daemon offline", severity="error")

    # ── Server config ──────────────────────────────────────────────
    def _save_server_config(self) -> None:
        app = self.app  # type: ignore[attr-defined]
        status_label = self.query_one("#settings-status", Static)

        host = self.query_one("#setting-host", Input).value.strip()
        port_str = self.query_one("#setting-port", Input).value.strip()

        if not host:
            status_label.update("[red]Host cannot be empty[/red]")
            return

        try:
            port = int(port_str)
        except ValueError:
            status_label.update("[red]Port must be a number[/red]")
            app.notify("Port must be a number", severity="error")
            return

        if not (1024 <= port <= 65535):
            status_label.update("[red]Port: 1024 – 65535[/red]")
            app.notify("Port must be between 1024 and 65535", severity="error")
            return

        try:
            result = app.client.set_server_port(port, host)
            if result.get("ok"):
                data = result.get("data", {})
                if data.get("restarted"):
                    msg = f"Server restarted → {host}:{port} ✓"
                else:
                    msg = f"Server config saved → {host}:{port} ✓"
                status_label.update(f"[green]✓ {msg}[/green]")
                app.notify(msg)
                app.update_sidebar_status()
            else:
                err = result.get("error", "Failed")
                status_label.update(f"[red]✗ {err}[/red]")
                app.notify(err, severity="error")
        except DaemonDisconnected:
            app.notify("Daemon offline", severity="error")

    # ── Preferences ────────────────────────────────────────────────
    def _save_preferences(self) -> None:
        app = self.app  # type: ignore[attr-defined]
        status_label = self.query_one("#settings-status", Static)

        try:
            # Auto-restore
            auto_restore = self.query_one("#setting-auto-restore", Switch).value
            app.client.set_auto_restore(auto_restore)

            # Log level
            log_level = self.query_one("#setting-log-level", Select).value
            if log_level and log_level != Select.BLANK:
                app.client.set_log_level(str(log_level))

            status_label.update("[green]✓ Preferences saved[/green]")
            app.notify("Preferences saved ✓")
        except DaemonDisconnected:
            app.notify("Daemon offline", severity="error")
        except Exception as exc:
            status_label.update(f"[red]✗ {exc}[/red]")
            app.notify(f"Error: {exc}", severity="error")
