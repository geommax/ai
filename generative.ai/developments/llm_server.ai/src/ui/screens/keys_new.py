"""API Keys screen — generate, list, revoke keys (daemon client)."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.widgets import Static, Button, Input, DataTable

from src.daemon.client import DaemonDisconnected


class APIKeysScreen(Container):
    """Manage API keys used to authenticate against the local server."""

    DEFAULT_CSS = """
    APIKeysScreen {
        padding: 1 2;
        overflow-y: auto;
    }
    .key-section {
        background: #24283b;
        border: solid #414868;
        padding: 1 2;
        margin: 0 0 1 0;
        height: auto;
    }
    #key-name-input {
        width: 1fr;
    }
    #gen-key-row {
        height: 3;
        margin: 0 0 1 0;
    }
    #last-key-display {
        color: #9ece6a;
        margin: 0 0 1 0;
        height: auto;
    }
    #keys-table {
        height: auto;
        max-height: 18;
        margin: 0 0 1 0;
    }
    .key-btn-row {
        height: 3;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static("🔑  API Keys", classes="panel-title")

        # ── Generate new key ───────────────────────────────────────
        with Container(classes="key-section"):
            yield Static("➕  [b]Generate New API Key[/b]", markup=True)
            with Horizontal(id="gen-key-row"):
                yield Input(
                    placeholder="Key name (e.g. dev, production)…",
                    id="key-name-input",
                )
                yield Button("Generate", id="btn-gen-key", variant="success")
            yield Static("", id="last-key-display")

        # ── Existing keys ──────────────────────────────────────────
        with Container(classes="key-section"):
            yield Static("📋  [b]Existing Keys[/b]", markup=True)
            yield DataTable(id="keys-table")
            with Horizontal(classes="key-btn-row"):
                yield Button("✓ Activate", id="btn-activate-key", variant="success")
                yield Button("✗ Revoke", id="btn-revoke-key", variant="warning")
                yield Button("🗑 Delete", id="btn-delete-key", variant="error")
                yield Button("♻ Refresh", id="btn-refresh-keys", variant="default")

    # ── Lifecycle ──────────────────────────────────────────────────
    def on_mount(self) -> None:
        tbl = self.query_one("#keys-table", DataTable)
        tbl.add_columns("ID", "Name", "Key (prefix)", "Created", "Status")
        tbl.cursor_type = "row"
        self._refresh_keys()

    # ── Refresh table ──────────────────────────────────────────────
    def _refresh_keys(self) -> None:
        client = self.app.client  # type: ignore[attr-defined]
        tbl = self.query_one("#keys-table", DataTable)
        tbl.clear()
        try:
            keys = client.list_keys()
        except DaemonDisconnected:
            return
        for k in keys:
            status = (
                "[green]active[/green]" if k["is_active"] else "[red]revoked[/red]"
            )
            tbl.add_row(
                str(k["id"]),
                k["name"],
                k["key"][:20] + "…",
                k["created_at"][:19],
                status,
                key=str(k["id"]),
            )

    # ── Handlers ───────────────────────────────────────────────────
    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id
        if bid == "btn-gen-key":
            self._generate_key()
        elif bid == "btn-activate-key":
            self._set_key_status(activate=True)
        elif bid == "btn-revoke-key":
            self._set_key_status(activate=False)
        elif bid == "btn-delete-key":
            self._delete_key()
        elif bid == "btn-refresh-keys":
            self._refresh_keys()
            self.app.notify("Refreshed ✓")  # type: ignore[attr-defined]

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "key-name-input":
            self._generate_key()

    def _generate_key(self) -> None:
        app = self.app  # type: ignore[attr-defined]
        name_input = self.query_one("#key-name-input", Input)
        name = name_input.value.strip() or "unnamed"

        try:
            result = app.client.add_key(name)
            if result["ok"]:
                key = result["data"]["key"]
                name_input.value = ""
                self.query_one("#last-key-display", Static).update(
                    f"  🔐 New key: [b]{key}[/b]\n  ⚠  Copy it now — it won't be shown in full again."
                )
                self._refresh_keys()
                app.notify(f"API key '{name}' generated ✓")
            else:
                app.notify(result.get("error", "Failed"), severity="error")
        except DaemonDisconnected:
            app.notify("Daemon offline", severity="error")

    def _set_key_status(self, activate: bool) -> None:
        tbl = self.query_one("#keys-table", DataTable)
        app = self.app  # type: ignore[attr-defined]
        if tbl.cursor_row is not None and tbl.row_count > 0:
            row_key, _ = tbl.coordinate_to_cell_key(tbl.cursor_coordinate)
            kid = int(row_key.value)
            try:
                if activate:
                    app.client.activate_key(kid)
                    app.notify("Key activated")
                else:
                    app.client.revoke_key(kid)
                    app.notify("Key revoked")
            except DaemonDisconnected:
                app.notify("Daemon offline", severity="error")
            self._refresh_keys()
        else:
            app.notify("Select a key first", severity="warning")

    def _delete_key(self) -> None:
        tbl = self.query_one("#keys-table", DataTable)
        app = self.app  # type: ignore[attr-defined]
        if tbl.cursor_row is not None and tbl.row_count > 0:
            row_key, _ = tbl.coordinate_to_cell_key(tbl.cursor_coordinate)
            kid = int(row_key.value)
            try:
                app.client.delete_key(kid)
                app.notify("Key deleted")
            except DaemonDisconnected:
                app.notify("Daemon offline", severity="error")
            self._refresh_keys()
        else:
            app.notify("Select a key first", severity="warning")
