"""API Keys screen — generate, list, revoke keys with usage tracking."""

from __future__ import annotations

import subprocess

from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.widgets import Static, Button, Input, DataTable, Select

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
        align: left middle;
    }
    #last-key-display {
        color: #9ece6a;
        margin: 0 0 0 0;
        height: auto;
    }
    #copy-key-row {
        height: 3;
        margin: 0 0 1 0;
        align: left middle;
    }
    #btn-copy-key {
        display: none;
    }
    #keys-table {
        height: auto;
        max-height: 18;
        margin: 0 0 1 0;
    }
    .key-btn-row {
        height: 3;
        align: left middle;
    }
    /* ── Usage section ── */
    #usage-table {
        height: auto;
        max-height: 14;
        margin: 0 0 1 0;
    }
    #usage-summary {
        height: auto;
        margin: 0 0 1 0;
        padding: 1;
        color: #c0caf5;
    }
    /* ── Filter row ── */
    #filter-row {
        height: 3;
        align: left middle;
        margin: 0 0 1 0;
    }
    #filter-label {
        width: auto;
        padding: 1 1 0 0;
    }
    #key-filter {
        width: 30;
    }
    /* ── Call history ── */
    #history-section {
        display: none;
    }
    #history-table {
        height: auto;
        max-height: 14;
        margin: 0 0 1 0;
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
            with Horizontal(id="copy-key-row"):
                yield Button("📋  Copy Key", id="btn-copy-key", variant="primary")

        # ── Existing keys ──────────────────────────────────────────
        with Container(classes="key-section"):
            yield Static("📋  [b]Existing Keys[/b]", markup=True)
            with Horizontal(id="filter-row"):
                yield Static("Filter: ", id="filter-label")
                yield Select(
                    [
                        ("All Keys", "all"),
                        ("Active", "active"),
                        ("Revoked", "revoked"),
                        ("Expired", "expired"),
                        ("Deleted", "deleted"),
                        ("Every Key", "every"),
                    ],
                    value="all",
                    id="key-filter",
                    allow_blank=False,
                )
            yield DataTable(id="keys-table")
            with Horizontal(classes="key-btn-row"):
                yield Button("✓ Activate", id="btn-activate-key", variant="success")
                yield Button("✗ Revoke", id="btn-revoke-key", variant="warning")
                yield Button("⏳ Expire", id="btn-expire-key", variant="warning")
                yield Button("🗑 Delete", id="btn-delete-key", variant="error")
                yield Button("📜 History", id="btn-history-key", variant="primary")
                yield Button("♻ Refresh", id="btn-refresh-keys", variant="default")

        # ── Call history ───────────────────────────────────────────
        with Container(classes="key-section", id="history-section"):
            yield Static("📜  [b]Call History[/b]", id="history-title", markup=True)
            yield DataTable(id="history-table")

        # ── Usage stats ────────────────────────────────────────────
        with Container(classes="key-section"):
            yield Static("📊  [b]Usage Statistics[/b]", markup=True)
            yield Static("", id="usage-summary")
            yield DataTable(id="usage-table")

    # ── Lifecycle ──────────────────────────────────────────────────
    def on_mount(self) -> None:
        tbl = self.query_one("#keys-table", DataTable)
        tbl.add_columns("Name", "Key (prefix)", "Created", "Status")
        tbl.cursor_type = "row"

        utbl = self.query_one("#usage-table", DataTable)
        utbl.add_columns("Key Name", "Calls", "Prompt Tok", "Compl Tok", "Total Tok", "Last Used")
        utbl.cursor_type = "row"

        htbl = self.query_one("#history-table", DataTable)
        htbl.add_columns("Date & Time", "Endpoint", "Prompt Tok", "Compl Tok", "Total Tok")
        htbl.cursor_type = "row"

        self._last_key: str | None = None
        self._refresh_keys()
        self._refresh_usage()

    # ── Refresh table ──────────────────────────────────────────────
    def _refresh_keys(self) -> None:
        client = self.app.client  # type: ignore[attr-defined]
        tbl = self.query_one("#keys-table", DataTable)
        tbl.clear()
        try:
            filt = self.query_one("#key-filter", Select).value
            status_filter = str(filt) if filt is not Select.BLANK else "all"
            keys = client.list_keys(status_filter=status_filter)
        except DaemonDisconnected:
            return
        for k in keys:
            st = k.get("status", "active")
            color_map = {
                "active": "green",
                "revoked": "yellow",
                "expired": "magenta",
                "deleted": "red",
            }
            color = color_map.get(st, "white")
            label = st
            if st == "deleted":
                d = k.get("deleted_at", "")
                if d:
                    label += f" {d[:10]}"
            status = f"[{color}]{label}[/{color}]"
            tbl.add_row(
                k["name"],
                k["key"][:20] + "…",
                k["created_at"][:19],
                status,
                key=str(k["id"]),
            )

    def _refresh_usage(self) -> None:
        """Fetch per-key usage stats from daemon and populate table."""
        client = self.app.client  # type: ignore[attr-defined]
        utbl = self.query_one("#usage-table", DataTable)
        utbl.clear()
        try:
            usage = client.key_usage()
        except DaemonDisconnected:
            return

        total_calls = 0
        total_tokens = 0

        for u in usage:
            calls = u.get("total_calls", 0)
            p_tok = u.get("prompt_tokens", 0)
            c_tok = u.get("completion_tokens", 0)
            t_tok = u.get("total_tokens", 0)
            last = u.get("last_used") or "—"
            if last and last != "—":
                last = last[:19]

            total_calls += calls
            total_tokens += t_tok

            utbl.add_row(
                u.get("name", "?"),
                f"{calls:,}",
                f"{p_tok:,}",
                f"{c_tok:,}",
                f"{t_tok:,}",
                last,
                key=str(u["id"]),
            )

        summary = self.query_one("#usage-summary", Static)
        summary.update(
            f"  Total API Calls: [b]{total_calls:,}[/b]  │  "
            f"Total Tokens: [b]{total_tokens:,}[/b]"
        )

    # ── Handlers ───────────────────────────────────────────────────
    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id
        if bid == "btn-gen-key":
            self._generate_key()
        elif bid == "btn-copy-key":
            self._copy_last_key()
        elif bid == "btn-activate-key":
            self._change_status("active")
        elif bid == "btn-revoke-key":
            self._change_status("revoked")
        elif bid == "btn-expire-key":
            self._change_status("expired")
        elif bid == "btn-delete-key":
            self._delete_key()
        elif bid == "btn-history-key":
            self._show_history()
        elif bid == "btn-refresh-keys":
            self._refresh_keys()
            self._refresh_usage()
            self.app.notify("Refreshed ✓")  # type: ignore[attr-defined]

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "key-name-input":
            self._generate_key()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "key-filter":
            self._refresh_keys()

    @staticmethod
    def _sys_clipboard_copy(text: str) -> bool:
        """Copy *text* to the system clipboard via xclip or xsel."""
        for cmd in (["xclip", "-selection", "clipboard"], ["xsel", "--clipboard", "--input"]):
            try:
                p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
                p.communicate(text.encode())
                if p.returncode == 0:
                    return True
            except FileNotFoundError:
                continue
        return False

    def _copy_last_key(self) -> None:
        """Copy the last generated key to the system clipboard."""
        app = self.app  # type: ignore[attr-defined]
        if not self._last_key:
            app.notify("No key to copy", severity="warning")
            return
        if self._sys_clipboard_copy(self._last_key):
            app.notify("API key copied to clipboard ✓")
        else:
            app.notify("Clipboard not available — install xclip", severity="warning")

    def _generate_key(self) -> None:
        app = self.app  # type: ignore[attr-defined]
        name_input = self.query_one("#key-name-input", Input)
        name = name_input.value.strip() or "unnamed"

        try:
            result = app.client.add_key(name)
            if result.get("ok"):
                data = result.get("data", {})
                key = data.get("key", "?")
                name_input.value = ""
                self._last_key = key
                self.query_one("#last-key-display", Static).update(
                    f"  🔐 New key: [b]{key}[/b]\n  ⚠  Copy it now — it won't be shown in full again."
                )
                self.query_one("#btn-copy-key", Button).display = True
                # Auto-copy to clipboard
                if self._sys_clipboard_copy(key):
                    app.notify(f"API key '{name}' generated & copied to clipboard ✓")
                else:
                    app.notify(f"API key '{name}' generated ✓ — press Copy to clipboard")
                self._refresh_keys()
                self._refresh_usage()
            else:
                app.notify(result.get("error", "Failed"), severity="error")
        except DaemonDisconnected:
            app.notify("Daemon offline", severity="error")

    def _change_status(self, new_status: str) -> None:
        """Change the selected key's status to *new_status*."""
        tbl = self.query_one("#keys-table", DataTable)
        app = self.app  # type: ignore[attr-defined]
        if tbl.cursor_row is not None and tbl.row_count > 0:
            row_key, _ = tbl.coordinate_to_cell_key(tbl.cursor_coordinate)
            kid = int(row_key.value)
            try:
                app.client.set_key_status(kid, new_status)
                app.notify(f"Key → {new_status} ✓")
                self._refresh_keys()
                self._refresh_usage()
            except DaemonDisconnected:
                app.notify("Daemon offline", severity="error")
        else:
            app.notify("Select a key first", severity="warning")

    def _delete_key(self) -> None:
        tbl = self.query_one("#keys-table", DataTable)
        app = self.app  # type: ignore[attr-defined]
        if tbl.cursor_row is not None and tbl.row_count > 0:
            row_key, _ = tbl.coordinate_to_cell_key(tbl.cursor_coordinate)
            kid = int(row_key.value)

            from src.ui.dialogs import ConfirmDialog

            def on_confirm(confirmed: bool) -> None:
                if confirmed:
                    try:
                        app.client.set_key_status(kid, "deleted")
                        app.notify("Key → deleted ✓")
                        self._refresh_keys()
                        self._refresh_usage()
                    except DaemonDisconnected:
                        app.notify("Daemon offline", severity="error")

            app.push_screen(
                ConfirmDialog(
                    "This API key will be marked as deleted.\n\n"
                    "Usage history will be preserved.\n"
                    "You can view it later via the\n"
                    "Deleted filter.",
                    title="🗑  Delete API Key",
                ),
                on_confirm,
            )
        else:
            app.notify("Select a key first", severity="warning")

    def _show_history(self) -> None:
        """Show per-call usage history for the selected key."""
        tbl = self.query_one("#keys-table", DataTable)
        app = self.app  # type: ignore[attr-defined]
        if tbl.cursor_row is None or tbl.row_count == 0:
            app.notify("Select a key first", severity="warning")
            return
        row_key, _ = tbl.coordinate_to_cell_key(tbl.cursor_coordinate)
        kid = int(row_key.value)
        row_data = tbl.get_row(row_key)
        key_name = row_data[0] if row_data else "?"

        try:
            history = app.client.key_usage_history(kid)
        except DaemonDisconnected:
            app.notify("Daemon offline", severity="error")
            return

        section = self.query_one("#history-section")
        section.display = True
        title = self.query_one("#history-title", Static)
        title.update(f"📜  [b]Call History — {key_name}[/b]")

        htbl = self.query_one("#history-table", DataTable)
        htbl.clear()

        if not history:
            app.notify(f"No call history for '{key_name}'")
            return

        for h in history:
            htbl.add_row(
                h.get("created_at", "?")[:19],
                h.get("endpoint", "?"),
                f"{h.get('prompt_tokens', 0):,}",
                f"{h.get('completion_tokens', 0):,}",
                f"{h.get('total_tokens', 0):,}",
            )
