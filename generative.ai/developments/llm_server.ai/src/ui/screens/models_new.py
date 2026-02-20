"""Models screen — search, download, load, delete HF models (daemon client)."""

from __future__ import annotations

import time

from textual import work
from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.widgets import (
    Static,
    Button,
    Input,
    DataTable,
    ProgressBar,
)

from src.daemon.client import DaemonDisconnected


class ModelsScreen(Container):
    """Browse Hugging Face Hub and manage downloaded models."""

    DEFAULT_CSS = """
    ModelsScreen {
        padding: 1 2;
        overflow-y: auto;
    }
    #search-row {
        height: 3;
        margin: 0 0 1 0;
    }
    #search-input {
        width: 1fr;
    }
    #search-btn {
        width: 18;
    }
    #search-results {
        height: auto;
        max-height: 12;
        margin: 0 0 1 0;
    }
    #downloaded-models {
        height: auto;
        max-height: 12;
        margin: 0 0 1 0;
    }
    .model-section {
        background: #24283b;
        border: solid #414868;
        padding: 1 2;
        margin: 0 0 1 0;
        height: auto;
    }
    .model-btn-row {
        height: 3;
        margin: 1 0 0 0;
    }
    #dl-status {
        height: auto;
        margin: 0 0 0 0;
        color: #7aa2f7;
    }

    /* ── Download panel (hidden until download starts) ── */
    #dl-panel {
        display: none;
        background: #1a1b26;
        border: solid #7aa2f7;
        padding: 1 2;
        margin: 0 0 1 0;
        height: auto;
    }
    #dl-header {
        color: #7aa2f7;
        text-style: bold;
        margin: 0 0 1 0;
    }
    #dl-file-table {
        height: auto;
        max-height: 14;
        margin: 0 0 1 0;
    }
    #dl-overall-label {
        margin: 0 0 0 0;
        color: #c0caf5;
    }
    #dl-progress {
        height: 1;
        margin: 0 0 1 0;
    }
    #dl-ctrl-row {
        height: 3;
        margin: 0;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static("🧠  Models", classes="panel-title")

        # ── Search HF Hub ──────────────────────────────────────────
        with Container(classes="model-section"):
            yield Static("🔍  [b]Search Hugging Face Hub[/b]", markup=True)
            with Horizontal(id="search-row"):
                yield Input(
                    placeholder="Enter model name (e.g. gpt2, mistralai/Mistral-7B)…",
                    id="search-input",
                )
                yield Button("Search", id="search-btn", variant="primary")

            yield DataTable(id="search-results")
            yield Static("", id="dl-status")

            with Horizontal(classes="model-btn-row"):
                yield Button(
                    "⬇  Download Selected", id="btn-download", variant="success"
                )

        # ── Download progress panel ────────────────────────────────
        with Container(id="dl-panel"):
            yield Static("", id="dl-header")
            yield DataTable(id="dl-file-table")
            yield Static("", id="dl-overall-label")
            yield ProgressBar(
                id="dl-progress", total=100,
                show_eta=True, show_percentage=True,
            )
            with Horizontal(id="dl-ctrl-row"):
                yield Button("⏹  Stop", id="btn-stop-dl", variant="error")
                yield Button("▶  Resume", id="btn-resume-dl", variant="warning")

        # ── Downloaded models ──────────────────────────────────────
        with Container(classes="model-section"):
            yield Static("📦  [b]Downloaded Models[/b]", markup=True)
            yield DataTable(id="downloaded-models")

            with Horizontal(classes="model-btn-row"):
                yield Button("🔄  Load", id="btn-load", variant="success")
                yield Button("⏏  Unload", id="btn-unload", variant="warning")
                yield Button("🗑  Delete", id="btn-delete", variant="error")
                yield Button("♻  Refresh", id="btn-refresh", variant="default")

    # ── Lifecycle ──────────────────────────────────────────────────
    def on_mount(self) -> None:
        # Search results table
        tbl = self.query_one("#search-results", DataTable)
        tbl.add_columns("Model ID", "Downloads", "Likes", "Pipeline")
        tbl.cursor_type = "row"

        # File-list table inside download panel
        ftbl = self.query_one("#dl-file-table", DataTable)
        ftbl.add_columns("Status", "File", "Size")
        ftbl.cursor_type = "none"
        ftbl.show_cursor = False

        # Downloaded table
        dtbl = self.query_one("#downloaded-models", DataTable)
        dtbl.add_columns("Model ID", "Size", "Files", "Status")
        dtbl.cursor_type = "row"

        # Resume hidden at first
        self.query_one("#btn-resume-dl", Button).display = False

        self._downloading_model_id: str | None = None
        self._polling_download = False
        self._dl_table_populated = False
        self._file_statuses: dict[str, str] = {}
        self._refresh_downloaded()

        # Check if a download is already in progress (daemon may have one running)
        self._check_existing_download()

    # ── Check existing download on mount ───────────────────────────
    def _check_existing_download(self) -> None:
        try:
            state = self.app.client.download_status()  # type: ignore[attr-defined]
            if state.get("active"):
                self._downloading_model_id = state.get("model_id")
                self._show_download_panel(state.get("model_id", ""))
                self._start_polling()
        except DaemonDisconnected:
            pass

    # ── Refresh downloaded list ────────────────────────────────────
    def _refresh_downloaded(self) -> None:
        client = self.app.client  # type: ignore[attr-defined]
        dtbl = self.query_one("#downloaded-models", DataTable)
        dtbl.clear()

        try:
            models = client.list_models()
        except DaemonDisconnected:
            return

        for m in models:
            status = (
                "[green]● Loaded[/green]"
                if m.get("is_loaded")
                else "[dim]available[/dim]"
            )
            dtbl.add_row(
                m["repo_id"],
                m["size_str"],
                str(m["nb_files"]),
                status,
                key=m["repo_id"],
            )

    # ── Button handlers ────────────────────────────────────────────
    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id
        if bid == "search-btn":
            query = self.query_one("#search-input", Input).value.strip()
            if query:
                self._search_models(query)
        elif bid == "btn-download":
            self._download_selected()
        elif bid == "btn-stop-dl":
            self._stop_download()
        elif bid == "btn-resume-dl":
            self._resume_download()
        elif bid == "btn-load":
            self._load_selected()
        elif bid == "btn-unload":
            self._unload_model()
        elif bid == "btn-delete":
            self._delete_selected()
        elif bid == "btn-refresh":
            self._refresh_downloaded()
            self.app.notify("Refreshed ✓")  # type: ignore[attr-defined]

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "search-input" and event.value.strip():
            self._search_models(event.value.strip())

    # ── Background workers ─────────────────────────────────────────
    @work(thread=True, exclusive=True, group="search")
    def _search_models(self, query: str) -> None:
        client = self.app.client  # type: ignore[attr-defined]
        status = self.query_one("#dl-status", Static)
        self.app.call_from_thread(status.update, f"Searching '{query}'...")

        try:
            results = client.search_models(query)
            self.app.call_from_thread(self._populate_search, results)
            self.app.call_from_thread(status.update, f"Found {len(results)} models")
        except DaemonDisconnected:
            self.app.call_from_thread(
                status.update, "[red]Daemon offline[/red]"
            )
        except Exception as exc:
            self.app.call_from_thread(
                status.update, f"[red]Search error: {exc}[/red]"
            )

    def _populate_search(self, results: list) -> None:
        tbl = self.query_one("#search-results", DataTable)
        tbl.clear()
        for m in results:
            tbl.add_row(
                m["id"],
                f"{m['downloads']:,}",
                str(m["likes"]),
                m["pipeline_tag"],
                key=m["id"],
            )

    # ── Download with polling ──────────────────────────────────────
    def _download_selected(self) -> None:
        tbl = self.query_one("#search-results", DataTable)
        if tbl.cursor_row is not None and tbl.row_count > 0:
            row_key, _ = tbl.coordinate_to_cell_key(tbl.cursor_coordinate)
            model_id = str(row_key.value)
            self._start_download(model_id)
        else:
            self.app.notify(  # type: ignore[attr-defined]
                "Select a model from search results first",
                severity="warning",
            )

    def _start_download(self, model_id: str) -> None:
        client = self.app.client  # type: ignore[attr-defined]
        try:
            result = client.download_model(model_id)
            if not result["ok"]:
                self.app.notify(  # type: ignore[attr-defined]
                    result.get("error", "Download failed"), severity="error"
                )
                return
        except DaemonDisconnected:
            self.app.notify("Daemon offline", severity="error")  # type: ignore[attr-defined]
            return

        self._downloading_model_id = model_id
        self._show_download_panel(model_id)
        self._start_polling()

    def _show_download_panel(self, model_id: str) -> None:
        panel = self.query_one("#dl-panel", Container)
        panel.display = True
        self.query_one("#dl-header", Static).update(
            f"⬇  Downloading [b]{model_id}[/b]"
        )
        ftbl = self.query_one("#dl-file-table", DataTable)
        ftbl.clear()
        pbar = self.query_one("#dl-progress", ProgressBar)
        pbar.update(total=100, progress=0)
        self.query_one("#btn-stop-dl", Button).display = True
        self.query_one("#btn-resume-dl", Button).display = False
        self.query_one("#dl-status", Static).update("")
        self._dl_table_populated = False
        self._file_statuses = {}

    def _start_polling(self) -> None:
        if not self._polling_download:
            self._polling_download = True
            self._poll_download_loop()

    @work(thread=True, exclusive=True, group="dl-poll")
    def _poll_download_loop(self) -> None:
        """Poll daemon for download progress every 500ms."""
        while self._polling_download:
            try:
                state = self.app.client.download_status()  # type: ignore[attr-defined]
            except DaemonDisconnected:
                self._polling_download = False
                self.app.call_from_thread(
                    self.app.notify, "Daemon offline", severity="error"
                )
                break

            self.app.call_from_thread(self._update_download_ui, state)

            phase = state.get("phase", "idle")
            if phase in ("completed", "cancelled", "error", "idle"):
                self._polling_download = False
                break

            time.sleep(0.5)

    def _update_download_ui(self, state: dict) -> None:
        """Update download UI elements from polled state."""
        phase = state.get("phase", "idle")
        files = state.get("files", [])
        model_id = state.get("model_id", "")

        ftbl = self.query_one("#dl-file-table", DataTable)
        pbar = self.query_one("#dl-progress", ProgressBar)
        overall_lbl = self.query_one("#dl-overall-label", Static)
        header = self.query_one("#dl-header", Static)

        # Populate file table once
        if files and not self._dl_table_populated:
            ftbl.clear()
            self._file_statuses = {}
            for f in files:
                key = f["name"]
                ftbl.add_row(
                    "[dim]⏳ pending[/dim]",
                    f["name"],
                    f.get("size_str", ""),
                    key=key,
                )
                self._file_statuses[key] = "pending"
            self._dl_table_populated = True

        # Update changed file statuses
        for f in files:
            fname = f["name"]
            fstatus = f.get("status", "pending")
            if self._file_statuses.get(fname) != fstatus:
                self._update_file_row(ftbl, fname, self._status_label(fstatus))
                self._file_statuses[fname] = fstatus

        # Progress bar
        pct = state.get("progress_pct", 0)
        pbar.update(total=100, progress=min(pct, 100))

        # Overall label with speed
        idx = state.get("current_idx", 0)
        total = state.get("total_files", 0)
        speed_str = state.get("speed_str", "—")
        bytes_done = state.get("bytes_done", 0)
        bytes_total = state.get("bytes_total", 0)

        if phase == "downloading":
            done_str = self._human_size(bytes_done)
            total_str = self._human_size(bytes_total)
            overall_lbl.update(
                f"  [{idx}/{total}]  {done_str} / {total_str}  ⚡ {speed_str}"
            )
        elif phase == "preparing":
            overall_lbl.update("  Preparing download…")

        # Handle terminal phases
        if phase == "completed":
            pbar.update(total=100, progress=100)
            header.update(f"[green]✓  Download complete — {model_id}[/green]")
            self.query_one("#btn-stop-dl", Button).display = False
            self._refresh_downloaded()
            self.app.notify(f"Model {model_id} downloaded ✓")  # type: ignore[attr-defined]
            self._downloading_model_id = None

        elif phase == "cancelled":
            header.update(f"[yellow]⏸  Paused:[/yellow] {model_id}")
            self.query_one("#btn-stop-dl", Button).display = False
            self.query_one("#btn-resume-dl", Button).display = True
            self.app.notify(  # type: ignore[attr-defined]
                "Download stopped — press Resume to continue",
                severity="warning",
            )

        elif phase == "error":
            err = state.get("error", "Unknown error")
            header.update(f"[red]✗  Error:[/red] {err}")
            self.query_one("#btn-stop-dl", Button).display = False
            self._downloading_model_id = None

    @staticmethod
    def _status_label(status: str) -> str:
        if status == "downloading":
            return "[cyan]⬇  downloading[/cyan]"
        elif status == "done":
            return "[green]✓  done[/green]"
        elif status == "stopped":
            return "[yellow]⏸ stopped[/yellow]"
        else:
            return "[dim]⏳ pending[/dim]"

    @staticmethod
    def _human_size(nbytes: int) -> str:
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if nbytes < 1024:
                return f"{nbytes:.1f} {unit}"
            nbytes /= 1024  # type: ignore[assignment]
        return f"{nbytes:.1f} PB"

    # ── Helpers: update file row in DataTable ──────────────────────
    def _update_file_row(
        self, ftbl: DataTable, fname: str, status_text: str
    ) -> None:
        try:
            from textual.widgets._data_table import RowKey

            row_key = RowKey(fname)
            col_key = ftbl.columns[list(ftbl.columns.keys())[0]].key
            ftbl.update_cell(row_key, col_key, status_text)
        except Exception:
            pass

    # ── Download actions ───────────────────────────────────────────
    def _stop_download(self) -> None:
        try:
            self.app.client.cancel_download()  # type: ignore[attr-defined]
            self.app.notify("Stopping after current file…")  # type: ignore[attr-defined]
        except DaemonDisconnected:
            self.app.notify("Daemon offline", severity="error")  # type: ignore[attr-defined]

    def _resume_download(self) -> None:
        if self._downloading_model_id:
            mid = self._downloading_model_id
            self.app.notify(f"Resuming {mid}…")  # type: ignore[attr-defined]
            self._start_download(mid)

    # ── Load / Unload / Delete ─────────────────────────────────────
    @work(thread=True, exclusive=True, group="load")
    def _do_load(self, model_id: str) -> None:
        status = self.query_one("#dl-status", Static)
        self.app.call_from_thread(
            status.update, f"Loading [b]{model_id}[/b]…"
        )
        try:
            result = self.app.client.load_model(model_id)  # type: ignore[attr-defined]
            if result["ok"]:
                self.app.call_from_thread(
                    status.update, f"[green]✓ Model {model_id} loaded[/green]"
                )
                self.app.call_from_thread(self._refresh_downloaded)
                self.app.call_from_thread(self.app.update_sidebar_status)
                self.app.call_from_thread(
                    self.app.notify, f"Model loaded: {model_id}"
                )
            else:
                self.app.call_from_thread(
                    status.update,
                    f"[red]Load error: {result.get('error', '?')}[/red]",
                )
        except DaemonDisconnected:
            self.app.call_from_thread(
                status.update, "[red]Daemon offline[/red]"
            )

    def _load_selected(self) -> None:
        dtbl = self.query_one("#downloaded-models", DataTable)
        if dtbl.cursor_row is not None and dtbl.row_count > 0:
            row_key, _ = dtbl.coordinate_to_cell_key(dtbl.cursor_coordinate)
            model_id = str(row_key.value)
            self._do_load(model_id)
        else:
            self.app.notify("Select a downloaded model first", severity="warning")  # type: ignore[attr-defined]

    def _unload_model(self) -> None:
        app = self.app  # type: ignore[attr-defined]
        try:
            result = app.client.unload_model()
            if result["ok"]:
                app.notify("Model unloaded")
                self._refresh_downloaded()
                self.query_one("#dl-status", Static).update(
                    "[yellow]Model unloaded[/yellow]"
                )
                app.update_sidebar_status()
            else:
                app.notify(result.get("error", "Failed"), severity="warning")
        except DaemonDisconnected:
            app.notify("Daemon offline", severity="error")

    def _delete_selected(self) -> None:
        dtbl = self.query_one("#downloaded-models", DataTable)
        if dtbl.cursor_row is not None and dtbl.row_count > 0:
            row_key, _ = dtbl.coordinate_to_cell_key(dtbl.cursor_coordinate)
            model_id = str(row_key.value)
            app = self.app  # type: ignore[attr-defined]
            try:
                result = app.client.delete_model(model_id)
                if result.get("ok"):
                    app.notify(f"Deleted {model_id}")
                else:
                    app.notify(
                        result.get("error", f"Could not delete {model_id}"),
                        severity="error",
                    )
            except DaemonDisconnected:
                app.notify("Daemon offline", severity="error")
            self._refresh_downloaded()
            app.update_sidebar_status()
        else:
            self.app.notify("Select a model first", severity="warning")  # type: ignore[attr-defined]
