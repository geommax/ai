"""Models screen — search, download, load, delete HF models (daemon client)."""

from __future__ import annotations

import time

from textual import work
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import (
    Static,
    Button,
    Input,
    DataTable,
    ProgressBar,
    Select,
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
        align: left middle;
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
        align: left middle;
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

    /* ── Backend selector ── */
    #backend-row {
        height: 3;
        margin: 0 0 0 0;
        align: left middle;
    }
    #backend-select {
        width: 28;
    }
    #backend-label {
        width: auto;
        margin: 0 1 0 0;
        color: #c0caf5;
    }
    #btn-switch-bk {
        width: 26;
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

        # ── Downloaded models ──────────────────────────────────────
        with Container(classes="model-section"):
            yield Static("📦  [b]Downloaded Models[/b]", markup=True)
            yield DataTable(id="downloaded-models")

            with Horizontal(id="backend-row"):
                yield Static("Engine:", id="backend-label")
                yield Select(
                    [("🔄 Auto", "auto"),
                     ("🔧 Transformers", "transformers"),
                     ("🦙 llama.cpp", "llama.cpp")],
                    value="auto",
                    id="backend-select",
                    allow_blank=False,
                )
                yield Button(
                    "⇄  Switch Backend", id="btn-switch-bk", variant="default"
                )

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
        dtbl.add_columns("Model ID", "Format", "Size", "Files", "Status")
        dtbl.cursor_type = "row"

        self._downloading_model_id: str | None = None
        self._file_keys: list[str] = []
        self._refresh_downloaded()
        self._check_existing_download()

    # ── Refresh downloaded list (daemon client) ────────────────────
    def _refresh_downloaded(self) -> None:
        client = self.app.client  # type: ignore[attr-defined]
        dtbl = self.query_one("#downloaded-models", DataTable)
        dtbl.clear()
        try:
            models = client.list_models()
        except DaemonDisconnected:
            return
        for m in models:
            fmt = m.get("format", "?")
            fmt_label = {
                "gguf": "[bold cyan]GGUF[/bold cyan]",
                "safetensors": "[bold green]SafeT[/bold green]",
                "pytorch": "[yellow]PyTorch[/yellow]",
            }.get(fmt, fmt)
            if m.get("is_loaded"):
                backend = m.get("backend", "")
                bk = f" ({backend})" if backend else ""
                status = f"[green]● Loaded{bk}[/green]"
            else:
                status = "[dim]available[/dim]"
            dtbl.add_row(
                m["repo_id"],
                fmt_label,
                m.get("size_str", "?"),
                str(m.get("nb_files", "?")),
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
        elif bid == "btn-load":
            self._load_selected()
        elif bid == "btn-switch-bk":
            self._switch_backend()
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

    # ── Search (daemon client) ─────────────────────────────────────
    @work(thread=True, exclusive=True, group="search")
    def _search_models(self, query: str) -> None:
        client = self.app.client  # type: ignore[attr-defined]
        status = self.query_one("#dl-status", Static)
        self.app.call_from_thread(status.update, f"Searching '{query}'…")
        try:
            results = client.search_models(query)
            self.app.call_from_thread(self._populate_search, results)
            self.app.call_from_thread(status.update, f"Found {len(results)} models")
        except DaemonDisconnected:
            self.app.call_from_thread(status.update, "[red]Daemon offline[/red]")
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

    # ── Download (daemon client + polling) ─────────────────────────
    def _download_selected(self) -> None:
        tbl = self.query_one("#search-results", DataTable)
        if tbl.cursor_row is not None and tbl.row_count > 0:
            row_key, _ = tbl.coordinate_to_cell_key(tbl.cursor_coordinate)
            model_id = str(row_key.value)
            self._maybe_pick_files(model_id)
        else:
            self.app.notify(
                "Select a model from search results first",
                severity="warning",
            )

    @work(thread=True, exclusive=True, group="file-pick")
    def _maybe_pick_files(self, model_id: str) -> None:
        """If the repo has multiple GGUF files, show a picker dialog."""
        client = self.app.client  # type: ignore[attr-defined]
        status = self.query_one("#dl-status", Static)
        self.app.call_from_thread(
            status.update, f"Fetching file list for [b]{model_id}[/b]…"
        )
        try:
            files = client.list_repo_files(model_id)
        except DaemonDisconnected:
            self.app.call_from_thread(
                status.update, "[red]Daemon offline[/red]"
            )
            return
        except Exception as exc:
            self.app.call_from_thread(
                status.update, f"[red]Error: {exc}[/red]"
            )
            return

        gguf_files = [f for f in files if f["name"].endswith(".gguf")]

        if len(gguf_files) > 1:
            # Multiple GGUF variants → show picker
            self.app.call_from_thread(status.update, "")
            self.app.call_from_thread(
                self._show_file_picker, model_id, gguf_files
            )
        else:
            # Single GGUF or non-GGUF repo → download everything
            self.app.call_from_thread(status.update, "")
            self.app.call_from_thread(self._start_download, model_id)

    def _show_file_picker(
        self, model_id: str, files: list[dict]
    ) -> None:
        from src.ui.dialogs import FilePickerDialog

        def on_result(result: list[str] | str | None) -> None:
            if result == "__cancel__":
                return  # user cancelled
            if result is None:
                # "Download All" — download everything
                self._start_download(model_id)
            else:
                # Download only the selected files
                self._start_download(model_id, filenames=result)

        self.app.push_screen(
            FilePickerDialog(model_id, files, title="📂  Select GGUF variant"),
            on_result,
        )

    def _start_download(
        self,
        model_id: str,
        filenames: list[str] | None = None,
    ) -> None:
        """Tell daemon to start downloading, then begin polling."""
        client = self.app.client  # type: ignore[attr-defined]
        try:
            result = client.download_model(model_id, filenames=filenames)
            if not result.get("ok"):
                self.app.notify(
                    result.get("error", "Download failed"), severity="error"
                )
                return
        except DaemonDisconnected:
            self.app.notify("Daemon offline", severity="error")
            return

        self._downloading_model_id = model_id
        # Show download panel
        panel = self.query_one("#dl-panel", Container)
        panel.display = True
        self.query_one("#dl-header", Static).update(
            f"⬇  Downloading [b]{model_id}[/b]"
        )
        ftbl = self.query_one("#dl-file-table", DataTable)
        ftbl.clear()
        self._file_keys = []
        self.query_one("#dl-progress", ProgressBar).update(total=100, progress=0)
        self.query_one("#btn-stop-dl", Button).display = True
        self.query_one("#dl-status", Static).update("")
        # Start polling loop
        self._poll_download_loop()

    @work(thread=True, exclusive=True, group="download-poll")
    def _poll_download_loop(self) -> None:
        """Poll daemon for download progress every 300ms."""
        client = self.app.client  # type: ignore[attr-defined]
        prev_file_count = 0

        while True:
            time.sleep(0.3)
            try:
                st = client.download_status()
            except DaemonDisconnected:
                self.app.call_from_thread(
                    self.query_one("#dl-header", Static).update,
                    "[red]Daemon disconnected[/red]",
                )
                break

            phase = st.get("phase", "idle")
            files = st.get("files", [])
            pct = st.get("progress_pct", 0)
            speed_str = st.get("speed_str", "—")
            idx = st.get("current_idx", 0)
            total = st.get("total_files", 0)
            bytes_done = st.get("bytes_done", 0)
            bytes_total = st.get("bytes_total", 0)

            # Rebuild file table when file count changes
            if len(files) != prev_file_count:
                self._rebuild_file_table(files)
                prev_file_count = len(files)
            else:
                self._update_file_statuses(files)

            # Progress bar
            self.app.call_from_thread(
                self.query_one("#dl-progress", ProgressBar).update,
                total=100,
                progress=min(pct, 100),
            )

            # Overall label
            size_done = self._human_size(bytes_done)
            size_total = self._human_size(bytes_total)
            self.app.call_from_thread(
                self.query_one("#dl-overall-label", Static).update,
                f"  [{idx}/{total}]  {size_done} / {size_total}  ⚡ {speed_str}",
            )

            # Terminal states
            if phase == "completed":
                mid = self._downloading_model_id
                self.app.call_from_thread(self._refresh_downloaded)
                self.app.call_from_thread(self._clean_dl_panel)
                self.app.call_from_thread(
                    self.app.notify,
                    f"Model {mid} downloaded ✓",
                )
                self._downloading_model_id = None
                break

            elif phase == "cancelled":
                self.app.call_from_thread(self._clean_dl_panel)
                self.app.call_from_thread(
                    self.app.notify,
                    "Download stopped ✓",
                    severity="warning",
                )
                self._downloading_model_id = None
                break

            elif phase == "error":
                err = st.get("error", "Unknown error")
                self.app.call_from_thread(self._clean_dl_panel)
                self.app.call_from_thread(
                    self.app.notify,
                    f"Download error: {err}",
                    severity="error",
                )
                self._downloading_model_id = None
                break

            elif phase == "idle" and not st.get("active"):
                break

    def _rebuild_file_table(self, files: list) -> None:
        """Rebuild the download file table from scratch."""
        ftbl = self.query_one("#dl-file-table", DataTable)
        self.app.call_from_thread(ftbl.clear)
        self._file_keys = []
        status_map = {
            "pending": "[dim]⏳ pending[/dim]",
            "downloading": "[cyan]⬇  downloading[/cyan]",
            "done": "[green]✓  done[/green]",
            "stopped": "[yellow]⏸ stopped[/yellow]",
        }
        for f in files:
            key = f["name"]
            self._file_keys.append(key)
            icon = status_map.get(f.get("status", "pending"), "[dim]⏳ pending[/dim]")
            self.app.call_from_thread(
                ftbl.add_row, icon, f["name"], f.get("size_str", "?"), key=key
            )

    def _update_file_statuses(self, files: list) -> None:
        """Update per-file status icons in the download table."""
        ftbl = self.query_one("#dl-file-table", DataTable)
        status_map = {
            "pending": "[dim]⏳ pending[/dim]",
            "downloading": "[cyan]⬇  downloading[/cyan]",
            "done": "[green]✓  done[/green]",
            "stopped": "[yellow]⏸ stopped[/yellow]",
        }
        try:
            from textual.widgets._data_table import RowKey

            col_key = ftbl.columns[list(ftbl.columns.keys())[0]].key
            for f in files:
                icon = status_map.get(f.get("status", "pending"), "[dim]?[/dim]")
                try:
                    self.app.call_from_thread(
                        ftbl.update_cell, RowKey(f["name"]), col_key, icon
                    )
                except Exception:
                    pass
        except Exception:
            pass

    def _stop_download(self) -> None:
        try:
            self.app.client.cancel_download()  # type: ignore[attr-defined]
        except DaemonDisconnected:
            self.app.notify("Daemon offline", severity="error")  # type: ignore[attr-defined]
            return
        # Immediately clean the panel for snappy UX;
        # the poll loop will also break on "cancelled" phase.
        self._clean_dl_panel()
        self._downloading_model_id = None
        self.app.notify("Download stopped ✓", severity="warning")  # type: ignore[attr-defined]

    def _clean_dl_panel(self) -> None:
        """Hide the download panel and reset all its widgets."""
        panel = self.query_one("#dl-panel", Container)
        panel.display = False
        self.query_one("#dl-header", Static).update("")
        self.query_one("#dl-overall-label", Static).update("")
        self.query_one("#dl-file-table", DataTable).clear()
        self.query_one("#dl-progress", ProgressBar).update(total=100, progress=0)
        self.query_one("#btn-stop-dl", Button).display = False
        self._file_keys = []

    def _check_existing_download(self) -> None:
        """If daemon has an active download, show the panel and start polling."""
        try:
            st = self.app.client.download_status()  # type: ignore[attr-defined]
            if st.get("active"):
                model_id = st.get("model_id", "?")
                self._downloading_model_id = model_id
                panel = self.query_one("#dl-panel", Container)
                panel.display = True
                self.query_one("#dl-header", Static).update(
                    f"⬇  Downloading [b]{model_id}[/b]"
                )
                self.query_one("#btn-stop-dl", Button).display = True
                self._poll_download_loop()
        except DaemonDisconnected:
            pass

    # ── Load / Unload / Delete (daemon client) ─────────────────────
    def _load_selected(self) -> None:
        dtbl = self.query_one("#downloaded-models", DataTable)
        if dtbl.cursor_row is not None and dtbl.row_count > 0:
            row_key, _ = dtbl.coordinate_to_cell_key(dtbl.cursor_coordinate)
            model_id = str(row_key.value)
            backend = self.query_one("#backend-select", Select).value
            self._do_load(model_id, backend=str(backend))
        else:
            self.app.notify(  # type: ignore[attr-defined]
                "Select a downloaded model first", severity="warning"
            )

    @work(thread=True, exclusive=True, group="load")
    def _do_load(self, model_id: str, backend: str = "auto") -> None:
        status = self.query_one("#dl-status", Static)
        bk_label = f" [{backend}]" if backend != "auto" else ""
        self.app.call_from_thread(
            status.update, f"Loading [b]{model_id}[/b]{bk_label}…"
        )
        try:
            result = self.app.client.load_model(  # type: ignore[attr-defined]
                model_id, backend=backend
            )
            if result.get("ok"):
                actual = result.get("data", {}).get("backend", backend)
                self.app.call_from_thread(
                    status.update,
                    f"[green]✓ {model_id} loaded ({actual})[/green]",
                )
                self.app.call_from_thread(self._refresh_downloaded)
                self.app.call_from_thread(self.app.update_sidebar_status)
                self.app.call_from_thread(
                    self.app.notify, f"Model loaded: {model_id} ({actual})"
                )
            else:
                err = result.get("error", "Failed")
                self.app.call_from_thread(
                    status.update, f"[red]Load error: {err}[/red]"
                )
        except DaemonDisconnected:
            self.app.call_from_thread(status.update, "[red]Daemon offline[/red]")

    def _switch_backend(self) -> None:
        """Switch the currently loaded model to the selected backend."""
        backend = str(self.query_one("#backend-select", Select).value)
        if backend == "auto":
            self.app.notify(  # type: ignore[attr-defined]
                "Select a specific backend (Transformers or llama.cpp)",
                severity="warning",
            )
            return
        self._do_switch(backend)

    @work(thread=True, exclusive=True, group="load")
    def _do_switch(self, backend: str) -> None:
        status = self.query_one("#dl-status", Static)
        self.app.call_from_thread(
            status.update, f"Switching to [b]{backend}[/b]…"
        )
        try:
            result = self.app.client.switch_backend(backend)  # type: ignore[attr-defined]
            if result.get("ok"):
                data = result.get("data", {})
                mid = data.get("model_id", "?")
                bk = data.get("backend", backend)
                self.app.call_from_thread(
                    status.update,
                    f"[green]✓ {mid} reloaded ({bk})[/green]",
                )
                self.app.call_from_thread(self._refresh_downloaded)
                self.app.call_from_thread(self.app.update_sidebar_status)
                self.app.call_from_thread(
                    self.app.notify,
                    f"Backend switched: {mid} → {bk}",
                )
            else:
                err = result.get("error", "Switch failed")
                self.app.call_from_thread(
                    status.update, f"[red]Switch error: {err}[/red]"
                )
        except DaemonDisconnected:
            self.app.call_from_thread(status.update, "[red]Daemon offline[/red]")

    def _unload_model(self) -> None:
        app = self.app  # type: ignore[attr-defined]
        try:
            result = app.client.unload_model()
            if result.get("ok"):
                app.notify("Model unloaded")
                self.query_one("#dl-status", Static).update(
                    "[yellow]Model unloaded[/yellow]"
                )
            else:
                app.notify(result.get("error", "Failed"), severity="warning")
        except DaemonDisconnected:
            app.notify("Daemon offline", severity="error")
        self._refresh_downloaded()
        app.update_sidebar_status()

    def _delete_selected(self) -> None:
        dtbl = self.query_one("#downloaded-models", DataTable)
        if dtbl.cursor_row is not None and dtbl.row_count > 0:
            row_key, _ = dtbl.coordinate_to_cell_key(dtbl.cursor_coordinate)
            model_id = str(row_key.value)
            app = self.app  # type: ignore[attr-defined]

            from src.ui.dialogs import ConfirmDialog

            def on_confirm(confirmed: bool) -> None:
                if confirmed:
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

            app.push_screen(
                ConfirmDialog(
                    f"Permanently delete all files for\n"
                    f"[b]{model_id}[/b]?\n\n"
                    f"This cannot be undone.",
                    title="🗑  Delete Model",
                ),
                on_confirm,
            )
        else:
            self.app.notify(  # type: ignore[attr-defined]
                "Select a model first", severity="warning"
            )

    # ── Helpers ────────────────────────────────────────────────────
    @staticmethod
    def _human_size(nbytes: int) -> str:
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if nbytes < 1024:
                return f"{nbytes:.1f} {unit}"
            nbytes /= 1024  # type: ignore[assignment]
        return f"{nbytes:.1f} PB"
