"""Models screen — search, download, load, delete HF models."""

from __future__ import annotations

from textual import work
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import (
    Static,
    Button,
    Input,
    DataTable,
    ProgressBar,
)


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
        self._file_keys: list[str] = []
        self._refresh_downloaded()

    # ── Refresh downloaded list ────────────────────────────────────
    def _refresh_downloaded(self) -> None:
        app = self.app  # type: ignore[attr-defined]
        dtbl = self.query_one("#downloaded-models", DataTable)
        dtbl.clear()

        for m in app.model_manager.list_downloaded_models():
            status = (
                "[green]● Loaded[/green]"
                if app.inference_engine.model_id == m["repo_id"]
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
        app = self.app  # type: ignore[attr-defined]
        status = self.query_one("#dl-status", Static)
        self.app.call_from_thread(status.update, f"Searching '{query}'...")

        try:
            results = app.model_manager.search_models(query)
            self.app.call_from_thread(self._populate_search, results)
            self.app.call_from_thread(status.update, f"Found {len(results)} models")
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

    # ── Download with full file list ───────────────────────────────
    @work(thread=True, exclusive=True, group="download")
    def _do_download(self, model_id: str) -> None:
        from src.llms.model_manager import DownloadCancelled

        self._downloading_model_id = model_id
        mm = self.app.model_manager  # type: ignore[attr-defined]

        panel = self.query_one("#dl-panel", Container)
        header = self.query_one("#dl-header", Static)
        ftbl = self.query_one("#dl-file-table", DataTable)
        overall_lbl = self.query_one("#dl-overall-label", Static)
        pbar = self.query_one("#dl-progress", ProgressBar)
        btn_stop = self.query_one("#btn-stop-dl", Button)
        btn_resume = self.query_one("#btn-resume-dl", Button)
        status_w = self.query_one("#dl-status", Static)

        # Show panel, reset UI
        self.app.call_from_thread(setattr, panel, "display", True)
        self.app.call_from_thread(
            header.update,
            f"⬇  Downloading [b]{model_id}[/b]",
        )
        self.app.call_from_thread(ftbl.clear)
        self.app.call_from_thread(pbar.update, total=100, progress=0)
        self.app.call_from_thread(setattr, btn_stop, "display", True)
        self.app.call_from_thread(setattr, btn_resume, "display", False)
        self.app.call_from_thread(status_w.update, "")

        # ── Callbacks ──────────────────────────────────────────────
        def _on_file_list(files: list[dict]) -> None:
            """Populate the file table with all repo files (pending)."""
            self._file_keys = []
            self.app.call_from_thread(ftbl.clear)
            for f in files:
                key = f["name"]
                self._file_keys.append(key)
                self.app.call_from_thread(
                    ftbl.add_row,
                    "[dim]⏳ pending[/dim]",
                    f["name"],
                    f["size_str"],
                    key=key,
                )
            total_size = mm._human_size(sum(f["size"] for f in files))
            self.app.call_from_thread(
                overall_lbl.update,
                f"  Total: {len(files)} files  |  {total_size}",
            )

        def _on_progress(
            fname: str,
            idx: int,
            total: int,
            bytes_done: int,
            bytes_total: int,
            file_status: str,
            speed: float = 0.0,
        ) -> None:
            pct = (bytes_done / bytes_total * 100) if bytes_total else 0
            size_done = mm._human_size(bytes_done)
            size_total = mm._human_size(bytes_total)
            speed_str = f"{mm._human_size(int(speed))}/s" if speed > 0 else "—"

            # Overall progress bar
            self.app.call_from_thread(
                pbar.update, total=100, progress=min(pct, 100)
            )
            self.app.call_from_thread(
                overall_lbl.update,
                f"  [{idx}/{total}]  {size_done} / {size_total}  ⚡ {speed_str}",
            )

            # Per-file status in the DataTable
            if file_status == "downloading":
                self._update_file_row(ftbl, fname, "[cyan]⬇  downloading[/cyan]")
            elif file_status == "done":
                self._update_file_row(ftbl, fname, "[green]✓  done[/green]")

        # ── Run download ───────────────────────────────────────────
        try:
            mm.download_model_with_progress(
                model_id,
                on_progress=_on_progress,
                on_file_list=_on_file_list,
            )
            # ── Success ────────────────────────────────────────────
            self.app.call_from_thread(pbar.update, total=100, progress=100)
            self.app.call_from_thread(
                header.update,
                f"[green]✓  Download complete — {model_id}[/green]",
            )
            self.app.call_from_thread(setattr, btn_stop, "display", False)
            self.app.call_from_thread(self._refresh_downloaded)
            self.app.call_from_thread(
                self.app.notify, f"Model {model_id} downloaded ✓"
            )
            self._downloading_model_id = None

        except DownloadCancelled:
            self._mark_remaining_stopped(ftbl)
            self.app.call_from_thread(
                header.update,
                f"[yellow]⏸  Paused:[/yellow] {model_id}",
            )
            self.app.call_from_thread(setattr, btn_stop, "display", False)
            self.app.call_from_thread(setattr, btn_resume, "display", True)
            self.app.call_from_thread(
                self.app.notify,
                "Download stopped — press Resume to continue",
                severity="warning",
            )

        except Exception as exc:
            self.app.call_from_thread(
                header.update,
                f"[red]✗  Error:[/red] {exc}",
            )
            self.app.call_from_thread(setattr, btn_stop, "display", False)
            self._downloading_model_id = None

    # ── Helpers: update file row in DataTable ──────────────────────
    def _update_file_row(
        self, ftbl: DataTable, fname: str, status_text: str
    ) -> None:
        """Update the Status column for *fname* in the file table."""
        try:
            from textual.widgets._data_table import RowKey

            row_key = RowKey(fname)
            col_key = ftbl.columns[list(ftbl.columns.keys())[0]].key
            self.app.call_from_thread(
                ftbl.update_cell, row_key, col_key, status_text
            )
        except Exception:
            pass  # row may not exist yet

    def _mark_remaining_stopped(self, ftbl: DataTable) -> None:
        """Mark any still-pending files as ⏸ stopped."""
        try:
            from textual.widgets._data_table import RowKey

            col_key = ftbl.columns[list(ftbl.columns.keys())[0]].key
            for fname in self._file_keys:
                row_key = RowKey(fname)
                try:
                    current = ftbl.get_cell(row_key, col_key)
                    if "pending" in str(current):
                        self.app.call_from_thread(
                            ftbl.update_cell,
                            row_key,
                            col_key,
                            "[yellow]⏸ stopped[/yellow]",
                        )
                except Exception:
                    pass
        except Exception:
            pass

    # ── Download actions ───────────────────────────────────────────
    def _download_selected(self) -> None:
        tbl = self.query_one("#search-results", DataTable)
        if tbl.cursor_row is not None and tbl.row_count > 0:
            row_key, _ = tbl.coordinate_to_cell_key(tbl.cursor_coordinate)
            model_id = str(row_key.value)
            self._do_download(model_id)
        else:
            self.app.notify(  # type: ignore[attr-defined]
                "Select a model from search results first",
                severity="warning",
            )

    def _stop_download(self) -> None:
        self.app.model_manager.cancel_download()  # type: ignore[attr-defined]
        self.app.notify("Stopping after current file…")  # type: ignore[attr-defined]

    def _resume_download(self) -> None:
        """Resume = re-run download; already-fetched files are auto-skipped."""
        if self._downloading_model_id:
            mid = self._downloading_model_id
            self.app.notify(f"Resuming {mid}…")  # type: ignore[attr-defined]
            self._do_download(mid)

    # ── Load / Unload / Delete ─────────────────────────────────────
    @work(thread=True, exclusive=True, group="load")
    def _do_load(self, model_id: str) -> None:
        status = self.query_one("#dl-status", Static)
        self.app.call_from_thread(
            status.update, f"Loading [b]{model_id}[/b]…"
        )
        try:
            self.app.inference_engine.load_model(model_id)  # type: ignore[attr-defined]
            self.app.config.active_model = model_id  # type: ignore[attr-defined]
            self.app.config.save()  # type: ignore[attr-defined]
            self.app.call_from_thread(
                status.update, f"[green]✓ Model {model_id} loaded[/green]"
            )
            self.app.call_from_thread(self._refresh_downloaded)
            self.app.call_from_thread(self.app._update_sidebar_status)
            self.app.call_from_thread(
                self.app.notify, f"Model loaded: {model_id}"
            )
        except Exception as exc:
            self.app.call_from_thread(
                status.update, f"[red]Load error: {exc}[/red]"
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
        if app.inference_engine.is_loaded:
            app.inference_engine.unload_model()
            app.config.active_model = ""
            app.config.save()
            app.notify("Model unloaded")
            self._refresh_downloaded()
            self.query_one("#dl-status", Static).update(
                "[yellow]Model unloaded[/yellow]"
            )
            app._update_sidebar_status()
        else:
            app.notify("No model is loaded", severity="warning")

    def _delete_selected(self) -> None:
        dtbl = self.query_one("#downloaded-models", DataTable)
        if dtbl.cursor_row is not None and dtbl.row_count > 0:
            row_key, _ = dtbl.coordinate_to_cell_key(dtbl.cursor_coordinate)
            model_id = str(row_key.value)
            app = self.app  # type: ignore[attr-defined]

            # Unload first if loaded
            if app.inference_engine.model_id == model_id:
                app.inference_engine.unload_model()
                app.config.active_model = ""
                app.config.save()

            if app.model_manager.delete_model(model_id):
                app.notify(f"Deleted {model_id}")
            else:
                app.notify(f"Could not delete {model_id}", severity="error")
            self._refresh_downloaded()
            app._update_sidebar_status()
        else:
            self.app.notify("Select a model first", severity="warning")  # type: ignore[attr-defined]
