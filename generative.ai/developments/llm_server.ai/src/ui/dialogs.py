"""Reusable modal dialogs for the TUI."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal
from textual.screen import ModalScreen
from textual.widgets import Static, Button, DataTable


class ConfirmDialog(ModalScreen[bool]):
    """Yes / No confirmation dialog (Tokyo Night themed).

    Usage::

        def callback(confirmed: bool) -> None:
            if confirmed:
                ...  # do the dangerous thing

        app.push_screen(
            ConfirmDialog("Delete this item?", title="🗑  Confirm Delete"),
            callback,
        )
    """

    DEFAULT_CSS = """
    ConfirmDialog {
        align: center middle;
    }
    #confirm-box {
        width: 60;
        height: auto;
        max-height: 18;
        background: #24283b;
        border: solid #7aa2f7;
        padding: 1 2;
    }
    #confirm-title {
        text-style: bold;
        color: #f7768e;
        text-align: center;
        margin: 0 0 1 0;
    }
    #confirm-message {
        color: #c0caf5;
        text-align: center;
        margin: 1 0;
    }
    #confirm-buttons {
        height: 3;
        align: center middle;
        margin: 1 0 0 0;
    }
    #confirm-yes {
        margin: 0 2 0 0;
    }
    """

    def __init__(
        self,
        message: str,
        *,
        title: str = "⚠  Confirm",
    ) -> None:
        super().__init__()
        self._message = message
        self._title = title

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm-box"):
            yield Static(f"[b]{self._title}[/b]", id="confirm-title", markup=True)
            yield Static(self._message, id="confirm-message", markup=True)
            with Horizontal(id="confirm-buttons"):
                yield Button("Yes", id="confirm-yes", variant="error")
                yield Button("No", id="confirm-no", variant="default")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "confirm-yes")

    def on_key(self, event) -> None:
        if event.key == "y":
            self.dismiss(True)
        elif event.key in ("n", "escape"):
            self.dismiss(False)


class FilePickerDialog(ModalScreen[list[str] | None]):
    """Let the user pick one or more files from a repo before downloading.

    Returns a list of selected filenames, or ``None`` if cancelled.
    """

    DEFAULT_CSS = """
    FilePickerDialog {
        align: center middle;
    }
    #fp-box {
        width: 80;
        height: auto;
        max-height: 32;
        background: #24283b;
        border: solid #7aa2f7;
        padding: 1 2;
    }
    #fp-title {
        text-style: bold;
        color: #7aa2f7;
        text-align: center;
        margin: 0 0 1 0;
    }
    #fp-hint {
        color: #565f89;
        margin: 0 0 1 0;
    }
    #fp-table {
        height: auto;
        max-height: 20;
        margin: 0 0 1 0;
    }
    #fp-buttons {
        height: 3;
        align: center middle;
        margin: 1 0 0 0;
    }
    #fp-dl-selected {
        margin: 0 1 0 0;
    }
    #fp-dl-all {
        margin: 0 1 0 0;
    }
    """

    def __init__(
        self,
        model_id: str,
        files: list[dict],
        *,
        title: str = "📂  Select files to download",
    ) -> None:
        super().__init__()
        self._model_id = model_id
        self._files = files
        self._title = title
        self._selected: set[str] = set()

    def compose(self) -> ComposeResult:
        with Vertical(id="fp-box"):
            yield Static(
                f"[b]{self._title}[/b]\n{self._model_id}",
                id="fp-title",
                markup=True,
            )
            yield Static(
                "Select a file with [b]Enter[/b] to toggle, then confirm.",
                id="fp-hint",
                markup=True,
            )
            yield DataTable(id="fp-table")
            with Horizontal(id="fp-buttons"):
                yield Button(
                    "⬇  Download Selected",
                    id="fp-dl-selected",
                    variant="success",
                )
                yield Button(
                    "📦 Download All",
                    id="fp-dl-all",
                    variant="primary",
                )
                yield Button("Cancel", id="fp-cancel", variant="default")

    def on_mount(self) -> None:
        tbl = self.query_one("#fp-table", DataTable)
        tbl.add_columns("✓", "File", "Size")
        tbl.cursor_type = "row"
        for f in self._files:
            tbl.add_row(
                " ",
                f["name"],
                f.get("size_str", "?"),
                key=f["name"],
            )

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Toggle selection when the user presses Enter on a row."""
        fname = str(event.row_key.value)
        tbl = self.query_one("#fp-table", DataTable)
        from textual.widgets._data_table import RowKey

        col_key = tbl.columns[list(tbl.columns.keys())[0]].key
        if fname in self._selected:
            self._selected.discard(fname)
            tbl.update_cell(RowKey(fname), col_key, " ")
        else:
            self._selected.add(fname)
            tbl.update_cell(RowKey(fname), col_key, "[green]✓[/green]")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id
        if bid == "fp-dl-selected":
            if self._selected:
                self.dismiss(list(self._selected))
            else:
                self.app.notify(
                    "Select at least one file first", severity="warning"
                )
        elif bid == "fp-dl-all":
            self.dismiss(None)  # None = download all
        elif bid == "fp-cancel":
            self.dismiss("__cancel__")  # sentinel: user cancelled

    def on_key(self, event) -> None:
        if event.key == "escape":
            self.dismiss("__cancel__")
