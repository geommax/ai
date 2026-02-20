"""Reusable modal dialogs for the TUI."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal
from textual.screen import ModalScreen
from textual.widgets import Static, Button


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
