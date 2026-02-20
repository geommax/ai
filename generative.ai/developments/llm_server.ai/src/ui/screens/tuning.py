"""Tuning screen — adjust generation hyper-parameters."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.widgets import Static, Button, Input, Switch, Label


class ParamRow(Horizontal):
    """A single parameter row: label + input + hint."""

    DEFAULT_CSS = """
    ParamRow {
        height: 3;
        margin: 0 0 1 0;
        align: left middle;
    }
    ParamRow > Label {
        width: 24;
        padding: 0 1;
    }
    ParamRow > Input {
        width: 16;
    }
    ParamRow > .hint {
        width: 1fr;
        padding: 0 2;
        color: #565f89;
    }
    """


class TuningScreen(Container):
    """Configure generation parameters (temperature, top-p, …)."""

    DEFAULT_CSS = """
    TuningScreen {
        padding: 1 2;
        overflow-y: auto;
    }
    .tune-section {
        background: #24283b;
        border: solid #414868;
        padding: 1 2;
        margin: 0 0 1 0;
        height: auto;
    }
    #tune-btn-row {
        height: 3;
        margin: 1 0 0 0;
    }
    #tune-status {
        height: auto;
        margin: 1 0;
        color: #9ece6a;
    }
    .switch-row {
        height: 3;
        margin: 0 0 1 0;
        align: left middle;
    }
    .switch-row > Label {
        width: 24;
        padding: 0 1;
    }
    .switch-row > .hint {
        width: 1fr;
        padding: 0 2;
        color: #565f89;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static("⚙️  Output Tuning", classes="panel-title")

        with Container(classes="tune-section"):
            yield Static(
                "[b]Generation Parameters[/b]  "
                "[dim]— these apply to both Testing and API calls[/dim]",
                markup=True,
            )

            # Temperature
            with ParamRow():
                yield Label("Temperature")
                yield Input("0.7", id="tune-temperature")
                yield Static("0.0 – 2.0  (lower = deterministic)", classes="hint")

            # Top P
            with ParamRow():
                yield Label("Top P")
                yield Input("0.9", id="tune-top-p")
                yield Static("0.0 – 1.0  (nucleus sampling)", classes="hint")

            # Top K
            with ParamRow():
                yield Label("Top K")
                yield Input("50", id="tune-top-k")
                yield Static("1 – 200  (vocabulary filter)", classes="hint")

            # Max Tokens
            with ParamRow():
                yield Label("Max New Tokens")
                yield Input("512", id="tune-max-tokens")
                yield Static("1 – 8192  (output length)", classes="hint")

            # Repetition Penalty
            with ParamRow():
                yield Label("Repetition Penalty")
                yield Input("1.1", id="tune-rep-penalty")
                yield Static("1.0 – 3.0  (1.0 = off)", classes="hint")

            # Do Sample switch
            with Horizontal(classes="switch-row"):
                yield Label("Do Sample")
                yield Switch(value=True, id="tune-do-sample")
                yield Static("On = stochastic, Off = greedy", classes="hint")

            # Buttons
            with Horizontal(id="tune-btn-row"):
                yield Button("💾  Save", id="btn-save-tuning", variant="success")
                yield Button("🔄  Reset Defaults", id="btn-reset-tuning", variant="warning")

            yield Static("", id="tune-status")

    # ── Lifecycle ──────────────────────────────────────────────────
    def on_mount(self) -> None:
        self._load_from_config()

    def _load_from_config(self) -> None:
        t = self.app.config.tuning  # type: ignore[attr-defined]
        self.query_one("#tune-temperature", Input).value = str(t.temperature)
        self.query_one("#tune-top-p", Input).value = str(t.top_p)
        self.query_one("#tune-top-k", Input).value = str(t.top_k)
        self.query_one("#tune-max-tokens", Input).value = str(t.max_tokens)
        self.query_one("#tune-rep-penalty", Input).value = str(t.repetition_penalty)
        self.query_one("#tune-do-sample", Switch).value = t.do_sample

    # ── Handlers ───────────────────────────────────────────────────
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-save-tuning":
            self._save_tuning()
        elif event.button.id == "btn-reset-tuning":
            self._reset_defaults()

    def _save_tuning(self) -> None:
        app = self.app  # type: ignore[attr-defined]
        t = app.config.tuning
        status = self.query_one("#tune-status", Static)
        try:
            t.temperature = self._clamp(
                float(self.query_one("#tune-temperature", Input).value), 0.0, 2.0
            )
            t.top_p = self._clamp(
                float(self.query_one("#tune-top-p", Input).value), 0.0, 1.0
            )
            t.top_k = int(
                self._clamp(
                    float(self.query_one("#tune-top-k", Input).value), 1, 200
                )
            )
            t.max_tokens = int(
                self._clamp(
                    float(self.query_one("#tune-max-tokens", Input).value), 1, 8192
                )
            )
            t.repetition_penalty = self._clamp(
                float(self.query_one("#tune-rep-penalty", Input).value), 1.0, 3.0
            )
            t.do_sample = self.query_one("#tune-do-sample", Switch).value

            app.config.save()
            # Reload inputs to show clamped values
            self._load_from_config()
            status.update("[green]✓ Settings saved[/green]")
            app.notify("Tuning parameters saved ✓")
        except ValueError:
            status.update("[red]Invalid number — check your inputs[/red]")
            app.notify("Invalid input", severity="error")

    def _reset_defaults(self) -> None:
        from src.tuning import TuningParams

        app = self.app  # type: ignore[attr-defined]
        app.config.tuning = TuningParams()
        app.config.save()
        self._load_from_config()
        self.query_one("#tune-status", Static).update(
            "[yellow]↺ Reset to defaults[/yellow]"
        )
        app.notify("Reset to defaults ✓")

    @staticmethod
    def _clamp(val: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, val))
