"""Tuning screen — adjust generation hyper-parameters (daemon client)."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.widgets import Static, Button, Input, Switch, Label

from src.daemon.client import DaemonDisconnected


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

            with ParamRow():
                yield Label("Temperature")
                yield Input("0.7", id="tune-temperature")
                yield Static("0.0 – 2.0  (lower = deterministic)", classes="hint")

            with ParamRow():
                yield Label("Top P")
                yield Input("0.9", id="tune-top-p")
                yield Static("0.0 – 1.0  (nucleus sampling)", classes="hint")

            with ParamRow():
                yield Label("Top K")
                yield Input("50", id="tune-top-k")
                yield Static("1 – 200  (vocabulary filter)", classes="hint")

            with ParamRow():
                yield Label("Max New Tokens")
                yield Input("512", id="tune-max-tokens")
                yield Static("1 – 8192  (output length)", classes="hint")

            with ParamRow():
                yield Label("Repetition Penalty")
                yield Input("1.1", id="tune-rep-penalty")
                yield Static("1.0 – 3.0  (1.0 = off)", classes="hint")

            with Horizontal(classes="switch-row"):
                yield Label("Do Sample")
                yield Switch(value=True, id="tune-do-sample")
                yield Static("On = stochastic, Off = greedy", classes="hint")

            with Horizontal(id="tune-btn-row"):
                yield Button("💾  Save", id="btn-save-tuning", variant="success")
                yield Button("🔄  Reset Defaults", id="btn-reset-tuning", variant="warning")

            yield Static("", id="tune-status")

    # ── Lifecycle ──────────────────────────────────────────────────
    def on_mount(self) -> None:
        self._load_from_config()

    def _load_from_config(self) -> None:
        try:
            cfg = self.app.client.get_config()  # type: ignore[attr-defined]
        except DaemonDisconnected:
            return
        self.query_one("#tune-temperature", Input).value = str(cfg.get("temperature", 0.7))
        self.query_one("#tune-top-p", Input).value = str(cfg.get("top_p", 0.9))
        self.query_one("#tune-top-k", Input).value = str(cfg.get("top_k", 50))
        self.query_one("#tune-max-tokens", Input).value = str(cfg.get("max_tokens", 512))
        self.query_one("#tune-rep-penalty", Input).value = str(cfg.get("repetition_penalty", 1.1))
        self.query_one("#tune-do-sample", Switch).value = bool(cfg.get("do_sample", True))

    # ── Handlers ───────────────────────────────────────────────────
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-save-tuning":
            self._save_tuning()
        elif event.button.id == "btn-reset-tuning":
            self._reset_defaults()

    def _save_tuning(self) -> None:
        app = self.app  # type: ignore[attr-defined]
        status = self.query_one("#tune-status", Static)
        try:
            temp = self._clamp(float(self.query_one("#tune-temperature", Input).value), 0.0, 2.0)
            top_p = self._clamp(float(self.query_one("#tune-top-p", Input).value), 0.0, 1.0)
            top_k = int(self._clamp(float(self.query_one("#tune-top-k", Input).value), 1, 200))
            max_tokens = int(self._clamp(float(self.query_one("#tune-max-tokens", Input).value), 1, 8192))
            rep = self._clamp(float(self.query_one("#tune-rep-penalty", Input).value), 1.0, 3.0)
            do_sample = self.query_one("#tune-do-sample", Switch).value

            result = app.client.set_tuning(
                temperature=temp,
                top_p=top_p,
                top_k=top_k,
                max_tokens=max_tokens,
                repetition_penalty=rep,
                do_sample=do_sample,
            )
            if result["ok"]:
                self._load_from_config()
                status.update("[green]✓ Settings saved[/green]")
                app.notify("Tuning parameters saved ✓")
            else:
                status.update(f"[red]{result.get('error', 'Failed')}[/red]")
        except ValueError:
            status.update("[red]Invalid number — check your inputs[/red]")
            app.notify("Invalid input", severity="error")
        except DaemonDisconnected:
            status.update("[red]Daemon offline[/red]")

    def _reset_defaults(self) -> None:
        app = self.app  # type: ignore[attr-defined]
        try:
            result = app.client.reset_tuning()
            if result["ok"]:
                self._load_from_config()
                self.query_one("#tune-status", Static).update(
                    "[yellow]↺ Reset to defaults[/yellow]"
                )
                app.notify("Reset to defaults ✓")
        except DaemonDisconnected:
            app.notify("Daemon offline", severity="error")

    @staticmethod
    def _clamp(val: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, val))
