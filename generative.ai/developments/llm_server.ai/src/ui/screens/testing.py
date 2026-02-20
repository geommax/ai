"""Testing screen — interactive model prompt / response playground."""

from __future__ import annotations

from textual import work
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Static, Button, TextArea, RichLog


class TestingScreen(Container):
    """Send prompts to the loaded model and inspect responses."""

    DEFAULT_CSS = """
    TestingScreen {
        padding: 1 2;
        overflow-y: auto;
    }
    .test-section {
        background: #24283b;
        border: solid #414868;
        padding: 1 2;
        margin: 0 0 1 0;
        height: auto;
    }
    #system-prompt {
        height: 4;
        margin: 0 0 1 0;
    }
    #user-prompt {
        height: 6;
        margin: 0 0 1 0;
    }
    #test-btn-row {
        height: 3;
        margin: 0 0 1 0;
    }
    #output-log {
        height: 16;
        border: solid #414868;
        background: #1a1b26;
        margin: 0 0 1 0;
    }
    #gen-status {
        height: auto;
        color: #7aa2f7;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static("🧪  Model Testing", classes="panel-title")

        with Container(classes="test-section"):
            yield Static("📝  [b]Active Model:[/b]", id="test-model-label", markup=True)
            yield Static("", id="test-active-model")

        # ── Input ──────────────────────────────────────────────────
        with Container(classes="test-section"):
            yield Static("[b]System Prompt[/b]  [dim](optional)[/dim]", markup=True)
            yield TextArea(
                "You are a helpful assistant.",
                id="system-prompt",
                language="markdown",
            )

            yield Static("[b]User Prompt[/b]", markup=True)
            yield TextArea(
                "",
                id="user-prompt",
                language="markdown",
            )

            with Horizontal(id="test-btn-row"):
                yield Button("🚀  Generate", id="btn-generate", variant="success")
                yield Button("🗑  Clear Output", id="btn-clear", variant="default")
            yield Static("", id="gen-status")

        # ── Output ─────────────────────────────────────────────────
        with Container(classes="test-section"):
            yield Static("[b]Output[/b]", markup=True)
            yield RichLog(id="output-log", highlight=True, markup=True)

    # ── Lifecycle ──────────────────────────────────────────────────
    def on_mount(self) -> None:
        self._refresh_model_label()

    def _refresh_model_label(self) -> None:
        engine = self.app.inference_engine  # type: ignore[attr-defined]
        label = self.query_one("#test-active-model", Static)
        if engine.is_loaded:
            label.update(f"  [green]✓[/green] {engine.model_id}")
        else:
            label.update("  [red]✗[/red] [dim]No model loaded — go to Models tab to load one[/dim]")

    # ── Handlers ───────────────────────────────────────────────────
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-generate":
            self._run_generate()
        elif event.button.id == "btn-clear":
            self.query_one("#output-log", RichLog).clear()
            self.query_one("#gen-status", Static).update("")

    @work(thread=True, exclusive=True, group="generate")
    def _run_generate(self) -> None:
        app = self.app  # type: ignore[attr-defined]
        engine = app.inference_engine

        if not engine.is_loaded:
            self.app.call_from_thread(
                app.notify, "No model loaded", severity="error"
            )
            return

        system_text = self.query_one("#system-prompt", TextArea).text.strip()
        user_text = self.query_one("#user-prompt", TextArea).text.strip()

        if not user_text:
            self.app.call_from_thread(
                app.notify, "Enter a prompt first", severity="warning"
            )
            return

        status = self.query_one("#gen-status", Static)
        self.app.call_from_thread(status.update, "⏳ Generating…")

        # Build messages
        messages = []
        if system_text:
            messages.append({"role": "system", "content": system_text})
        messages.append({"role": "user", "content": user_text})

        tuning = app.config.tuning
        try:
            response = engine.chat_generate(
                messages,
                max_tokens=tuning.max_tokens,
                temperature=tuning.temperature,
                top_p=tuning.top_p,
                top_k=tuning.top_k,
                repetition_penalty=tuning.repetition_penalty,
                do_sample=tuning.do_sample,
            )
            log = self.query_one("#output-log", RichLog)
            self.app.call_from_thread(log.write, f"[bold cyan]User:[/bold cyan] {user_text}")
            self.app.call_from_thread(log.write, f"[bold green]Assistant:[/bold green] {response}")
            self.app.call_from_thread(log.write, "─" * 60)
            self.app.call_from_thread(status.update, "[green]✓ Done[/green]")
        except Exception as exc:
            self.app.call_from_thread(
                status.update, f"[red]Error: {exc}[/red]"
            )
