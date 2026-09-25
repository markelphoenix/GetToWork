"""Terminal UI helpers built on `rich`.

All user interaction goes through the `UI` class so the game can be driven by
scripted input in tests: pass `input_fn` / `secret_fn` / `open_url_fn` and a
`Console(record=True)` or `Console(file=io.StringIO())`.
"""

from __future__ import annotations

import getpass
import webbrowser
from contextlib import contextmanager
from typing import Callable, Iterable, Iterator, Optional, Sequence

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.table import Table

InputFn = Callable[[str], str]


class UserQuit(Exception):
    """Raised when the player presses Ctrl+C / Ctrl+D at a prompt."""


class UI:
    def __init__(
        self,
        console: Optional[Console] = None,
        input_fn: Optional[InputFn] = None,
        secret_fn: Optional[InputFn] = None,
        open_url_fn: Optional[Callable[[str], bool]] = None,
    ) -> None:
        self.console = console or Console()
        self._input = input_fn or (lambda prompt: self.console.input(prompt))
        self._secret = secret_fn or getpass.getpass
        self._open_url = open_url_fn or webbrowser.open

    # -- output -------------------------------------------------------------

    def say(self, text: str = "", style: Optional[str] = None) -> None:
        self.console.print(text, style=style, highlight=False)

    def markdown(self, text: str) -> None:
        self.console.print(Markdown(text))

    def heading(self, text: str) -> None:
        self.console.rule(f"[bold]{text}[/bold]")

    def narrate(self, text: str, title: Optional[str] = None) -> None:
        """Story text from the game master (the local LLM)."""
        self.console.print(Panel(text.strip(), title=title, border_style="magenta", padding=(1, 2)))

    def teach(self, title: str, body: str) -> None:
        """An edutainment aside explaining a concept. `body` is Markdown."""
        self.console.print(Panel(Markdown(body), title=f"Learn: {title}", border_style="cyan", padding=(0, 1)))

    def info(self, text: str) -> None:
        self.console.print(f"[cyan]i[/cyan] {text}", highlight=False)

    def success(self, text: str) -> None:
        self.console.print(f"[green]OK[/green] {text}", highlight=False)

    def warn(self, text: str) -> None:
        self.console.print(f"[yellow]![/yellow] {text}", highlight=False)

    def error(self, text: str) -> None:
        self.console.print(f"[red]x[/red] {text}", highlight=False)

    def table(self, title: Optional[str], columns: Sequence[str], rows: Iterable[Sequence[str]]) -> None:
        t = Table(title=title, show_lines=False)
        for c in columns:
            t.add_column(c)
        for r in rows:
            t.add_row(*[str(x) for x in r])
        self.console.print(t)

    def json(self, data: object, title: Optional[str] = None) -> None:
        import json as _json

        from rich.syntax import Syntax

        text = _json.dumps(data, indent=2, ensure_ascii=False, default=str)
        self.console.print(Panel(Syntax(text, "json", word_wrap=True), title=title, border_style="blue"))

    @contextmanager
    def status(self, text: str) -> Iterator[None]:
        """Spinner while waiting (e.g. for the LLM or Jev)."""
        with self.console.status(text):
            yield

    @contextmanager
    def download_progress(self, description: str, total_bytes: Optional[int]) -> Iterator[Callable[[int], None]]:
        """Yields `advance(n_bytes)`; renders a download bar."""
        progress = Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
            console=self.console,
        )
        with progress:
            task = progress.add_task(description, total=total_bytes)
            yield lambda n: progress.advance(task, n)

    # -- input --------------------------------------------------------------

    def ask(self, prompt: str, default: Optional[str] = None) -> str:
        suffix = f" [dim]({default})[/dim]" if default else ""
        try:
            raw = self._input(f"[bold]{prompt}[/bold]{suffix} > ")
        except (EOFError, KeyboardInterrupt) as exc:
            raise UserQuit() from exc
        raw = (raw or "").strip()
        return raw if raw else (default or "")

    def confirm(self, prompt: str, default: bool = False) -> bool:
        hint = "Y/n" if default else "y/N"
        while True:
            ans = self.ask(f"{prompt} [{hint}]").lower()
            if not ans:
                return default
            if ans in ("y", "yes"):
                return True
            if ans in ("n", "no"):
                return False
            self.warn("Please answer y or n.")

    def choose(self, prompt: str, options: Sequence[tuple[str, str]], default: Optional[str] = None) -> str:
        """Menu. `options` is [(key, label)]; returns the chosen key.

        The player may type the key or the 1-based number of the option.
        """
        keys = [k for k, _ in options]
        while True:
            for i, (k, label) in enumerate(options, 1):
                marker = " [dim](default)[/dim]" if k == default else ""
                self.console.print(f"  [bold cyan]{i}[/bold cyan]) [bold]{k}[/bold] - {label}{marker}", highlight=False)
            ans = self.ask(prompt, default=default)
            if ans in keys:
                return ans
            if ans.isdigit() and 1 <= int(ans) <= len(options):
                return keys[int(ans) - 1]
            lowered = {k.lower(): k for k in keys}
            if ans.lower() in lowered:
                return lowered[ans.lower()]
            self.warn("Pick one of the options above (number or name).")

    def secret(self, prompt: str) -> str:
        """Hidden input (API keys). Nothing is echoed."""
        try:
            return (self._secret(f"{prompt}: ") or "").strip()
        except (EOFError, KeyboardInterrupt) as exc:
            raise UserQuit() from exc

    def open_url(self, url: str) -> None:
        """Try to open a browser; always print the URL too, in case it fails."""
        self.info(f"Opening [link={url}]{url}[/link]")
        try:
            ok = self._open_url(url)
        except Exception:
            ok = False
        if not ok:
            self.warn(f"Couldn't open a browser automatically. Copy this link instead: {url}")
