"""Live terminal dashboard for experiment runs."""

from __future__ import annotations

import os
import shutil
import statistics
import sys
import time
from collections import OrderedDict


class TerminalDashboard:
    """Render a compact, Claude Code-inspired progress view on a TTY."""

    _HEIGHT = 13

    def __init__(
        self,
        *,
        agent: str,
        environment: str,
        optimizer: str,
        fitness_strategy: str,
        runs: int,
        workers: int,
        enabled: bool = True,
    ):
        self.agent = agent
        self.environment = environment
        self.optimizer = optimizer
        self.fitness_strategy = fitness_strategy
        self.runs = runs
        self.workers = workers
        self.completed: OrderedDict[int, tuple[int, object]] = OrderedDict()
        self.started_at = time.perf_counter()
        self.enabled = enabled and (
            sys.stdout.isatty()
            and os.environ.get("TERM", "") != "dumb"
        )
        self.color_enabled = self.enabled and "NO_COLOR" not in os.environ
        self.width = max(54, min(shutil.get_terminal_size((80, 24)).columns, 88))
        self._has_rendered = False
        self.status = "running"

    def start(self):
        if self.enabled:
            self._render()

    def update(self, run_index: int, seed: int, result):
        self.completed[run_index] = (seed, result)
        if self.enabled:
            self._render(status=self.status)

    def set_status(self, status: str):
        self.status = status
        if self.enabled:
            self._render(status=status)

    def finish(self, status: str = "complete"):
        if not self.enabled:
            return
        self.status = status
        self._render(status=status)
        sys.stdout.write("\n")
        sys.stdout.flush()

    def _render(self, status: str = "running"):
        frame = self._frame(status)
        if self._has_rendered:
            sys.stdout.write(f"\x1b[{self._HEIGHT}A\r")
        sys.stdout.write("\n".join(frame) + "\n")
        sys.stdout.flush()
        self._has_rendered = True

    def _frame(self, status: str) -> list[str]:
        done = len(self.completed)
        duration = time.perf_counter() - self.started_at
        result_values = [self._final_metrics(result) for _, result in self.completed.values()]
        success_rates = [values[0] for values in result_values]

        progress_width = min(24, self.width - 34)
        filled = round(progress_width * done / max(self.runs, 1))
        progress = self._paint("█" * filled, "orange") + self._paint(
            "░" * (progress_width - filled), "muted"
        )
        if status == "complete":
            state = self._paint("✓ concluído", "green")
        elif status == "stopping":
            state = self._paint("… encerrando workers", "orange")
        elif status == "stopped":
            state = self._paint("■ encerrado", "muted")
        elif status == "failed":
            state = self._paint("! falhou", "red")
        else:
            state = self._paint("● em execução", "orange")

        if success_rates:
            average_success = statistics.fmean(success_rates) * 100
            result_summary = f"Sucesso médio {average_success:5.1f}%"
        else:
            result_summary = "Aguardando a primeira execução concluir"

        recent_runs = list(self.completed.items())[-3:]
        recent_lines = []
        for offset in range(3):
            if offset < len(recent_runs):
                run_index, (seed, result) = recent_runs[offset]
                success, best_fitness = self._final_metrics(result)
                line = (
                    f"✓ run {run_index:02d}  seed {seed}"
                    f"  sucesso {success * 100:5.1f}%"
                    f"  melhor {best_fitness:.4f}"
                )
                recent_lines.append(self._paint(line, "green"))
            else:
                recent_lines.append(self._paint("· aguardando execução", "muted"))

        inner_width = self.width - 4
        title = (
            f"❯ evolutionary-agent-testbed  {state}"
        )
        context = (
            f"{self.environment}  ·  {self.agent}  ·  {self.optimizer}"
            f"  ·  {self.fitness_strategy}"
        )
        progress_line = f"[{progress}]  {done:02d}/{self.runs:02d} runs"
        footer = (
            f"workers {self.workers}  ·  tempo {self._format_duration(duration)}"
            "  ·  Ctrl+C para parar"
        )

        content = [
            self._border("╭", "─", "╮"),
            self._row(title, inner_width),
            self._border("├", "─", "┤"),
            self._row(context, inner_width),
            self._row(progress_line, inner_width),
            self._row(result_summary, inner_width),
            self._border("├", "─", "┤"),
            self._row("Execuções recentes", inner_width),
            *(self._row(line, inner_width) for line in recent_lines),
            self._row(footer, inner_width),
            self._border("╰", "─", "╯"),
        ]

        # Keep a stable height so redraws never leave stale rows behind.
        while len(content) < self._HEIGHT:
            content.insert(-1, self._row("", inner_width))
        return content[:self._HEIGHT]

    @staticmethod
    def _final_metrics(result) -> tuple[float, float]:
        metric = result.metrics[-1]
        return metric.success_rate, metric.best_fitness

    def _border(self, left: str, fill: str, right: str) -> str:
        return self._paint(left + fill * (self.width - 2) + right, "muted")

    def _row(self, text: str, inner_width: int) -> str:
        visible = self._visible_length(text)
        if visible > inner_width:
            text = self._truncate(text, inner_width)
            visible = inner_width
        padding = " " * (inner_width - visible)
        return f"│ {text}{padding} │"

    def _truncate(self, text: str, limit: int) -> str:
        plain_limit = max(0, limit - 1)
        result = []
        visible = 0
        index = 0
        while index < len(text) and visible < plain_limit:
            if text[index] == "\x1b":
                end = text.find("m", index)
                if end >= 0:
                    result.append(text[index:end + 1])
                    index = end + 1
                    continue
            result.append(text[index])
            visible += 1
            index += 1
        return "".join(result) + self._paint("…", "muted")

    @staticmethod
    def _visible_length(text: str) -> int:
        visible = 0
        index = 0
        while index < len(text):
            if text[index] == "\x1b":
                end = text.find("m", index)
                if end >= 0:
                    index = end + 1
                    continue
            visible += 1
            index += 1
        return visible

    def _paint(self, text: str, color: str) -> str:
        if not self.color_enabled:
            return text
        codes = {
            "orange": "38;5;208",
            "green": "38;5;114",
            "red": "38;5;203",
            "muted": "38;5;245",
        }
        return f"\x1b[{codes[color]}m{text}\x1b[0m"

    @staticmethod
    def _format_duration(seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.0f}s"
        minutes, remaining = divmod(int(seconds), 60)
        return f"{minutes}m {remaining:02d}s"
