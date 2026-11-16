"""Native terminal menus for experiment setup and run control."""

from __future__ import annotations

import csv
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

from experiment.aggregated_generation_metrics import (
    AggregatedGenerationMetrics,
)
from experiment.experiment_runner import (
    ExperimentCancelled,
    ExperimentRunner,
)
from experiment.experiment_types import ExperimentType
from visualization.experiment_plotter import ExperimentPlotter


EXPERIMENT_TYPES = (
    ("movement", "MOVEMENT — sequência de ações"),
    ("adaline", "ADALINE — modelo linear"),
    ("mlp", "MLP — rede neural"),
)

EXPERIMENT_MODES = (
    ("single", "Execução única"),
    ("fitness_comparison", "Comparação de estratégias de fitness"),
    ("configuration_comparison", "Comparação de configurações MLP"),
)

OPTIMIZERS = (
    ("genetic", "Algoritmo genético"),
    ("cross_entropy", "Cross-Entropy Method"),
    ("differential_evolution", "Differential Evolution"),
)

ENVIRONMENTS = (
    ("frozen_lake", "Frozen Lake"),
    ("minigrid", "MiniGrid"),
)


class ExperimentTerminalUI:
    """Small keyboard-driven TUI using only Python's standard library."""

    def __init__(self, config, logger, *, output=None, key_reader=None):
        self.config = config
        self.logger = logger
        self.output = output or sys.stdout
        self.key_reader = key_reader or self._read_key
        self.runner: ExperimentRunner | None = None
        self.experiment_running = False
        self.experiment_completed = False
        self.color_enabled = (
            getattr(self.output, "isatty", lambda: False)()
            and "NO_COLOR" not in os.environ
            and os.environ.get("TERM", "") != "dumb"
        )

    @staticmethod
    def mode_options(experiment_type):
        return tuple(
            (value, label)
            for value, label in EXPERIMENT_MODES
            if value != "configuration_comparison"
            or experiment_type == "mlp"
        )

    @staticmethod
    def optimizer_options(experiment_type):
        return OPTIMIZERS if experiment_type == "mlp" else OPTIMIZERS[:1]

    def run(self):
        """Run experiments or create plots from previously saved results."""
        while True:
            try:
                action = self._choose(
                    "Ação principal",
                    (
                        ("run", "Configurar e executar experimento"),
                        ("plot_saved", "Gerar gráfico de resultados salvos"),
                        ("exit", "Encerrar"),
                    ),
                    current="run",
                )
            except KeyboardInterrupt:
                self._write("\n■ Terminal encerrado.\n")
                return

            if action in (None, "exit"):
                self._write("\nTerminal encerrado.\n")
                return

            if action == "plot_saved":
                try:
                    self._plot_saved_results()
                except KeyboardInterrupt:
                    self._write("\n■ Retornando ao menu principal.\n")
                except Exception as error:
                    self._write(
                        f"\n× Falha ao gerar o gráfico: "
                        f"{type(error).__name__}: {error}\n"
                    )
                    self.logger.exception("Saved-result plotting failed")
                continue

            try:
                self._run_configured_experiment()
            except KeyboardInterrupt:
                self._write("\nTerminal encerrado.\n")
                return

    def _run_configured_experiment(self):
        selections = self._select_configuration()
        if selections is None:
            return

        for name, value in selections.items():
            setattr(self.config, name, value)

        self._show_run_header()
        self.runner = ExperimentRunner(
            logger=self.logger,
            experiment_type=self.config.EXPERIMENT_TYPE,
        )
        self.experiment_running = True
        try:
            self.runner.run()
        except ExperimentCancelled:
            self.experiment_completed = False
            self._write(
                "\n■ Experimento cancelado. "
                "Resultados parciais foram descartados.\n"
            )
        except Exception as error:
            self._write(
                f"\n× Falha: {type(error).__name__}: {error}\n"
            )
            self.logger.exception("Experiment failed in terminal UI")
        else:
            self.experiment_completed = True
            self._write("\n✓ Experimento concluído.\n")
        finally:
            self.experiment_running = False

    @staticmethod
    def _saved_plot_files():
        required_columns = {
            "generation",
            "experiment_type",
            "mean_success_rate_percent",
            "std_success_rate_percentage_points",
            "margin_of_error_percentage_points",
            "lower_bound_percent",
            "upper_bound_percent",
        }
        result_directories = (
            Path("logs"),
            Path("experiment_results"),
        )
        saved_files = []

        for directory in result_directories:
            if not directory.is_dir():
                continue
            for path in directory.glob("*.csv"):
                try:
                    with path.open(
                        newline="",
                        encoding="utf-8-sig",
                    ) as csv_file:
                        reader = csv.DictReader(csv_file)
                        if not required_columns.issubset(
                            reader.fieldnames or ()
                        ):
                            continue
                        first_row = next(reader, None)
                except (OSError, UnicodeError, csv.Error):
                    continue

                if not first_row:
                    continue

                agent_type = first_row.get("experiment_type", "")
                environment = first_row.get("environment", "")
                experiment_mode = first_row.get("experiment_mode", "single")
                if experiment_mode == "configuration_comparison":
                    if not {
                        "configuration_name",
                        "hidden_size",
                        "optimizer",
                    }.issubset(reader.fieldnames or ()):
                        continue
                elif experiment_mode != "single":
                    continue

                timestamp_parts = path.stem.rsplit("_", 3)
                try:
                    timestamp = datetime.strptime(
                        "_".join(timestamp_parts[-3:]),
                        "%Y%m%d_%H%M%S_%f",
                    )
                except ValueError:
                    timestamp = datetime.fromtimestamp(path.stat().st_mtime)
                if experiment_mode == "configuration_comparison":
                    summary = (
                        "Comparação MLP · "
                        f"{first_row.get('optimizer', 'otimizador')} · "
                        f"{first_row.get('runs', '?')} execuções"
                    )
                else:
                    summary = (
                        f"{first_row.get('agent', 'Agente')} · "
                        f"{first_row.get('optimizer', 'otimizador')} · "
                        f"{first_row.get('runs', '?')} execuções · "
                        f"{first_row.get('actual_generations', '?')} ciclos"
                    )
                label = f"{summary} · {timestamp:%d/%m/%Y %H:%M:%S}"
                saved_files.append((
                    path,
                    label,
                    agent_type,
                    environment,
                    experiment_mode,
                ))

        return sorted(
            saved_files,
            key=lambda item: item[0].stat().st_mtime,
            reverse=True,
        )

    def _plot_saved_results(self):
        saved_files = self._saved_plot_files()
        if not saved_files:
            self._write(
                "\nNão encontrei arquivos CSV de resultados compatíveis "
                "em logs/ ou experiment_results/.\n"
            )
            return

        agent_labels = {
            "movement": "Agente baseado em movimentos",
            "adaline": "ADALINE",
            "mlp": "MLP",
        }
        available_agents = sorted({item[2] for item in saved_files})
        selected_agent = self._choose(
            "Resultados salvos · agente",
            tuple(
                (agent, agent_labels.get(agent, agent.upper()))
                for agent in available_agents
            ),
        )
        if selected_agent is None:
            return

        agent_files = [
            item for item in saved_files if item[2] == selected_agent
        ]
        environment_labels = {
            "frozen_lake": "Frozen Lake",
            "minigrid": "MiniGrid",
        }
        available_environments = sorted({item[3] for item in agent_files})
        selected_environment = self._choose(
            "Resultados salvos · ambiente",
            tuple(
                (
                    environment,
                    environment_labels.get(environment, environment),
                )
                for environment in available_environments
            ),
        )
        if selected_environment is None:
            return

        environment_files = [
            item for item in agent_files if item[3] == selected_environment
        ]
        selected_path = self._choose(
            "Resultados salvos · execução",
            tuple(
                (str(path), label)
                for path, label, _, _, _ in environment_files
            ),
        )
        if selected_path is None:
            return

        result_path = Path(selected_path)
        with result_path.open(
            newline="",
            encoding="utf-8-sig",
        ) as csv_file:
            reader = csv.DictReader(csv_file)
            rows = list(reader)

        if not rows:
            raise ValueError("O arquivo selecionado não contém métricas.")

        first_row = rows[0]
        experiment_type = ExperimentType(first_row["experiment_type"])
        experiment_mode = first_row.get("experiment_mode", "single")
        if experiment_mode == "configuration_comparison":
            saved_metrics = {}
            for row in rows:
                saved_metrics.setdefault(row["configuration_name"], []).append(
                    AggregatedGenerationMetrics(
                        generation=int(row["generation"]),
                        mean_success_rate=(
                            float(row["mean_success_rate_percent"]) / 100
                        ),
                        std_success_rate=(
                            float(
                                row["std_success_rate_percentage_points"]
                            ) / 100
                        ),
                        margin_of_error=(
                            float(row["margin_of_error_percentage_points"])
                            / 100
                        ),
                        lower_bound=(
                            float(row["lower_bound_percent"]) / 100
                        ),
                        upper_bound=(
                            float(row["upper_bound_percent"]) / 100
                        ),
                    )
                )
            hidden_sizes = sorted({
                int(row["hidden_size"]) for row in rows
            })
            plotter = ExperimentPlotter({}, experiment_type)
            plotter.plot_configuration_comparison(
                saved_metrics=saved_metrics,
                hidden_sizes=hidden_sizes,
                optimizer=first_row.get("optimizer", "genetic"),
            )
        else:
            metrics = [
                AggregatedGenerationMetrics(
                    generation=int(row["generation"]),
                    mean_success_rate=(
                        float(row["mean_success_rate_percent"]) / 100
                    ),
                    std_success_rate=(
                        float(row["std_success_rate_percentage_points"]) / 100
                    ),
                    margin_of_error=(
                        float(row["margin_of_error_percentage_points"]) / 100
                    ),
                    lower_bound=float(row["lower_bound_percent"]) / 100,
                    upper_bound=float(row["upper_bound_percent"]) / 100,
                )
                for row in rows
            ]
            title = first_row.get("plot_title") or (
                f"Taxa de sucesso por geração | "
                f"ambiente={first_row.get('environment', 'desconhecido')} | "
                f"{first_row.get('agent', experiment_type.value)} | "
                f"{first_row.get('optimizer', 'otimizador')}"
            )
            plotter = ExperimentPlotter(metrics, experiment_type)
            plotter.plot_experiment(title=title)

        output_directory = Path("experiment_results") / "plots"
        output_directory.mkdir(parents=True, exist_ok=True)
        output_path = output_directory / f"{result_path.stem}.png"
        plotter.save(output_path)
        plotter.show()
        self._write(f"\n✓ Gráfico salvo em: {output_path}\n")

    def _select_configuration(self):
        experiment_type = self._choose(
            "EXPERIMENT_TYPE · Agente",
            EXPERIMENT_TYPES,
            current=self.config.EXPERIMENT_TYPE,
        )
        if experiment_type is None:
            return None

        mode = self._choose(
            "EXPERIMENT_MODE · Modo",
            self.mode_options(experiment_type),
            current=self.config.EXPERIMENT_MODE,
        )
        if mode is None:
            return None

        optimizer = self._choose(
            "OPTIMIZER · Otimizador",
            self.optimizer_options(experiment_type),
            current=self.config.EXPERIMENT_OPTIMIZER,
        )
        if optimizer is None:
            return None

        environment = self._choose(
            "ENVIRONMENT · Ambiente",
            ENVIRONMENTS,
            current=self.config.ENVIRONMENT_TYPE,
        )
        if environment is None:
            return None

        return {
            "EXPERIMENT_TYPE": experiment_type,
            "EXPERIMENT_MODE": mode,
            "EXPERIMENT_OPTIMIZER": optimizer,
            "ENVIRONMENT_TYPE": environment,
        }

    def _choose(self, title, options, *, current=None):
        values = [value for value, _ in options]
        selected_index = values.index(current) if current in values else 0

        while True:
            self._render_selector(title, options, selected_index)
            key = self.key_reader()

            if key == "up":
                selected_index = (selected_index - 1) % len(options)
            elif key == "down":
                selected_index = (selected_index + 1) % len(options)
            elif key == "enter":
                return options[selected_index][0]
            elif key == "escape":
                return None
            elif key == "ctrl_c":
                raise KeyboardInterrupt
            elif key.isdigit() and 1 <= int(key) <= len(options):
                return options[int(key) - 1][0]

    def _render_selector(self, title, options, selected_index):
        self._clear()
        width = max(
            52,
            min(shutil.get_terminal_size((80, 24)).columns, 88),
        )
        inner_width = width - 4

        self._write(self._paint("❯ evolutionary-agent-testbed", "orange"))
        self._write("")
        self._write(self._border("╭", "─", "╮", width))
        self._write(self._row(f" {title}", inner_width))
        self._write(self._row(" ↑/↓ navegar · Enter selecionar · Esc sair", inner_width))
        self._write(self._border("├", "─", "┤", width))

        for index, (_, label) in enumerate(options):
            marker = "❯" if index == selected_index else " "
            line = f" {marker} {label}"
            visible = line[:inner_width]
            color = "orange" if index == selected_index else "muted"
            self._write(self._row(self._paint(visible, color), inner_width, len(visible)))

        self._write(self._border("╰", "─", "╯", width))
        self.output.flush()

    def _show_run_header(self):
        self._clear()
        self._write(self._paint("❯ evolutionary-agent-testbed", "orange"))
        self._write("")
        self._write(
            f"Agente: {self.config.EXPERIMENT_TYPE.upper()}  ·  "
            f"Modo: {self.config.EXPERIMENT_MODE}"
        )
        self._write(
            f"Otimizador: {self.config.EXPERIMENT_OPTIMIZER}  ·  "
            f"Ambiente: {self.config.ENVIRONMENT_TYPE}"
        )
        self._write("Iniciando…  (Ctrl+C cancela o experimento atual)")
        self._write("")
        self.output.flush()

    def _read_key(self):
        if not getattr(sys.stdin, "isatty", lambda: False)():
            try:
                value = input("Selecione [1-9] ou pressione Enter: ").strip()
            except EOFError:
                return "escape"
            return value if value else "enter"

        if os.name == "nt":
            import msvcrt

            key = msvcrt.getwch()
            if key in ("\x00", "\xe0"):
                return {"H": "up", "P": "down"}.get(msvcrt.getwch(), "")
        else:
            import select
            import termios
            import tty

            descriptor = sys.stdin.fileno()
            previous_mode = termios.tcgetattr(descriptor)
            try:
                tty.setraw(descriptor)
                key = os.read(descriptor, 1).decode("utf-8", errors="ignore")
                if key == "\x1b":
                    if select.select([descriptor], [], [], 0.08)[0]:
                        prefix = os.read(descriptor, 1).decode(
                            "utf-8",
                            errors="ignore",
                        )
                        if prefix in ("[", "O") and select.select(
                            [descriptor], [], [], 0.08
                        )[0]:
                            return {"A": "up", "B": "down"}.get(
                                os.read(descriptor, 1).decode(
                                    "utf-8",
                                    errors="ignore",
                                ),
                                "",
                            )
                    return "escape"
            finally:
                termios.tcsetattr(
                    descriptor,
                    termios.TCSADRAIN,
                    previous_mode,
                )

        if key in ("\r", "\n"):
            return "enter"
        if key == "\x03":
            return "ctrl_c"
        if key == "\x1b":
            return "escape"
        return key.lower()

    def _clear(self):
        if getattr(self.output, "isatty", lambda: False)():
            self._write("\x1b[2J\x1b[H")

    def _write(self, value):
        self.output.write(f"{value}\n")

    def _paint(self, text, color):
        if not self.color_enabled:
            return text
        codes = {"orange": "38;5;208", "muted": "38;5;245"}
        return f"\x1b[{codes[color]}m{text}\x1b[0m"

    @staticmethod
    def _border(left, fill, right, width):
        return left + fill * (width - 2) + right

    @staticmethod
    def _row(text, inner_width, visible_length=None):
        if visible_length is None:
            visible_length = len(text)
        padding = " " * max(0, inner_width - visible_length)
        return f"│ {text}{padding} │"
