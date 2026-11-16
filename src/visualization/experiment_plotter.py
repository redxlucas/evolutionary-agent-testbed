import math
import statistics

from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

import config
from evolution import FitnessStrategy
from experiment.aggregated_generation_metrics import (
    AggregatedGenerationMetrics,
)
from experiment.experiment_analyzer import ExperimentAnalyzer
from experiment.experiment_types import ExperimentType
from visualization import Plotter


class ExperimentPlotter(Plotter):
    """
    Plots aggregated metrics from multiple experiment runs.
    """

    def __init__(
        self,
        metrics: list[AggregatedGenerationMetrics],
        experiment_type: ExperimentType
    ):
        super().__init__(figsize=(14, 7))
        self.metrics = metrics
        self.experiment_type = experiment_type

    @staticmethod
    def _uses_iterations(title: str | None) -> bool:
        if not title:
            return False
        normalized_title = title.casefold()
        return (
            "cross_entropy" in normalized_title
            or "cross-entropy" in normalized_title
            or "entropia cruzada" in normalized_title
        )

    @staticmethod
    def _format_experiment_title(title: str) -> str:
        parts = [part.strip() for part in title.split("|")]
        if len(parts) < 4 or not parts[1].startswith("ambiente="):
            return title

        environments = {
            "frozen_lake": "Frozen Lake",
            "minigrid": "MiniGrid",
        }
        agents = {
            "MOVEMENT": "Agente baseado em movimentos",
            "ADALINE": "ADALINE",
            "MLP": "MLP",
        }
        optimizers = {
            "genetic": "Algoritmo genético",
            "cross_entropy": "Método da entropia cruzada",
            "differential_evolution": "Evolução Diferencial",
        }
        environment = parts[1].removeprefix("ambiente=")
        agent_description = " · ".join(parts[2:-1])
        title_prefix = parts[0]
        if ExperimentPlotter._uses_iterations(parts[-1]):
            title_prefix = title_prefix.replace("geração", "iteração")
        return (
            f"{title_prefix}\n"
            f"{environments.get(environment, environment)} · "
            f"{agents.get(agent_description, agent_description)} · "
            f"{optimizers.get(parts[-1], parts[-1])}"
        )

    @staticmethod
    def _nice_tick_step(value: float) -> float:
        """Choose a readable axis interval close to one tenth of the range."""
        if value <= 1:
            return 1
        magnitude = 10 ** math.floor(math.log10(value))
        normalized = value / magnitude
        if normalized <= 1:
            multiplier = 1
        elif normalized <= 2:
            multiplier = 2
        elif normalized <= 5:
            multiplier = 5
        else:
            multiplier = 10
        return multiplier * magnitude

    def plot_experiment(self, title: str | None = None):
        generations = [
            metric.generation
            for metric in self.metrics
        ]

        success_rate = [
            metric.mean_success_rate * 100
            for metric in self.metrics
        ]

        std_success_rate = [
            metric.std_success_rate * 100
            for metric in self.metrics
        ]

        margin_of_error = [
            metric.margin_of_error * 100
            for metric in self.metrics
        ]

        lower_bound = [
            metric.lower_bound * 100
            for metric in self.metrics
        ]

        upper_bound = [
            metric.upper_bound * 100
            for metric in self.metrics
        ]

        success_line, = self.axes.plot(
            generations,
            success_rate,
            label="Taxa de sucesso",
            linewidth=1.8,
            zorder=3,
        )

        self.axes.fill_between(
            generations,
            lower_bound,
            upper_bound,
            color=success_line.get_color(),
            alpha=0.14,
            linewidth=0,
            label="IC 95%",
            zorder=2,
        )

        mean_std = statistics.mean(std_success_rate)
        mean_margin = statistics.mean(margin_of_error)

        cycle_label = "Iteração" if self._uses_iterations(title) else "Geração"
        display_title = title or (
            "Evolução da Taxa de Sucesso ao Longo das Gerações"
            f" - Agente {self.experiment_type.value}"
        )
        self.configure(
            title=self._format_experiment_title(display_title),
            xlabel=cycle_label,
            ylabel="Taxa de sucesso (%)",
        )
        self.axes.set_xlim(generations[0], generations[-1])
        self.axes.set_ylim(
            0,
            min(110, max(10, math.ceil(max(upper_bound) * 1.08 / 10) * 10)),
        )
        generation_span = max(1, generations[-1] - generations[0])
        generation_step = self._nice_tick_step(generation_span / 10)
        self.axes.xaxis.set_major_locator(MultipleLocator(generation_step))
        self.axes.xaxis.set_minor_locator(AutoMinorLocator(2))
        self.axes.yaxis.set_major_locator(MultipleLocator(10))
        self.axes.yaxis.set_minor_locator(AutoMinorLocator(2))
        self.axes.grid(True, which="major", alpha=0.3)
        self.axes.grid(
            True,
            which="minor",
            alpha=0.12,
            linestyle=":",
        )
        self.axes.set_axisbelow(True)
        self.axes.set_title(self.axes.get_title(), fontsize=17, pad=16)
        self.axes.set_xlabel(cycle_label, fontsize=14)
        self.axes.set_ylabel("Taxa de sucesso (%)", fontsize=14)
        self.axes.tick_params(axis="both", labelsize=12)

        legend = self.axes.legend(loc="lower right")

        self.axes.text(
            0.98,
            0.13,
            (
                f"Desvio padrão médio: {mean_std:.2f} p.p.\n"
                f"Margem de erro média: {mean_margin:.2f} p.p."
            ),
            transform=self.axes.transAxes,
            horizontalalignment="right",
            verticalalignment="bottom",
            fontsize=11,
            bbox=dict(
                boxstyle="round",
                facecolor="white",
                alpha=0.8,
            ),
            zorder=legend.get_zorder() + 1,
        )

    def plot_fitness_comparison(self):
        strategy_labels = {
            FitnessStrategy.REWARD: "Recompensa",
            FitnessStrategy.FINAL_DISTANCE: "Distância Manhattan",
            FitnessStrategy.PROGRESS: "Progresso (passos)",
        }

        stats_text = []
        confidence_upper_bounds = []

        for strategy, experiments in self.metrics.items():
            analyzer = ExperimentAnalyzer(experiments)
            metrics = analyzer.analyze()

            generations = [
                metric.generation
                for metric in metrics
            ]

            mean_success_rate = [
                metric.mean_success_rate
                for metric in metrics
            ]

            lower_bound = [
                metric.lower_bound
                for metric in metrics
            ]

            upper_bound = [
                metric.upper_bound
                for metric in metrics
            ]
            confidence_upper_bounds.extend(
                value * 100 for value in upper_bound
            )

            self.plot_line(
                generations,
                [value * 100 for value in mean_success_rate],
                label=strategy_labels[strategy],
            )

            self.axes.fill_between(
                generations,
                [value * 100 for value in lower_bound],
                [value * 100 for value in upper_bound],
                alpha=0.15,
            )

            mean_std = statistics.mean(
                metric.std_success_rate
                for metric in metrics
            ) * 100

            mean_margin = statistics.mean(
                metric.margin_of_error
                for metric in metrics
            ) * 100

            stats_text.append(
                f"{strategy_labels[strategy]}\n"
                f"  Desvio padrão médio: {mean_std:.4f} p.p.\n"
                f"  Margem de erro média: {mean_margin:.4f} p.p."
            )

        self.axes.text(
            0.02,
            0.97,
            "Estatísticas (IC 95%)\n\n"
            + "\n\n".join(stats_text),
            transform=self.axes.transAxes,
            verticalalignment="top",
            bbox=dict(
                boxstyle="round",
                facecolor="white",
                alpha=0.8,
            ),
        )

        cycle_label = (
            "Iteração"
            if config.EXPERIMENT_OPTIMIZER == "cross_entropy"
            else "Geração"
        )
        self.configure(
            title=f"Comparação da Taxa de Sucesso por Estratégia de Fitness - Agente {self.experiment_type.value}",
            xlabel=cycle_label,
            ylabel="Taxa de sucesso (%)",
        )

        y_max = min(
            110,
            max(
                10,
                math.ceil(max(confidence_upper_bounds) * 1.08 / 10) * 10,
            ),
        )
        self.axes.set_ylim(0, y_max)
        self.axes.yaxis.set_major_locator(MultipleLocator(10))

    def plot_configuration_comparison(
        self,
        *,
        saved_metrics=None,
        hidden_sizes=None,
        optimizer=None,
    ):
        """Compare activation functions in one panel per hidden-layer size."""
        activations = (
            ("MLP + ReLU", "ReLU"),
            (
                "MLP + Leaky ReLU (alpha = 0.25)",
                "Leaky ReLU (alpha = 0.25)",
            ),
            ("MLP + PReLU + Alpha", "PReLU (alpha evolutivo)"),
        )
        hidden_sizes = hidden_sizes or config.MLP_HIDDEN_SIZES
        optimizer = optimizer or config.EXPERIMENT_OPTIMIZER

        plt.close(self.figure)
        self.figure, axes = plt.subplots(
            1,
            len(hidden_sizes),
            figsize=(14, 7),
            sharey=True,
            squeeze=False,
        )
        axes = axes[0]
        self.axes = axes[0]
        self.figure.layout_rect = (0.05, 0.40, 0.98, 0.96)

        stats_rows = []
        confidence_bounds = []
        cycle_label = (
            "Iteração"
            if optimizer == "cross_entropy"
            else "Geração"
        )
        if saved_metrics is None:
            last_generation = config.GENERATIONS - 1
        else:
            last_generation = max(
                metric.generation
                for metrics in saved_metrics.values()
                for metric in metrics
            )
        for ax, hidden_size in zip(axes, hidden_sizes):
            ax.set_title(f"{hidden_size} neurônios ocultos")
            ax.set_xlabel(cycle_label)
            ax.set_xlim(-1, last_generation + 1)
            generation_ticks = [
                tick
                for tick in ax.get_xticks()
                if 0 <= tick < last_generation
            ]
            ax.set_xticks(sorted(set(generation_ticks + [0, last_generation])))
            ax.grid(True)

            for config_name, display_name in activations:
                result_key = f"{config_name} | {hidden_size} neurônios"
                if saved_metrics is None:
                    metrics = ExperimentAnalyzer(
                        self.metrics[result_key]
                    ).analyze()
                else:
                    metrics = saved_metrics[result_key]
                generations = [metric.generation for metric in metrics]
                mean_success_rate = [
                    metric.mean_success_rate * 100
                    for metric in metrics
                ]
                lower_bound = [metric.lower_bound * 100 for metric in metrics]
                upper_bound = [metric.upper_bound * 100 for metric in metrics]
                confidence_bounds.extend(lower_bound)
                confidence_bounds.extend(upper_bound)

                ax.plot(
                    generations,
                    mean_success_rate,
                    label=display_name,
                )
                ax.fill_between(
                    generations,
                    lower_bound,
                    upper_bound,
                    alpha=0.12,
                )

                mean_std = statistics.mean(
                    metric.std_success_rate
                    for metric in metrics
                ) * 100
                mean_margin = statistics.mean(
                    metric.margin_of_error
                    for metric in metrics
                ) * 100
                stats_rows.append([
                    str(hidden_size),
                    display_name,
                    f"{mean_std:.4f}",
                    f"{mean_margin:.4f}",
                ])

        if confidence_bounds:
            data_max = max(confidence_bounds)
            y_max = min(110, max(10, math.ceil(data_max * 1.08 / 10) * 10))
            for ax in axes:
                ax.set_ylim(0, y_max)
                ax.yaxis.set_major_locator(MultipleLocator(10))

        axes[0].set_ylabel("Taxa de sucesso (%)")

        for ax in axes:
            ax.legend(loc="lower right", fontsize=8, framealpha=0.9)

        table_axes = self.figure.add_axes([0.03, 0.02, 0.94, 0.27])
        table_axes.axis("off")
        table_axes.set_in_layout(False)
        table = table_axes.table(
            cellText=stats_rows,
            colLabels=[
                "Neurônios ocultos",
                "Ativação",
                "Desvio padrão médio (p.p.)",
                "Margem de erro média, IC 95% (p.p.)",
            ],
            colWidths=[0.17, 0.35, 0.22, 0.26],
            cellLoc="center",
            bbox=[0, 0, 1, 1],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
