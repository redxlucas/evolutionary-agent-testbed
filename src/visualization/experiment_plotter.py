import statistics

from matplotlib import pyplot as plt

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
        super().__init__()
        self.metrics = metrics
        self.experiment_type = experiment_type

    def plot_experiment(self):
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

        self.plot_line(
            generations,
            success_rate,
            label="Taxa de sucesso",
        )

        self.axes.fill_between(
            generations,
            lower_bound,
            upper_bound,
            alpha=0.2,
            label="IC 95%",
        )

        mean_std = statistics.mean(std_success_rate)
        mean_margin = statistics.mean(margin_of_error)

        self.configure(
            title=(
                "Evolução da Taxa de Sucesso ao Longo das Gerações"
                f"- Agente {self.experiment_type.value}"
            ),
            xlabel="Geração",
            ylabel="Taxa de sucesso (%)",
        )

        self.axes.text(
            0.02,
            0.97,
            (
                f"Desvio padrão médio: {mean_std:.2f} p.p.\n"
                f"Margem de erro média: {mean_margin:.2f} p.p.\n"
                f"IC: 95%"
            ),
            transform=self.axes.transAxes,
            verticalalignment="top",
            bbox=dict(
                boxstyle="round",
                facecolor="white",
                alpha=0.8,
            ),
        )

    def plot_fitness_comparison(self):
        strategy_labels = {
            FitnessStrategy.REWARD: "Recompensa",
            FitnessStrategy.FINAL_DISTANCE: "Distância Manhattan",
            FitnessStrategy.PROGRESS: "Progresso (passos)",
        }

        stats_text = []

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

        self.configure(
            title=f"Comparação da Taxa de Sucesso por Estratégia de Fitness - Agente {self.experiment_type.value}",
            xlabel="Geração",
            ylabel="Taxa de sucesso (%)",
        )

        self.axes.set_ylim(0, 100)

    def plot_configuration_comparison(self):
        """Compare activation functions in one panel per hidden-layer size."""
        activations = (
            ("MLP + ReLU", "ReLU"),
            (
                "MLP + Leaky ReLU (alpha = 0.25)",
                "Leaky ReLU (alpha = 0.25)",
            ),
            ("MLP + PReLU + Alpha", "PReLU (alpha evolutivo)"),
        )
        hidden_sizes = config.MLP_HIDDEN_SIZES

        plt.close(self.figure)
        self.figure, axes = plt.subplots(
            1,
            len(hidden_sizes),
            figsize=(18, 10),
            sharey=True,
            squeeze=False,
        )
        axes = axes[0]
        self.axes = axes[0]
        self.figure.layout_rect = (0.05, 0.36, 0.98, 0.92)
        self.figure.suptitle(
            "Comparação de ativações com fitness por distância Manhattan",
            fontsize=14,
        )

        stats_rows = []
        confidence_bounds = []
        last_generation = config.GENERATIONS - 1
        for ax, hidden_size in zip(axes, hidden_sizes):
            ax.set_title(f"{hidden_size} neurônios ocultos")
            ax.set_xlabel("Geração")
            ax.set_xlim(0, last_generation)
            generation_ticks = [
                tick
                for tick in ax.get_xticks()
                if 0 <= tick < last_generation
            ]
            ax.set_xticks(sorted(set(generation_ticks + [0, last_generation])))
            ax.grid(True)

            for config_name, display_name in activations:
                result_key = f"{config_name} | {hidden_size} neurônios"
                metrics = ExperimentAnalyzer(
                    self.metrics[result_key]
                ).analyze()
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
            data_min = min(confidence_bounds)
            data_max = max(confidence_bounds)
            padding = max((data_max - data_min) * 0.08, 1.0)
            y_min = max(0, data_min - padding)
            y_max = min(100, data_max + padding)
            for ax in axes:
                ax.set_ylim(y_min, y_max)

        axes[0].set_ylabel("Taxa de sucesso (%)")

        for ax in axes:
            ax.legend(loc="lower right", fontsize=8, framealpha=0.9)

        table_axes = self.figure.add_axes([0.03, 0.02, 0.94, 0.29])
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
