import statistics

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