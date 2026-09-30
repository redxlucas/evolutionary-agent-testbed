import statistics

from evolution import FitnessStrategy
from experiment.aggregated_generation_metrics import (
    AggregatedGenerationMetrics,
)
from visualization import Plotter


class ExperimentPlotter(Plotter):
    """
    Plots aggregated metrics from multiple experiment runs.
    """

    def __init__(
        self,
        metrics: list[AggregatedGenerationMetrics],
    ):
        super().__init__()
        self.metrics = metrics

    def plot_experiment(self):
        generations = [
            metric.generation
            for metric in self.metrics
        ]

        mean_fitness = [
            metric.mean_fitness
            for metric in self.metrics
        ]

        std_fitness = [
            metric.std_fitness
            for metric in self.metrics
        ]

        margin_of_error = [
            metric.margin_of_error
            for metric in self.metrics
        ]

        lower_bound = [
            metric.lower_bound
            for metric in self.metrics
        ]

        upper_bound = [
            metric.upper_bound
            for metric in self.metrics
        ]

        self.plot_line(
            generations,
            mean_fitness,
            label="Média Fitness",
        )

        self.axes.fill_between(
            generations,
            lower_bound,
            upper_bound,
            alpha=0.2,
            label="IC 95%",
        )

        mean_std = statistics.mean(std_fitness)
        mean_margin = statistics.mean(margin_of_error)

        self.configure(
            title=(
                "Evolução do Fitness Médio "
                "ao Longo das Gerações"
            ),
            xlabel="Geração",
            ylabel="Fitness Médio",
        )

        self.axes.text(
            0.02,
            0.97,
            (
                f"Desvio padrão médio: {mean_std:.4f}\n"
                f"Margem de erro média: {mean_margin:.4f}\n"
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
            generations = [
                metric.generation
                for metric in experiments[0]
            ]

            mean_fitness = []
            std_fitness = []
            margin_of_error = []
            lower_bound = []
            upper_bound = []

            for generation_index in range(len(generations)):
                values = [
                    run[generation_index].average_fitness
                    for run in experiments
                ]

                mean = statistics.mean(values)

                std = (
                    statistics.stdev(values)
                    if len(values) > 1
                    else 0.0
                )

                n = len(values)

                standard_error = (
                    std / (n ** 0.5)
                    if n > 1
                    else 0.0
                )

                # Approximation for a 95% confidence interval.
                margin = 1.96 * standard_error

                mean_fitness.append(mean)
                std_fitness.append(std)
                margin_of_error.append(margin)

                lower_bound.append(mean - margin)
                upper_bound.append(mean + margin)

            self.plot_line(
                generations,
                mean_fitness,
                label=strategy_labels[strategy],
            )

            self.axes.fill_between(
                generations,
                lower_bound,
                upper_bound,
                alpha=0.15,
            )

            mean_std = statistics.mean(std_fitness)
            mean_margin = statistics.mean(margin_of_error)

            stats_text.append(
                f"{strategy_labels[strategy]}\n"
                f"  Desvio padrão médio: {mean_std:.4f}\n"
                f"  Margem de erro média: {mean_margin:.4f}"
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
            title="Comparação das Estratégias de Fitness",
            xlabel="Geração",
            ylabel="Fitness Médio",
        )