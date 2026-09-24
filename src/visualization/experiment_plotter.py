import numpy as np

from experiment import AggregatedGenerationMetrics
from visualization import Plotter

class ExperimentPlotter(Plotter):

    def __init__(self, metrics: list[AggregatedGenerationMetrics]):
        super().__init__()
        self.metrics = metrics

    def plot_experiment(self):
        generations = [metric.generation for metric in self.metrics]
        mean_fitness = [metric.mean_fitness for metric in self.metrics]
        std_fitness = [metric.std_fitness for metric in self.metrics]
        margin_of_error = [metric.margin_of_error for metric in self.metrics]
        lower_bound = [metric.lower_bound for metric in self.metrics]
        upper_bound = [metric.upper_bound for metric in self.metrics]

        self.plot_line(generations, mean_fitness, label="Média Fitness")

        self.configure(
            title="Evolução do Fitness Médio ao Longo das Gerações (30 experimentos)",
            xlabel="Geração",
            ylabel="Fitness Médio"
        )


        self.axes.fill_between(
            generations,
            lower_bound,
            upper_bound,
            alpha=0.2,
            label="IC 95%"
        )

        mean_std = np.mean(std_fitness)
        mean_margin = np.mean(margin_of_error)

        self.axes.text(
            0.02,
            0.97,
            (
                f"Desvio padrão médio: {mean_std:.4f}\n"
                f"Margem de erro médio: {mean_margin:.4f}\n"
                f"IC: 95%"
            ),
            transform=self.axes.transAxes,
            verticalalignment="top",
            bbox=dict(
                boxstyle="round",
                facecolor="white",
                alpha=0.8
            )
        )
