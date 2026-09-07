import matplotlib.pyplot as plt

from evolution import GenerationMetrics

class GenerationPlotter:

    def __init__(
        self,
        metrics: list[GenerationMetrics] | None = None
    ):
        self.metrics = metrics

    def plot_fitness(self):
        generations = [
            metric.generation
            for metric in self.metrics
        ]

        best_fitness = [
            metric.best_fitness
            for metric in self.metrics
        ]

        average_fitness = [
            metric.average_fitness
            for metric in self.metrics
        ]

        worst_fitness = [
            metric.worst_fitness
            for metric in self.metrics
        ]

        plt.figure(figsize=(10, 5))

        plt.plot(
            generations,
            best_fitness,
            label="Melhor fitness"
        )

        plt.plot(
            generations,
            average_fitness,
            label="Fitness médio"
        )

        plt.plot(
            generations,
            worst_fitness,
            label="Pior fitness"
        )

        plt.xlabel("Geração")
        plt.ylabel("Fitness")
        plt.title("Evolução do fitness por geração")

        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.show()

    def plot_fitness_comparison(
        self,
        without_elitism_metrics,
        with_elitism_metrics
    ):

        without_elitism_generations = [
            metric.generation
            for metric in without_elitism_metrics
        ]

        without_elitism_best_fitness = [
            metric.best_fitness
            for metric in without_elitism_metrics
        ]

        with_elitism_generations = [
            metric.generation
            for metric in with_elitism_metrics
        ]

        with_elitism_best_fitness = [
            metric.best_fitness
            for metric in with_elitism_metrics
        ]

        plt.figure()

        plt.plot(
            without_elitism_generations,
            without_elitism_best_fitness,
            label="Sem elitismo"
        )

        plt.plot(
            with_elitism_generations,
            with_elitism_best_fitness,
            label="Com elitismo (3)"
        )

        plt.xlabel("Geração")
        plt.ylabel("Melhor fitness")
        plt.title("Comparação do fitness com e sem elitismo")

        plt.legend()

        plt.show()