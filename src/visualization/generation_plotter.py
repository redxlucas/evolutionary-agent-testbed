import matplotlib.pyplot as plt

from evolution import GenerationMetrics

class GenerationPlotter:

    def __init__(self, metrics: list[GenerationMetrics]):
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