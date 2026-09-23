from evolution import GenerationMetrics
from visualization.plotter import Plotter

class GenerationPlotter(Plotter):

    def __init__(self, metrics: list[GenerationMetrics]):
        super().__init__()
        self.metrics = metrics

    def plot_fitness(self):
        generations = [metric.generation for metric in self.metrics]
        best = [metric.best_fitness for metric in self.metrics]
        average = [metric.average_fitness for metric in self.metrics]
        worst = [metric.worst_fitness for metric in self.metrics]

        self.plot_line(generations, best, label="Melhor Fitness")
        self.plot_line(generations, average, label="Média do Fitness")
        self.plot_line(generations, worst, label="Pior Fitness")

        self.configure(
            title="Evolução do Fitness com Adaline (sem elitismo, seleção, crossover e mutação)",
            xlabel="Geração",
            ylabel="Fitness"
        )