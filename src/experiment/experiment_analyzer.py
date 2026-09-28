import numpy as np
from scipy import stats

from experiment.aggregated_generation_metrics import AggregatedGenerationMetrics

class ExperimentAnalyzer:

    def __init__(self, results):
        self.results = results

    def analyze(self):
        generations = len(self.results[0])

        aggregated_metrics = []

        for generation in range(generations):
            values = [
                run[generation].average_fitness
                for run in self.results
            ]

            (
                mean,
                std,
                margin_of_error,
                lower_bound,
                upper_bound,
            ) = self._calculate_statistics(values)

            aggregated_metrics.append(
                AggregatedGenerationMetrics(
                    generation=generation,
                    mean_fitness=mean,
                    std_fitness=std,
                    margin_of_error=margin_of_error,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                )
            )

        return aggregated_metrics

    def _calculate_statistics(self, values):
        values = np.array(values)

        mean = np.mean(values)
        std = np.std(values, ddof=1)

        standard_error = std / np.sqrt(len(values))

        t_critical = stats.t.ppf(
            0.975,
            df=len(values) - 1
        )

        margin_of_error = t_critical * standard_error

        return (
            mean,
            std,
            margin_of_error,
            mean - margin_of_error,
            mean + margin_of_error,
        )