import numpy as np
from scipy import stats

from experiment.aggregated_generation_metrics import AggregatedGenerationMetrics

class ExperimentAnalyzer:
    """
    Analyzes the results of multiple experiment runs.
    """

    def __init__(self, results):
        self.results = results

    def analyze(self) -> list[AggregatedGenerationMetrics]:
        """
        Aggregates the fitness metrics for each generation.
        """

        if not self.results:
            raise ValueError("No experiment results were provided.")

        generations = len(self.results[0])

        aggregated_metrics = []

        for generation in range(generations):
            values = [
                run[generation].success_rate
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
                    mean_success_rate=mean,
                    std_success_rate=std,
                    margin_of_error=margin_of_error,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                )
            )

        return aggregated_metrics

    def _calculate_statistics(self, values):
        """
        Calculates the mean, standard deviation, margin of error,
        and 95% confidence interval for the provided values.
        """

        values = np.asarray(values)

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