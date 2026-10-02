from dataclasses import dataclass

@dataclass
class AggregatedGenerationMetrics:
    """
    Stores aggregated statistics for a single generation.
    """
    generation: int
    mean_success_rate: float
    std_success_rate: float
    margin_of_error: float
    lower_bound: float
    upper_bound: float