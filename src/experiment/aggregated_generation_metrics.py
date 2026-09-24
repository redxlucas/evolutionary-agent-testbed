from dataclasses import dataclass

@dataclass
class AggregatedGenerationMetrics:
    generation: int
    mean_fitness: float
    std_fitness: float
    margin_of_error: float
    lower_bound: float
    upper_bound: float