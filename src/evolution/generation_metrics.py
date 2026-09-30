from dataclasses import dataclass

@dataclass
class GenerationMetrics:
    """
    Stores the fitness metrics obtained for a single generation.
    """
    generation: int
    best_fitness: float
    average_fitness: float
    worst_fitness: float
    success_rate: float