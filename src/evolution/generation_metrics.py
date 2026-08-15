from dataclasses import dataclass

@dataclass
class GenerationMetrics:
    generation: int
    best_fitness: float
    average_fitness: float
    worst_fitness: float
    success_rate: float