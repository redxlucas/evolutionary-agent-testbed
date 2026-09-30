from enum import Enum

class ExperimentMode(Enum):
    SINGLE = "single"
    FITNESS_COMPARISON = "fitness_comparison"


class ExperimentType(Enum):
    ADALINE = "adaline"
    MOVEMENT = "movement"