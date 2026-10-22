from enum import Enum

class ExperimentMode(Enum):
    SINGLE = "single"
    FITNESS_COMPARISON = "fitness_comparison"
    CONFIGURATION_COMPARISON = "configuration_comparison"


class ExperimentType(Enum):
    ADALINE = "adaline"
    MOVEMENT = "movement"
    MLP = "mlp"

class ExperimentOptimizer(Enum):
    GENETIC = "genetic"
    CROSS_ENTROPY = "cross_entropy"

class ActivationFunction(Enum):
    NONE = "none"
    RELU = "relu"
    PRELU = "prelu"
