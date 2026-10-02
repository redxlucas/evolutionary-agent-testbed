from enum import Enum

class FitnessStrategy(Enum):
    """
    Defines the fitness evaluation strategies available in the experiments.
    """
    REWARD = "reward"
    FINAL_DISTANCE = "final_distance"
    PROGRESS = "progress"
    COMBINED = "combined"
