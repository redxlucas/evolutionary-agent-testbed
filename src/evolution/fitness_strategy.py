from enum import Enum

class FitnessStrategy(Enum):
    REWARD = "reward"
    FINAL_DISTANCE = "final_distance"
    PROGRESS = "progress"