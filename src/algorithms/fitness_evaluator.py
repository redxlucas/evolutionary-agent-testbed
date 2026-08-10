from simulation import SimulationResult
from utils import manhattan_distance

class FitnessEvaluator:

    @staticmethod
    def _calculate_distance_reward(pos_a, pos_b):
        distance = manhattan_distance(pos_a, pos_b)

        return 1 / (1 + distance) # função inversa deslocada

    def evaluate(self, result: SimulationResult) -> float:
        distance_reward = self._calculate_distance_reward(
            result.final_position, 
            result.goal_position
        )

        return result.total_reward + distance_reward


