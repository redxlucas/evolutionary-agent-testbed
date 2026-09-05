import config
from simulation import SimulationResult
from utils import manhattan_distance

class FitnessEvaluator:
    
    @staticmethod
    def _calculate_distance_reward(pos_a, pos_b):
        distance = manhattan_distance(pos_a, pos_b)

        return 1 / (1 + distance) # função inversa deslocada

    @staticmethod
    def _calculate_steps_reward(steps: int, max_steps: int, success: bool) -> float:

        if success < 1:
            return 0.0

        return 1 - (steps / max_steps)



    def evaluate(self, result: SimulationResult) -> float:
        distance_reward = self._calculate_distance_reward(
            result.final_position, 
            result.goal_position
        )
        
        steps_reward = self._calculate_steps_reward(
            steps=result.steps,
            max_steps=config.GENOME_LENGTH,
            success=result.total_reward
        )

        return result.total_reward + distance_reward + steps_reward


