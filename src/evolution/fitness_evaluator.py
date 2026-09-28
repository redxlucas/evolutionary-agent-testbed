import config
from evolution import FitnessStrategy
from simulation import SimulationResult
from utils import manhattan_distance
class FitnessEvaluator:

    def __init__(self, strategy: FitnessStrategy):
        self.strategy = strategy

    def evaluate(self, result: SimulationResult) -> float:

        if self.strategy == FitnessStrategy.REWARD:
            return self._calculate_reward(result)

        if self.strategy == FitnessStrategy.FINAL_DISTANCE:
            return self._calculate_distance_reward(
                result.final_position,
                result.goal_position
            )

        if self.strategy == FitnessStrategy.PROGRESS:
            return self._calculate_steps_reward(
                steps=result.steps,
                max_steps=config.MAX_STEPS,
                success=result.total_reward >= 1
            )

        raise ValueError(
            f"Unknown fitness strategy: {self.strategy}"
        )
    
    def _calculate_reward(self, result: SimulationResult) -> float:
        return result.total_reward

    @staticmethod
    def _calculate_distance_reward(pos_a, pos_b):
        distance = manhattan_distance(pos_a, pos_b)

        return 1 / (1 + distance) # função inversa deslocada

    @staticmethod
    def _calculate_steps_reward(steps: int, max_steps: int, success: bool) -> float:

        if not success:
            return 0.0

        return 1 - (steps / max_steps)
    
    # def _evaluate_final_distance(self, result: SimulationResult) -> float:
    #     distance_reward = self._calculate_distance_reward(
    #         result.final_position, 
    #         result.goal_position
    #     )

    #     return result.total_reward + distance_reward
    
    # def _evaluate_progress(self, result: SimulationResult) -> float:
    #     steps_reward = self._calculate_steps_reward(
    #         steps=result.steps,
    #         max_steps=config.GENOME_LENGTH,
    #         success=result.total_reward
    #     )
    #     return result.total_reward + steps_reward


