from simulation import SimulationResult

class FitnessEvaluator:

    def evaluate(self, result: SimulationResult) -> float:
        return result.total_reward

