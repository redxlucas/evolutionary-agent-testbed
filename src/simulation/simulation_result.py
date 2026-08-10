from dataclasses import dataclass

@dataclass
class SimulationResult:
    total_reward: float
    steps: int
    terminated: bool
    truncated: bool
    final_position: tuple[int, int]
    goal_position: tuple[int, int]