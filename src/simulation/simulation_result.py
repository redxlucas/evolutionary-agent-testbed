from dataclasses import dataclass

@dataclass
class SimulationResult:
    total_reward: float
    steps: int
    terminated: bool
    truncated: bool