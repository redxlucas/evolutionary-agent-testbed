from dataclasses import dataclass

@dataclass
class ExperimentConfig:
    genome_factory: callable
    agent_creator: callable
    evolve: bool