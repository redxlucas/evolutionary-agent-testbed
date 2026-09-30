from dataclasses import dataclass

@dataclass
class ExperimentConfig:
    """
    Defines the components and evolution settings of an experiment.
    """
    genome_factory: callable
    agent_creator: callable
    evolve: bool