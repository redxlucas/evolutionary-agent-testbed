from dataclasses import dataclass

from .experiment_types import ActivationFunction


@dataclass
class ExperimentConfig:
    """
    Defines the components and evolution settings of an experiment.
    """
    name: str
    genome_factory: callable
    agent_creator: callable
    evolve: bool
    activation: ActivationFunction = ActivationFunction.NONE
    evolve_alpha: bool = False
    hidden_size: int | None = None
