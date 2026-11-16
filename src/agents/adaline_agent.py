import logging

import numpy as np

from agents.genome import AdalineGenome

logger = logging.getLogger(__name__)

class AdalineAgent:
    """
    Agent that selects actions using an ADALINE model.
    """

    def __init__(self, weights: np.ndarray, bias: np.ndarray):
        self.weights = weights
        self.bias = bias
        self.trace_context = ""
        self._input_indices_cache = {}

    def reset(self):
        pass

    def act(self, observation, agent_observation_index=None):
        """
        Selects an action based on the given observation.
        """
        if agent_observation_index is None:
            agent_observation_index = observation.size // 2
        cache_key = (observation.size, agent_observation_index)
        input_indices = self._input_indices_cache.get(cache_key)
        if input_indices is None:
            if not 0 <= agent_observation_index < observation.size:
                raise ValueError(
                    "Agent observation index is outside the observation."
                )
            input_indices = np.concatenate((
                np.arange(agent_observation_index),
                np.arange(agent_observation_index + 1, observation.size),
            ))
            self._input_indices_cache[cache_key] = input_indices

        inputs = observation.ravel()[input_indices]
        outputs = self.predict(inputs)
        action = int(np.argmax(outputs))

        logger.debug(
            "ADALINE decision | %s | observation=%s | inputs=%s | "
            "outputs=%s | action=%d",
            self.trace_context,
            observation.tolist(),
            inputs.tolist(),
            outputs.tolist(),
            action,
        )

        return action

    def predict(self, inputs):
        """
        Computes the output of the ADALINE model.
        """
        return inputs @ self.weights + self.bias

    @staticmethod
    def create(genome: AdalineGenome):
        return AdalineAgent(
            weights=genome.get_weights(),
            bias=genome.get_bias()
        )
