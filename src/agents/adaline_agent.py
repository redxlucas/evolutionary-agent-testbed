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

    def reset(self):
        pass

    def act(self, observation):
        """
        Selects an action based on the given observation.
        """
        inputs = np.delete(observation.flatten(), observation.size // 2)
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
