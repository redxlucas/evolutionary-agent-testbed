import logging

import numpy as np

from agents.genome.mlp_genome import MLPGenome

logger = logging.getLogger(__name__)

class MLPAgent:

    def __init__(
        self,
        genome: MLPGenome,
        activation="prelu",
        evolve_alpha: bool = False,
    ):
        self.genome = genome
        self.activation = activation
        self.evolve_alpha = evolve_alpha
        self.trace_context = ""

    def reset(self):
        pass

    def act(self, observation):
        inputs = np.delete(
            observation.flatten(),
            observation.size // 2
        )

        outputs = self.predict(inputs)
        action = int(np.argmax(outputs))

        logger.debug(
            "MLP decision | %s | observation=%s | inputs=%s | "
            "outputs=%s | action=%d",
            self.trace_context,
            observation.tolist(),
            inputs.tolist(),
            outputs.tolist(),
            action,
        )

        return action

    def predict(self, inputs):
        input_weights = self.genome.get_input_weights()
        hidden_bias = self.genome.get_hidden_bias()
        output_weights = self.genome.get_output_weights()
        output_bias = self.genome.get_output_bias()
        activation = getattr(self.activation, "value", self.activation)
        if activation == "relu":
            alpha = 0.0
        elif self.evolve_alpha:
            alpha = self.genome.get_alpha()
        else:
            alpha = 0.25

        hidden = inputs @ input_weights + hidden_bias
        hidden = np.maximum(alpha * hidden, hidden)

        return hidden @ output_weights + output_bias

    @staticmethod
    def create(
        genome: MLPGenome,
        activation="prelu",
        evolve_alpha: bool = False,
    ):
        return MLPAgent(genome, activation, evolve_alpha)
