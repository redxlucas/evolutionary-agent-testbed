import numpy as np

from agents.genome import AdalineGenome

class AdalineAgent:
    """
    Agent that selects actions using an ADALINE model.
    """

    def __init__(self, weights: np.ndarray, bias: np.ndarray):
        self.weights = weights
        self.bias = bias

    def reset(self):
        pass

    def act(self, observation):
        """
        Selects an action based on the given observation.
        """
        inputs = np.delete(observation.flatten(), observation.size // 2)
        outputs = self.predict(inputs)

        return int(np.argmax(outputs))

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