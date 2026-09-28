import numpy as np


class AdalineAgent:

    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def reset(self):
        pass

    def act(self, observation):
        inputs = np.delete(observation.flatten(), 4)
        outputs = self.predict(inputs)

        return int(np.argmax(outputs))

    def predict(self, inputs):
        return inputs @ self.weights + self.bias
    
    def create_adaline_agent(genome):
        return AdalineAgent(
            weights=genome.get_weights(),
            bias=genome.get_bias()
        )

    def predict_debug(self, observation):
        inputs = np.delete(observation.flatten(), 4)
        outputs = self.predict(inputs)
        action = int(np.argmax(outputs))

        return inputs, outputs, action