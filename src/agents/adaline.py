import numpy as np

class Adaline:

    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def act(self, inputs):
        outputs = self.predict(inputs)
        return np.argmax(outputs)

    def predict(self, inputs):
        return inputs @ self.weights + self.bias