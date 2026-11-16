import random

from agents.genome import Genome
import config


class MLPGenome(Genome):

    INPUT_SIZE = config.OBSERVATION_SIZE ** 2 - 1 # 8
    HIDDEN_SIZE = INPUT_SIZE # camada oculta deve sempre possuir valores iguais ao de entrada? # 8
    DEFAULT_OUTPUT_SIZE = 4

    GENE_MIN = -1.0
    GENE_MAX = 1.0

    def __init__(self, genes, hidden_size=None, output_size=None):
        self.hidden_size = hidden_size or self.HIDDEN_SIZE
        self.output_size = output_size or self.DEFAULT_OUTPUT_SIZE
        self.input_weights_size = self.INPUT_SIZE * self.hidden_size
        self.hidden_bias_size = self.hidden_size
        self.alpha_size = self.hidden_size
        self.output_weights_size = self.hidden_size * self.output_size
        self.output_bias_size = self.output_size
        self.genome_length = (
            self.input_weights_size
            + self.hidden_bias_size
            + self.output_weights_size
            + self.output_bias_size
            + self.alpha_size
        )
        super().__init__(genes)

    def get_gene_values(self) -> tuple[float, float]:
        return self.GENE_MIN, self.GENE_MAX

    def get_gene_bounds(self) -> tuple[float, float]:
        return self.GENE_MIN, self.GENE_MAX

    @staticmethod
    def random_gene():
        return random.uniform(MLPGenome.GENE_MIN, MLPGenome.GENE_MAX)

    @classmethod
    def random(cls, length=None, hidden_size=None, output_size=None):
        hidden_size = hidden_size or cls.HIDDEN_SIZE
        output_size = output_size or cls.DEFAULT_OUTPUT_SIZE
        length = length or cls.genome_length_for(
            hidden_size,
            output_size,
        )

        alpha_size = hidden_size
        network_parameter_count = length - alpha_size

        genes = [
            cls.random_gene()
            for _ in range(network_parameter_count)
        ]

        genes.extend(
            [0.0] * alpha_size
        )

        return cls(
            genes,
            hidden_size=hidden_size,
            output_size=output_size,
        )

    @classmethod
    def genome_length_for(cls, hidden_size, output_size=None):
        output_size = output_size or cls.DEFAULT_OUTPUT_SIZE
        return (
            cls.INPUT_SIZE * hidden_size
            + hidden_size
            + hidden_size * output_size
            + output_size
            + hidden_size
        )

    def copy(self):
        return MLPGenome(
            self.genes.copy(),
            hidden_size=self.hidden_size,
            output_size=self.output_size,
        )

    def get_input_weights(self):
        end = self.input_weights_size

        return self.genes[:end].reshape(
            self.INPUT_SIZE,
            self.hidden_size
        )

    def get_hidden_bias(self):
        start = self.input_weights_size
        end = start + self.hidden_bias_size

        return self.genes[start:end]

    def get_output_weights(self):
        start = (
            self.input_weights_size
            + self.hidden_bias_size
        )
        end = start + self.output_weights_size

        return self.genes[start:end].reshape(
            self.hidden_size,
            self.output_size,
        )

    def get_output_bias(self):
        start = (
            self.input_weights_size
            + self.hidden_bias_size
            + self.output_weights_size
        )
        end = start + self.output_bias_size

        return self.genes[start:end]

    def get_alpha(self):
        start = (
            self.input_weights_size
            + self.hidden_bias_size
            + self.output_weights_size
            + self.output_bias_size
        )
        end = start + self.alpha_size

        return self.genes[start:end]
