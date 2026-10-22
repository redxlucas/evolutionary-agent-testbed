import random

from agents.genome import Genome
import config


class MLPGenome(Genome):

    INPUT_SIZE = config.OBSERVATION_SIZE ** 2 - 1 # 8
    HIDDEN_SIZE = INPUT_SIZE # camada oculta deve sempre possuir valores iguais ao de entrada? # 8
    OUTPUT_SIZE = config.NUM_MOVEMENTS # 4

    INPUT_WEIGHTS_SIZE = INPUT_SIZE * HIDDEN_SIZE # 8 * 8 = 64
    HIDDEN_BIAS_SIZE = HIDDEN_SIZE # 8
    ALPHA_SIZE = HIDDEN_SIZE

    OUTPUT_WEIGHTS_SIZE = HIDDEN_SIZE * OUTPUT_SIZE # 8 x 4 = 32
    OUTPUT_BIAS_SIZE = OUTPUT_SIZE # 4

    GENOME_LENGTH = (
        INPUT_WEIGHTS_SIZE
        + HIDDEN_BIAS_SIZE
        + OUTPUT_WEIGHTS_SIZE
        + OUTPUT_BIAS_SIZE
        + ALPHA_SIZE
    )

    GENE_MIN = -1.0
    GENE_MAX = 1.0

    def __init__(self, genes, hidden_size=None):
        self.hidden_size = hidden_size or self.HIDDEN_SIZE
        self.input_weights_size = self.INPUT_SIZE * self.hidden_size
        self.hidden_bias_size = self.hidden_size
        self.alpha_size = self.hidden_size
        self.output_weights_size = self.hidden_size * self.OUTPUT_SIZE
        self.output_bias_size = self.OUTPUT_SIZE
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

    @staticmethod
    def random_gene():
        return random.uniform(MLPGenome.GENE_MIN, MLPGenome.GENE_MAX)

    @classmethod
    def random(cls, length=None, hidden_size=None):
        hidden_size = hidden_size or cls.HIDDEN_SIZE
        length = length or cls.genome_length_for(hidden_size)

        alpha_size = hidden_size
        network_parameter_count = length - alpha_size

        genes = [
            cls.random_gene()
            for _ in range(network_parameter_count)
        ]

        genes.extend(
            [0.0] * alpha_size
        )

        return cls(genes, hidden_size=hidden_size)

    @classmethod
    def genome_length_for(cls, hidden_size):
        return (
            cls.INPUT_SIZE * hidden_size
            + hidden_size
            + hidden_size * cls.OUTPUT_SIZE
            + cls.OUTPUT_SIZE
            + hidden_size
        )

    def copy(self):
        return MLPGenome(
            self.genes.copy(),
            hidden_size=self.hidden_size,
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
            self.OUTPUT_SIZE,
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
