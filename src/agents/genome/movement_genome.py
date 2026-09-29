from __future__ import annotations

import random
from .genome import Genome
import config

class MovementGenome(Genome):

    def __init__(self, genes):
        super().__init__(genes)
        self.genes = self.genes.astype(int)

    @staticmethod
    def random_gene():
        return random.randrange(config.NUM_MOVEMENTS)