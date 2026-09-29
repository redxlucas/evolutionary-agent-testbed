from agents.genome import MovementGenome

class MovementAgent:
    """
    Agent that sequentially executes the actions defined by a MovementGenome.

    The agent maintains the current position in the genome and returns one action at each step of the environment.
    """

    def __init__(self, genome: MovementGenome):
        self.genome = genome
        self.current_gene = 0

    def reset(self):
        self.current_gene = 0

    def act(self, observation=None):
        """
        Returns the next action defined by the genome.
        """
        action = self.genome[self.current_gene]
        self.current_gene += 1

        return action

    @staticmethod
    def create(genome: MovementGenome):
        return MovementAgent(genome)