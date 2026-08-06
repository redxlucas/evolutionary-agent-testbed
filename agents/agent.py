class Agent:

    def __init__(self, genome):
        self.genome = genome
        self.position = None
        self.fitness = 0

    def reset(self, start_position):
        self.position = start_position
        self.fitness = 0

    def act(self, observation):
        """
        Retorna uma ação baseada no genoma.
        """
        pass

    def update_fitness(self, reward):
        self.fitness += reward