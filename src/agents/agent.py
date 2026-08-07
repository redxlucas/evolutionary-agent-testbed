from src.agents.genome import Genome

class Agent:
    """
    Classe que representa um agente que interage com o ambiente.
    Guarda o genoma, posição atual e fitness, e fornece métodos para
    resetar estado, agir com base na observação e atualizar fitness.
    """

    def __init__(self, genome: Genome):
        self.genome = genome
        self.position = None
        self.fitness = 0
        self.current_gene = 0

    def reset(self, start_position):
        self.position = start_position
        self.fitness = 0

    def act(self, observation=None):
        """
        Retorna uma ação baseada no genoma.
        """
        action = self.genome.get_gene(self.current_gene)
        self.current_gene += 1

        return action

    def update_fitness(self, reward):
        self.fitness += reward