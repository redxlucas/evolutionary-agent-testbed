from .genome import Genome

class GenomeAgent:
    """
    Classe que representa um agente que interage com o ambiente.
    Guarda o genoma, posição atual e fitness, e fornece métodos para
    resetar estado, agir com base na observação e atualizar fitness.
    """

    def __init__(self, genome: Genome):
        self.genome = genome
        self.current_gene = 0

    def __repr__(self):
        return f"Agent(fitness={self.fitness})"

    def reset(self):
        self.current_gene = 0

    def act(self, observation=None):
        """
        Retorna uma ação baseada no genoma.
        """
        
        action = self.genome.get_gene(self.current_gene)
        self.current_gene += 1

        return action
    
    def create_genome_agent(genome):
        return GenomeAgent(genome)