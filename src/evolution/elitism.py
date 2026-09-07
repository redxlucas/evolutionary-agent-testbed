from agents import Agent
from evolution import Population

class Elitism:

    def __init__(self, elite_size: int):
        if elite_size < 0:
            raise ValueError(
                "Elite size cannot be negative."
            )
        
        self.elite_size = elite_size

    def select_elites(self, population: Population) -> Population:
         
        if self.elite_size > len(population):
            raise ValueError(
                "Elite size cannot be greater than population size."
            )
         
        selected_agents = sorted(
             population, # ver se precisa chamar .individuals
             key=lambda agent: agent.fitness,
             reverse=True     
        )[:self.elite_size]

        return self._copy_population(selected_agents=selected_agents)
    
    def _copy_population(self, selected_agents) -> Population:

        elites = []

        for agent in selected_agents:

            elite_genome = agent.genome.copy()
            elite_agent = Agent(elite_genome)
            elite_agent.fitness = agent.fitness
            elite_agent.success = agent.success

            elites.append(elite_agent)

        return Population(elites)

         
        