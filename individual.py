from genome_parameters import GenomeParameters
from evolution_parameters import EvolutionParameters
from mutation import Mutation
from crossover import Crossover
from genomes import Genome
from fitness import FitnessEvaluator

class Individual:
    def __init__(self,
                 evolution_params : EvolutionParameters, 
                 genome_params : GenomeParameters, 
                 device,
                 genome = None):
        if genome:
            self.genome = genome
        else:
            self.genome = Genome(genome_params)
        self.evolution_params = evolution_params
        self.genome_params = genome_params
        self.device = device
        self.measure_fitness()

    def mutate(self):
        '''Mutate the individual with a given probability
           TODO: extend if new mutation functions are added'''
        Mutation.random_resetting(self, self.evolution_params.mutation_probability)
    
    def crossover(self, other):
        '''Crossover with another individual'''
        if True: #TODO: if other crossover functions are added, we can check evolution_params.crossover_type here
            res = Individual(self.evolution_params, self.genome_params, self.device, genome = Crossover.uniform(self, other))
            return res
            
    def measure_fitness(self):
        '''Measure the fitness of the individual'''
        self.fitness = FitnessEvaluator(self.evolution_params, self, self.device).evaluate()
        #TOFO: refine; the next operation should be done in fitness.py
        #TODO: does the indexing make sense here even??
        self.fitness = -self.fitness[0][-1] #Take the last accuracy value
        
        
        
        