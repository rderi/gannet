import numpy as np
import torch
import dill

from individual import Individual
from genomes import Genome
from genome_parameters import GenomeParameters
from evolution_parameters import EvolutionParameters
from selection import Selection

class Generation:
    def __init__(self, population, evolution_params : EvolutionParameters,
                 genome_params : GenomeParameters, device = None):
            self.device = device if device else self.get_device()
            self.curr_generation = 1
            self.population_cnt = population
            self.evolution_params = evolution_params
            self.genome_params = genome_params
            self.population = self.generate_population(self.population_cnt)
    
    def generate_population(self, population_size):
        '''Generate a population of individuals with random genomes'''
        return [Individual(self.evolution_params, self.genome_params, self.device) 
                for _ in range(population_size)]
    
    def mutate_population(self):
        '''Mutate the entire population'''
        for individual in self.population:
            individual.mutate()
            if individual.fitness_needs_update:
                individual.measure_fitness()
    
    def crossover_population(self):
        
        #Figure out how many offspring are produced via crossover; using wacky numpy because its probably much faster
        crossover_cnt = ((np.random.random(self.population_cnt)) < self.evolution_params.crossover_probability).sum()
        for _ in range(crossover_cnt):
            parent1, parent2 = Selection.tournament_selection(self.population, self.evolution_params.tournament_size, 2)
            print(f'Crossing over {parent1.id} and {parent2.id}.')
            res = parent1.crossover(parent2)
            res.measure_fitness()
            self.population.append(res)
        
    
    def progress_generation(self):
        '''Progress the generation by one step
           This uses a slightly modified version of (μ, λ) selection to chose parents.
           First, it does mutation and crossover as usual, adding them offspring to the list. 
           We then sort the list, and select the population_cnt best individuals as the new population.
           TODO: Let's see how this works; we might need to add more selection methods / use another one'''
           
        self.mutate_population()
        self.crossover_population()
        self.population.sort(key = lambda x: x.fitness, reverse = True)
        self.population = self.population[:self.population_cnt]
        self.curr_generation += 1
        

    @staticmethod
    def get_device():
        return (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    
    def serialize(self):
        return dill.dumps(self)
