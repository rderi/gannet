import random
import numpy as np
import torch

class Mutation:
    '''Collection of mutation functions
       TODO: possibly add others'''

    @staticmethod
    def random_resetting(individual, probability):
        '''Randomly reset the value of a gene in the genome with a given probability.'''
        params = individual.genome_params
        genome = individual.genome
        for key in vars(genome):
            if random.random() < probability:
                individual.fitness_needs_update = True
                print(f'Mutating individual {individual.id} at key {key}')
                if key == 'epochs':
                    genome.epochs = np.random.randint(params.min_epochs, params.max_epochs)
                elif key == 'batch_size':
                    genome.batch_size = np.random.randint(params.min_batch_size, params.max_batch_size)
                elif key == 'hidden_layers':
                    genome.hidden_layers = np.random.randint(params.min_hidden_layers, params.max_hidden_layers)
                    
                elif key == 'activation_fns':
                    idx = np.random.randint(0, genome.hidden_layers)
                    genome.activation_fns[idx] = random.choice(params.activation_functions)
                elif key == 'neuron_cnt':
                    idx = np.random.randint(0, genome.hidden_layers)
                    genome.neuron_cnt[idx] = np.random.randint(params.min_hidden_neurons, params.max_hidden_neurons)
                    
                    #Avoid reducing number of features when activation function is not linear
                    #TODO: does this put a bottleneck on mutation?
                genome._verify_hidden_layers()
                genome._verify_neuron_cnt()
        