import random
import copy

import numpy as np
import torch

class Crossover:
    '''Collection of crossover functions
        TODO: possibly add others'''

    @staticmethod
    def n_point(ind1, ind2):
        '''N-point crossover
           TODO: implement'''
        raise NotImplementedError("N-point crossover is not implemented yet.")
    
    @staticmethod
    def uniform(ind1, ind2):
        '''Uniform crossover'''
        g1 = ind1.genome
        g2 = ind2.genome
        g3 = copy.deepcopy(g1)
        for key in vars(g3):
            if random.random() < 0.5:
                  vars(g3)[key] = vars(g2)[key]
        g3._verify_hidden_layers()
        g3._verify_neuron_cnt()
        return g3

        

       
        