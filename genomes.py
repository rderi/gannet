import random

import numpy as np
import torch

from activation_functions import ActivationFunctions
from genome_parameters import GenomeParameters

class Genome:
    '''A genome that represents a neural network and its evaluation circumstances (e.g., batch size)'''

    def __init__(self, params: GenomeParameters):
        '''Initialize random genome matching the given parameters
           TODO: the if/else blocks are kinda clunky, find a better way if i have a lot of time'''

        self.params = params
        if params.min_epochs == params.max_epochs:
            self.epochs = params.min_epochs
        else:
            self.epochs = np.random.randint(params.min_epochs, params.max_epochs)
            
        if params.min_batch_size == params.max_batch_size:
            self.batch_size = params.min_batch_size
        else:
            self.batch_size = np.random.randint(params.min_batch_size, params.max_batch_size)
        
        if params.min_hidden_layers == params.max_hidden_layers:
            self.hidden_layers = params.min_hidden_layers
        else:
            self.hidden_layers = np.random.randint(params.min_hidden_layers, params.max_hidden_layers)
            
        self.activation_fns = [random.choice(params.activation_functions) for _ in range(self.hidden_layers)]
        
        if params.min_hidden_neurons == params.max_hidden_neurons:
            self.neuron_cnt = np.full(self.hidden_layers, params.min_hidden_neurons)
        else:
            self.neuron_cnt = np.random.randint(params.min_hidden_neurons, params.max_hidden_neurons, size=self.hidden_layers)

        self._verify_neuron_cnt()
    
    def _verify_neuron_cnt(self, i = None):
        if i: #If layer to check is given
            if i > 0 and self.activation_fns[i-1] != torch.nn.Linear:
                self.neuron_cnt[i] = self.neuron_cnt[i-1]
        elif self.hidden_layers == 1:
            return
        else:
            for i in range(1, self.neuron_cnt.size):
                if self.activation_fns[i-1] != torch.nn.Linear:
                    self.neuron_cnt[i] = self.neuron_cnt[i-1]
    
    def _verify_hidden_layers(self):
        if len(self.activation_fns) != self.hidden_layers:
            common_len = min(len(self.activation_fns), self.hidden_layers)
            self.hidden_layers = common_len
            self.activation_fns = np.random.choice(self.activation_fns, common_len,
                                                   replace=False)
        if len(self.neuron_cnt) != self.hidden_layers:
            common_len = min(len(self.neuron_cnt), self.hidden_layers)
            self.hidden_layers = common_len
            self.neuron_cnt = np.random.choice(self.neuron_cnt, common_len,
                                               replace=False)


    def __str__(self):
        
        return f'[<Genome>:\n\tepochs = {self.epochs}\n\tbatch_size = {self.batch_size}\n\t' +\
               f'hidden_layers = {self.hidden_layers}\n\tneuron_cnt = {self.neuron_cnt}\n\t'+\
               f'activation_functions = {self.activation_fns}'
