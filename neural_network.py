
from collections import defaultdict, OrderedDict

import torch
from torch import nn

from genomes import Genome

# Define model
class NeuralNetwork(nn.Module):
    '''Neural network that is derive from a genome.'''
    def __init__(self, genome, evolution_parameters):
        '''Initialize neural netwrok from genome.
        
           Optional arguments:
            - input layer: tuple of form (activation function, input features)
            - output layer: tuple of form (activation function, output features)'''
        super().__init__()
        self.inputlayer = evolution_parameters.input_layer
        self.outputlayer = evolution_parameters.output_layer
        self.flatten = nn.Flatten()
        self.model = nn.Sequential(self.create_layers(genome))

    def forward(self, x):
        x = self.flatten(x)
        logits = self.model(x)
        return logits
    
    def create_layers(self, genome):
        '''Create layers from genome. If needed, add user-specified input and output layers'''
        
        #Stores apperance count of activation functions, so each layer has unique name
        activation_cnts = defaultdict(lambda: 0) 
        
        layers = OrderedDict()

        if self.inputlayer is not None:
            input_activation = self.inputlayer[0]
            if input_activation == nn.Linear:
                input_features = self.inputlayer[1]
                layers["input"] = input_activation(input_features, genome.neuron_cnt[0])
            else: #TODO: make sure these two cases are enough, or else expand
                layers["input"] = input_activation()

        for i in range(genome.hidden_layers):
            activation = genome.activation_fns[i] #Stor
            layer_name = activation.__name__ + str(activation_cnts[activation.__name__])
            activation_cnts[activation.__name__] += 1
            curr_neurons = genome.neuron_cnt[i]
            #if i+1 < genome.hidden_layers:
            
            if activation == nn.Linear:
                if i + 1 == genome.hidden_layers:
                    layers[layer_name] = activation(curr_neurons, genome.neuron_cnt[-1]) #TODO: neuron cnt stays the same; maybe have neuron_cnt+1th neuron count generated
                else:
                    next_neurons = genome.neuron_cnt[i+1]
                    layers[layer_name] = activation(curr_neurons, next_neurons)
            else:
                layers[layer_name] = activation() 
        
        if self.outputlayer is not None:
            output_activation = self.outputlayer[0]
            if output_activation == nn.Linear:
                output_features = self.outputlayer[1]
                layers["output"] = output_activation(genome.neuron_cnt[-1], output_features)
            else: #TODO: make sure these two cases are enough, or else expand
                layers["output"] = output_activation()

        return layers 
            
