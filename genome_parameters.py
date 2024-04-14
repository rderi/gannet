from activation_functions import ActivationFunctions

class GenomeParameters:
    def __init__(self, min_epochs=1, max_epochs=100, min_batch_size = 1, max_batch_size = 100,\
                 min_hidden_layers = 1, max_hidden_layers = 100, min_hidden_neurons = 1,\
                 max_hidden_neurons = 100, activation_functions = None):
        '''Initialize genome parameter configuration'''
        self.min_epochs = min_epochs
        self.max_epochs = max_epochs
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.min_hidden_layers = min_hidden_layers
        self.max_hidden_layers = max_hidden_layers
        self.min_hidden_neurons = min_hidden_neurons
        self.max_hidden_neurons = max_hidden_neurons
        self.activation_functions = activation_functions if activation_functions\
                                       else ActivationFunctions.get_activation_functions('all')
        