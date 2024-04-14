from torch import nn

all_activation_fns = [nn.Linear, nn.Sigmoid, nn.Tanh, nn.ReLU,
                      nn.LeakyReLU, nn.PReLU, nn.ELU, nn.Softmax,
                      nn.Softmin, nn.Hardswish, nn.GELU, nn.SELU]
                      #nn.Conv1d, nn.Conv2d, nn.Conv3d]

class ActivationFunctions:
    '''Static class containing methods to specify possible activation functions for a genome.'''
    @staticmethod
    def get_activation_functions(desc):
        '''Get a list of pytorch activation functions matching desc.
           
           Options:
                - 'all': get all supported activation functions'''
        if desc == 'all':
            return all_activation_fns