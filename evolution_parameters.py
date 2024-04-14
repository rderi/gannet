class EvolutionParameters:
    def __init__(self, mutation_probability, crossover_probability,
                 training_data, test_data, loss_fn, optimizer, tournament_size,
                 input_layer = None, output_layer = None, elitism = True):
        self.mutation_probability = mutation_probability
        self.crossover_probability = crossover_probability
        self.training_data = training_data
        self.test_data = test_data
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.tournament_size = tournament_size
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.elitism = elitism