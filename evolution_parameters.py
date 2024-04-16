class EvolutionParameters:
    def __init__(self, mutation_probability, crossover_probability,
                 training_data, test_data, loss_fn, optimizer, tournament_size,
                 failure_threshold = 0.1, failure_window = 3, failure_bypass = 1.8,
                 input_layer = None, output_layer = None, elitism = True):
        self.mutation_probability = mutation_probability
        self.crossover_probability = crossover_probability
        self.training_data = training_data
        self.test_data = test_data
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.tournament_size = tournament_size
        self.failure_threshold = failure_threshold
        self.failure_window = failure_window
        self.failure_bypass = failure_bypass
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.elitism = elitism