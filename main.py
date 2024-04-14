import numpy as np
import torch
from individual import Individual
from neural_network import NeuralNetwork
from fitness import FitnessEvaluator
from torchvision import datasets
from torchvision.transforms import ToTensor
from generation import Generation
from evolution_parameters import EvolutionParameters
from genome_parameters import GenomeParameters

population_cnt = 10
mutation_prob = 0.3
crossover_prob = 0.1
generations = 10

training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

evo_params = EvolutionParameters(mutation_prob, crossover_prob,
                                 training_data, test_data,
                                 torch.nn.CrossEntropyLoss(),
                                 lambda x: torch.optim.SGD(x, lr=1e-3),
                                 5,
                                 input_layer=(torch.nn.Linear, 28*28),
                                 output_layer=(torch.nn.Linear, 10))

genome_params = GenomeParameters(min_hidden_neurons=256, max_hidden_neurons=1024,
                max_hidden_layers=10,
                min_batch_size=64, max_epochs=3)


def run_ea():
        print("Starting EA...")
        generation = Generation(population_cnt, evo_params, genome_params)
        
        for _ in range(generations):
                for idx, individual in enumerate(generation.population):
                        print(f'Individual #{idx} fitness: {individual.fitness}')
                generation.progress_generation()
                
        print("EA finished.")
        
        

        
        

if __name__ == '__main__':
        run_ea()