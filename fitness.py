
import numpy as np
import torch
from torch.utils.data import DataLoader

import neural_network
from evolution_parameters import EvolutionParameters

class FitnessEvaluator:
    def __init__(self, evolution_params : EvolutionParameters, 
                 individual, device):

        self.device = device
        self.individual = individual
        self.evolution_parameters = evolution_params
        self.nn = neural_network.NeuralNetwork(individual.genome, evolution_params)
        self.model = self.nn.to(device)
        self.training_dataloader = DataLoader(evolution_params.training_data, batch_size=individual.genome.batch_size)
        self.test_dataloader = DataLoader(evolution_params.test_data, batch_size=individual.genome.batch_size)
        self.loss_fn = evolution_params.loss_fn
        self.optimizer = evolution_params.optimizer(self.model.parameters())
        
    
    def _init_device(self):
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

    def evaluate(self):
        self.individual.failed = False #Reset failure
        
        epochs = self.individual.genome.epochs
        accuracy_series = np.zeros(epochs)
        loss_series = np.zeros(epochs)
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            self.train()
            accuracy, loss = self.test()
            accuracy_series[t] = accuracy
            loss_series[t] = loss
            
            #TODO: if we're using anything else than average loss for fitness, then the logic must be changed 
            if t >= self.evolution_parameters.failure_window: # We can check for failures (early stopping)

                if loss_series[t] > loss_series[t-self.evolution_parameters.failure_window] - self.evolution_parameters.failure_threshold \
                    and loss_series[t] > self.evolution_parameters.failure_bypass:
                    print('Individual failed.')
                    self.individual.failed = True
                    return (accuracy_series, loss_series)
                
        return (accuracy_series, loss_series)
        


    
#######################################################################
# In a single training loop, the model makes predictions on the training dataset (fed to it in batches), and
# backpropagates the prediction error to adjust the model's parameters.

    def train(self):
        dataloader = self.training_dataloader
        nnet = self.model
        loss_fn = self.loss_fn
        optimizer = self.optimizer

        size = len(dataloader.dataset)
        nnet.train()
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(self.device), y.to(self.device)

            # Compute prediction error
            pred = nnet(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

##############################################################################
# We also check the model's performance against the test dataset to ensure it is learning.

    def test(self):
        dataloader = self.test_dataloader
        nnet = self.model
        loss_fn = self.loss_fn

        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        nnet.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                pred = nnet(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= num_batches # Accuracy
        correct /= size # Average loss
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

        return (test_loss, correct)

