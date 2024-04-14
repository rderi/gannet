import random

class Selection:
    '''A collection of parent selection methods'''
    @staticmethod
    def tournament_selection(population, tournament_size, selection_size):
        '''Select a parent using tournament selection'''
        tournament = random.sample(population, tournament_size)
        if selection_size == 1:
            return max(tournament, key = lambda x: x.fitness)
        else:
            tournament.sort(key = lambda x: x.fitness, reverse = True)
            return tournament[:selection_size]
        