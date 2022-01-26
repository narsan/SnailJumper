import copy
import math

from player import Player
import numpy as np


class Evolution:
    def __init__(self):
        self.game_mode = "Neuroevolution"
        self.generation_number = 0

    def calculate_fitness(self, players, delta_xs):
        for i, p in enumerate(players):
            p.fitness = delta_xs[i]

    def mutation(self, child, pw, pb):
        random_number = np.random.uniform(0, 1, 1)
        if random_number <= pw:
            child.nn.W1 += np.random.normal(0, 0.3, child.nn.W1.shape)
        if random_number <= pb:
            child.nn.b1 += np.random.normal(0, 0.3, child.nn.b1.shape)
        if random_number <= pw:
            child.nn.W2 += np.random.normal(0, 0.3, child.nn.W2.shape)
        if random_number <= pb:
            child.nn.b2 += np.random.normal(0, 0.3, child.nn.b2.shape)
        return child

    def q_tournament(self, players, q):
        q_selected = np.random.choice(players, q)
        return max(q_selected, key=lambda player: player.fitness)

    def roulette_wheel(self, players, num_players):
        population_fitness = sum([player.fitness for player in players])
        probability = [player.fitness / population_fitness for player in players]
        next_gen = np.random.choice(players, size=num_players, p=probability, replace=False)
        return next_gen

    def next_population_selection(self, players, num_players):
        """
        Gets list of previous and current players (μ + λ) and returns num_players number of players based on their
        fitness value.

        :param players: list of players in the previous generation
        :param num_players: number of players that we return
        """
        # TODO (Implement top-k algorithm here)
        sorted_players = sorted(players, key=lambda x: x.fitness, reverse=True)
        self.plot_data_saving(players[: num_players])
        return sorted_players[: num_players]

        # TODO (Additional: Implement roulette wheel here)
        # next_population = self.roulette_wheel(players, num_players).tolist()
        # self.plot_data_saving(next_population)
        # return next_population

    def cross_over(self, p1, p2):
        threshold = 0.8
        crossover_pos = math.floor(p1.shape[0] / 2)

        random_number = np.random.uniform(0, 1, 1)
        if random_number > threshold:
            return p1, p2

        else:
            child1_array = np.concatenate((p1[:crossover_pos], p2[crossover_pos:]), axis=0)
            child2_array = np.concatenate((p2[:crossover_pos], p1[crossover_pos:]), axis=0)
        return child1_array, child2_array

    def reproduction(self, parent1, parent2):
        child1 = self.clone_player(parent1)
        child2 = self.clone_player(parent2)
        child1.nn.W1, child2.nn.W1 = self.cross_over(parent1.nn.W1, parent2.nn.W1)
        child1.nn.W2, child2.nn.W2 = self.cross_over(parent1.nn.W2, parent2.nn.W2)
        child1.nn.b1, child2.nn.b1 = self.cross_over(parent1.nn.b1, parent2.nn.b1)
        child1.nn.b2, child2.nn.b2 = self.cross_over(parent1.nn.b2, parent2.nn.b2)
        child1 = self.mutation(child1, 0.35, 0.7)
        child2 = self.mutation(child2, 0.35, 0.7)
        return child1, child2


    def generate_new_population(self, num_players, prev_players=None):
        """
        Gets survivors and returns a list containing num_players number of children.

        :param num_players: Length of returning list
        :param prev_players: List of survivors
        :return: A list of children
        """
        first_generation = prev_players is None
        if first_generation:
            return [Player(self.game_mode) for _ in range(num_players)]
        else:
            parents = []
            new_players = prev_players
            for _ in range(num_players):
                # TODO (Additional) implement q tournament
                parents.append(self.q_tournament(prev_players, q=3))
            for i in range(0, len(parents), 2):
                parent1 = parents[i]
                parent2 = parents[i + 1]
                child1, child2 = self.reproduction(parent1, parent2)
                new_players.append(child1)
                new_players.append(child2)

            new_players.sort(key=lambda x: x.fitness, reverse=True)
            new_players = new_players[: num_players]
            return new_players

    def clone_player(self, player):
        """
        Gets a player as an input and produces a clone of that player.
        """
        new_player = Player(self.game_mode)
        new_player.nn = copy.deepcopy(player.nn)
        new_player.fitness = player.fitness
        return new_player

    # TODO (Additional) plot answers

    def plot_data_saving(self, players):
        with open(f'max.csv', 'a') as file:
            file.write(str(sorted(players, key=lambda x: x.fitness, reverse=True)[0].fitness))
            file.write("\n")
        with open(f'min.csv', 'a') as file:
            file.write(str(sorted(players, key=lambda x: x.fitness)[0].fitness))
            file.write("\n")
        ave = 0
        for p in players:
            ave += p.fitness
        ave /= len(players)
        with open(f'average.csv', 'a') as file:
            file.write(str(ave))
            file.write("\n")
