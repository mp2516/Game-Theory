from mesa import Agent
import random
import math
from .logger import logger
import numpy as np


def round_to_unity(probabilities):
    probs_round = [np.floor(prob) for prob in probabilities]
    remainder = [(probabilities[i] - prob_round) for (i, prob_round) in enumerate(probs_round)]
    while sum(probs_round) < 1:
        key = np.argmax(remainder)
        probs_round[key] += 1
        remainder[key] = 0
    return probs_round


class GameAgent(Agent):

    unique_id = 1

    def __init__(self, pos, model):
        super().__init__(GameAgent.unique_id, model)
        GameAgent.unique_id += 1
        self.pos = pos

        # the length of this array is 8 as their are 8 neighbours for every agent
        if self.model.diagonal_neighbours:
            self.scores = [0, 0, 0, 0, 0, 0, 0, 0]
        else:
            self.scores = [0, 0, 0, 0]
        self.total_score = 0

        self.move = None
        self.next_move = None

        self.probabilities = [1/3, 1/3, 1/3]
        self.new_probabilities = [1/3, 1/3, 1/3]

        self.strategy = ""
        self.new_strategy = ""

        self.evolved = False
        self.neighbors = []

        if self.model.biomes:
            if int(self.model.dimension / 9) < self.pos[0] < int(self.model.dimension * 4 / 9) and int(self.model.dimension * 5 / 9) < self.pos[1] < int(self.model.dimension * 8 / 9):
                self.strategy = "all_r"
            elif int(self.model.dimension * 5 / 9) < self.pos[0] < int(self.model.dimension * 8 / 9) and int(self.model.dimension * 5 / 9) < self.pos[1] < int(self.model.dimension * 8 / 9):
                self.strategy = "all_p"
            elif int(self.model.dimension / 3) < self.pos[0] < int(self.model.dimension * 2 / 3) and int(self.model.dimension * 1 / 9) < self.pos[1] < int(self.model.dimension * 4 / 9):
                self.strategy = "all_s"

            else:
                self.strategy = "empty"
                
        else:
            self.strategy = np.random.choice(a=self.model.agent_strategies, p=self.model.initial_population_sizes)

    def exchange(self):
#        self.neighbors = self.model.grid.get_neighbors(self.pos, moore=self.model.diagonal_neighbours)
        if random.random() < self.model.probability_exchange:
            random_neighbor = random.choice([neighbor for neighbor in self.model.grid.get_neighbors(self.pos, moore=self.model.diagonal_neighbours)])
#            a = self.strategy
            self.new_strategy = random_neighbor.strategy
            random_neighbor.new_strategy = self.strategy
            self.implement_strategy()    
            random_neighbor.implement_strategy()
        

    def identify_crowded(self):
        self.neighbors = self.model.grid.get_neighbors(self.pos, moore=self.model.diagonal_neighbours)
        if all(self.strategy == neighbour.strategy for neighbour in self.neighbors):
            self.model.crowded_players.append(self)

    #        crowded_players = [player for player in self.model.schedule.agents if all(self.strategy == neighbour.strategy for neighbour in self.neighbors)]


    def kill_crowded(self):
        for player in self.model.crowded_players:
            player.new_strategy = "empty"

#        self.model.crowded_players.clear()


    def kill_weak(self):
        if random.random() < self.model.probability_cull_score_decrease:
            cull_threshold = self.model.cull_score - 1
        else:
            cull_threshold = self.model.cull_score
        if self.total_score < cull_threshold and random.random() <= self.model.probability_death:
            self.new_strategy = "empty"
        else:
            self.new_strategy = self.strategy
            self.new_probabilities = self.probabilities


    def reproduce(self):
        self.neighbors = self.model.grid.get_neighbors(self.pos, moore=self.model.diagonal_neighbours)
        if not all(neighbour.strategy == "empty" for neighbour in self.neighbors):
            neighbours_not_empty = [neighbour for neighbour in self.neighbors if neighbour.strategy != "empty"]
            random.shuffle(neighbours_not_empty)
            strongest_neighbour = neighbours_not_empty[
                np.argmax([neighbour.total_score for neighbour in neighbours_not_empty])]
            random_neighbour = random.choice(neighbours_not_empty)
            rand = random.random()
            if self.strategy == "empty" and strongest_neighbour.total_score >= 0 and rand <= self.model.probability_adoption and rand > self.model.probability_mutation:
                self.new_strategy = strongest_neighbour.strategy
            elif self.strategy == "empty" and strongest_neighbour.total_score >= 0 and rand <= self.model.probability_adoption and rand <= self.model.probability_mutation:
                self.new_strategy = random.choice(
                    random.choices(population=["all_r", "all_p", "all_s"], weights=[100, 100, 100], k=1))
            else:
                self.new_strategy = self.strategy
                self.new_probabilities = self.probabilities
        else:
            self.new_strategy = self.strategy
            self.new_probabilities = self.probabilities


class RPSAgent(GameAgent):

    def __init__(self, pos, model):
        super().__init__(pos, model)

        self.implement_pure_strategies()

    def set_score_to_zero(self):
        self.scores = [0, 0, 0, 0, 0, 0, 0, 0]
    
    def increment_score(self):
        for num, neighbor in enumerate(self.model.grid.neighbor_iter(self.pos)):
            if self.model.game_mode == "Pure":
#                self.scores[num] = 0
                if random.random() < self.model.probability_playing:
                    self.scores[num] += self.model.payoff[self.move, neighbor.move]
                    neighbor.scores[num] += self.model.payoff[neighbor.move, self.move]
        
    def sum_scores(self):
        if self.pos == (0,0) or self.pos == (0, self.model.dimension-1) or self.pos == (self.model.dimension-1, 0) or self.pos == (self.model.dimension-1, self.model.dimension-1):
            self.total_score = sum(self.scores) * 4 / 3
        elif self.pos[0] == 0 or self.pos[0] == self.model.dimension-1 or self.pos[1] == 0 or self.pos[1] == self.model.dimension-1:
            self.total_score = sum(self.scores) * 4 / 5
        else:
            self.total_score = sum(self.scores)/2

    def implement_pure_strategies(self):
        if self.strategy == "all_r":
            self.probabilities = [1, 0, 0]
            self.move = "R"
        elif self.strategy == "all_p":
            self.probabilities = [0, 1, 0]
            self.move = "P"
        elif self.strategy == "all_s":
            self.probabilities = [0, 0, 1]
            self.move = "S"
        elif self.strategy == "empty":
            self.move = "E"

    def implement_strategy(self):
        self.strategy = self.new_strategy
        self.probabilities = self.new_probabilities

        self.implement_pure_strategies()