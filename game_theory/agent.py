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
        self.scores = [0, 0, 0, 0, 0, 0, 0, 0]
        self.total_score = 0

        self.move = None
        self.next_move = None

        self.strategy = ""
        self.new_strategy = ""
        self.alive = True
        # self.crowded = []
        self.neighbors = []

        if self.model.biomes:
            for i in range(len(self.model.biome_boundaries)-1):
                if self.model.biome_boundaries[i] <= self.pos[0] < self.model.biome_boundaries[i+1]:
                    self.strategy = self.model.agent_strategies[i]
        else:
            self.strategy = np.random.choice(a=self.model.agent_strategies, p=self.model.initial_population_sizes)

    def kill_weak(self):
        if random.random() < self.model.probability_death and self.alive:
            self.neighbors = self.model.grid.get_neighbors(self.pos, moore=True)
            # if all(self.strategy == neighbour.strategy for neighbour in self.neighbors) and self.model.kill_crowded:
            #     self.alive = False

            if self.total_score < self.model.cull_score:
                self.model.num_dying += 1
                self.alive = False

    def reproduce_strong(self):
        self.neighbors = self.model.grid.get_neighbors(self.pos, moore=True)
        logger.warn("Alive: {}"
                    "\nStrategy: {}"
                    "\nScores (total): {} {}".format(self.alive, self.strategy, self.scores, self.total_score))
        if not self.alive:
            if random.random() < self.model.probability_adoption:
                if self.strategy == "empty":
                    self.new_strategy = self.neighbors[np.argmax(
                            [neighbor.total_score for neighbor in self.neighbors])].strategy
                    logger.warn("Empty strategy adopting...")
                    self.model.num_dead -= 1
                else:
                    # the strongest neighbour is that which beat self the most
                    self.new_strategy = self.neighbors[np.argmin(self.scores)].strategy
                    logger.warn("Dead agent adopting...")
                self.alive = True
                self.model.num_evolving += 1
            else:
                self.new_strategy = "empty"
                if self.strategy != "empty":
                    logger.warn("Dead agent became empty...")
                    self.model.num_dead += 1
                else:
                    logger.warn("Empty agent did not change its state...")
        else:
            logger.warn("This agent did not change its strategy...")
            self.new_strategy = self.strategy

        if random.random() < self.model.probability_mutation:
            available_strategies = [strategy for strategy in self.model.agent_strategies if
                                    strategy != self.strategy]
            self.new_strategy = random.choice(available_strategies)

            if self.strategy == "empty":
                self.model.num_dead -= 1
                logger.warn("Dead agent mutating...")
            else:
                logger.warn("Agent mutating...")
            self.model.num_mutating += 1

        logger.warn("New Strategy: {}".format(self.new_strategy))

    def exchange(self):
        self.neighbors = self.model.grid.get_neighbors(self.pos, moore=True)
        if random.random() < self.model.probability_exchange:
            random_neighbor = random.choice([neighbor for neighbor in self.neighbors])
            self.new_strategy = random_neighbor.strategy
            random_neighbor.new_strategy = self.strategy
        else:
            self.new_strategy = self.strategy

class RPSAgent(GameAgent):

    def __init__(self, pos, model):
        super().__init__(pos, model)

        self.implement_strategy()

    def increment_score(self):
        for num, neighbor in enumerate(self.model.grid.neighbor_iter(self.pos)):
            self.scores[num] = 0
            if random.random() < self.model.probability_playing:
                self.scores[num] += self.model.payoff[self.move, neighbor.move]
        self.total_score = sum(self.scores)

    def implement_strategy(self):
        if self.strategy == "all_r":
            self.move = "R"
        elif self.strategy == "all_p":
            self.move = "P"
        elif self.strategy == "all_s":
            self.move = "S"
        elif self.strategy == "empty":
            self.move = "E"

    def update_strategy(self):
        self.strategy = self.new_strategy



