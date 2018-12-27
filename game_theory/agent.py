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

        self.probabilities = [1/3, 1/3, 1/3]
        self.new_probabilities = [1/3, 1/3, 1/3]

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
            if all(self.strategy == neighbour.strategy for neighbour in self.neighbors) and self.model.kill_crowded:
                self.alive = False

            elif self.total_score < self.model.cull_score:
                self.alive = False

    def reproduce_strong(self):


        if random.random() < self.model.probability_adoption and not self.alive:
            strongest_neighbour = self.neighbors[np.argmin(self.scores)]
            logger.debug("The scores {} for agent {} with position {}".format(self.scores, self.unique_id, self.pos))
            # if strongest_neighbour.total_score > self.total_score:
            self.new_strategy = strongest_neighbour.strategy
            # FIXME: mutation is occuring even in homogenous systems
            self.model.num_evolving += 1
            logger.debug(
                "Old strategy: {}, new strategy: {}, mutation number: {}".format(self.strategy, self.new_strategy,
                                                                                 self.model.num_evolving))
        else:
            self.new_strategy = self.strategy
            self.new_probabilities = self.probabilities


    def exchange(self):
        self.neighbors = self.model.grid.get_neighbors(self.pos, moore=True)
        if random.random() < self.model.probability_exchange:
            random_neighbor = random.choice([neighbor for neighbor in self.neighbors])
            self.new_strategy = random_neighbor.strategy
            random_neighbor.new_strategy = self.strategy
        else:
            self.new_strategy = self.strategy
            self.new_probabilities = self.probabilities

    def evolve_strategy(self):
        self.neighbors = self.model.grid.get_neighbors(self.pos, moore=True)

        # mutate the agents with a probability
        if random.random() <= self.model.probability_mutation:
            if self.model.game_mode == "Pure":
                available_strategies = [strategy for strategy in self.model.agent_strategies
                                        if strategy != self.strategy]
                self.new_strategy = random.choice(available_strategies)
                self.model.num_mutating += 1
            elif self.model.game_mode == "Impure":
                # the larger the strength_of_mutation
                # the smaller the weights which means a bigger variance on the initial probabilities
                self.model.new_probabilities = np.random.dirichlet(
                    [prob * (1/self.model.strength_of_mutation) for prob in self.probabilities])

        elif random.random() < self.model.probability_exchange:
            random_neighbor = random.choice([neighbor for neighbor in self.neighbors])
            if not self.strategy == random_neighbor.strategy:
                self.new_strategy = random_neighbor.strategy
                random_neighbor.new_strategy = self.strategy

        elif all(self.strategy == neighbour.strategy for neighbour in self.neighbors) and self.model.kill_crowded:
            self.strategy = "empty"

        # kill the weakest agents
        elif self.total_score < self.model.cull_score:
            # self.new_strategy = "empty"
            if random.random() <= self.model.probability_adoption:
                strongest_neighbour = self.neighbors[np.argmin(self.scores)]
                logger.debug("The scores {} for agent {} with position {}".format(self.scores, self.unique_id, self.pos))
                # if strongest_neighbour.total_score > self.total_score:
                if self.model.game_mode == "Impure":
                    for num, i in enumerate(self.probabilities):
                        for j in strongest_neighbour.probabilities:
                            # bad_agents probabilities will tend towards the probabilities of the strongest_neighbour
                            # with the strength_of_adoption dictating how much it tends towards
                            self.new_probabilities[num] = i + ((j - i) * self.model.strength_of_adoption)

                elif self.model.game_mode == "Pure":
                    self.new_strategy = strongest_neighbour.strategy
                    # FIXME: mutation is occuring even in homogenous systems
                    self.model.num_evolving += 1
                    logger.debug("Old strategy: {}, new strategy: {}, mutation number: {}".format(self.strategy, self.new_strategy, self.model.num_evolving))
                else:
                    self.new_strategy = self.strategy
                    self.new_probabilities = self.probabilities
        else:
            self.new_strategy = self.strategy
            self.new_probabilities = self.probabilities


    def reproduce(self):
        self.neighbors = self.model.grid.get_neighbors(self.pos, moore=True)
        if not all(neighbour.strategy == "empty" for neighbour in self.neighbors):
            neighbours_not_empty = [neighbour for neighbour in self.neighbors if neighbour.strategy != "empty"]
            random.shuffle(neighbours_not_empty)
            strongest_neighbour = neighbours_not_empty[np.argmin(self.score)]
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

    def increment_score(self):
        for num, neighbor in enumerate(self.model.grid.neighbor_iter(self.pos)):
            self.scores[num] = 0
            if random.random() < self.model.probability_playing:
                self.scores[num] += self.model.payoff[self.move, neighbor.move]
        self.total_score = sum(self.scores)

    def implement_strategy(self):
        self.strategy = self.new_strategy
        self.probabilities = self.new_probabilities
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



