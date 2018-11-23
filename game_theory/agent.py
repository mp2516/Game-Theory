from mesa import Agent
import random
import math
from .logger import logger
import numpy as np

def reverse_key(z):
    """
    Args:
        z: the integer hash value
    Returns:
        Inverted Cantor pair function of z to give x and y
    """
    w = math.floor((math.sqrt((8 * z) + 1) - 1) / 2)
    t = (w ** 2 + w) / 2
    y = z - t
    x = w - y
    return [int(x), int(y)]

class GameAgent(Agent):
    unique_id = 1
    def __init__(self, model):
        super(GameAgent, self).__init__(GameAgent.unique_id, model)
        GameAgent.unique_id += 1
        self.pos = reverse_key(self.unique_id)
        # the length of this array is 8 as their are 8 neighbours for every agent
        self.scores = [0, 0, 0, 0, 0, 0, 0, 0]
        self.total_score = 0
        self.move, self.next_move = None, None
        self.neighbours = []
        self.probabilities = [1/3, 1/3, 1/3]
        self.new_probabilities = [1/3, 1/3, 1/3]
        self.evolved = False
        self.strategy = ""
        self.new_strategy = ""
        # an empty strategy

    def step(self):
        ''' Get the neighbors' moves, and change own move accordingly. '''
        self.neighbours = self.model.grid.get_neighbors(self.pos, True, include_center=False)
        for i in range(self.model.num_moves_per_set):
            self.play_reactive_strategies()
            self.increment_score()
        self.evolve_strategies()
        self.implement_strategies()

    def increment_score(self):
        neighbours_moves = [neighbour.moves for neighbour in self.neighbours]
        score = 0
        # if self.model.game_mode == "Pure":
        #     for j in range(len(neighbours_moves)):
        #         # iterates through all the neighbours
        #         score += sum(self.model.payoff[self.moves, moves] for moves in neighbours_moves[j])
        #     self.total_score = score
        # else:

        for j in range(len(neighbours_moves)):
            for i in range(0, self.model.num_moves_per_set):
                score += (sum(self.model.payoff[self.moves[i], move] for move in neighbours_moves[j]))
            self.scores[j] = score
        self.total_score = sum(self.scores)

    def evolve_strategies(self):
        """
        Identifies the bottom 50% of poorly performing players and eliminates them from the pool.
        The strategies of these weak_players are replaced by the strongest_neighbour (the neighbour with the biggest
        score)
        :return:
        """
        if self.total_score <= self.model.cull_score and random.random() <= self.model.probability_adoption:
            # TODO: Benchmark strongest_neighbour to test whether this part of the code is even necessary
            strongest_neighbour = self.neighbours[np.argmax([neighbour.total_score for neighbour in self.neighbours])]
            if strongest_neighbour.total_score > self.total_score:
                if self.model.game_mode == "Impure":
                    for num, i in enumerate(self.probabilities):
                        for j in strongest_neighbour.probabilities:
                            # the bad_agents probabilities will tend towards the probabilities of the strongest_neighbour
                            # with the strength_of_adoption dictating how much it tends towards
                            self.new_probabilities[num] = i + ((j - i) * self.model.strength_of_adoption)
                elif self.model.game_mode == "Pure":
                    self.new_strategy = strongest_neighbour.strategy
            else:
                self.new_strategy = self.strategy
                self.new_probabilities = self.probabilities
        else:
            self.new_strategy = self.strategy
            self.new_probabilities = self.probabilities

    def implement_strategies(self):
        # this function is overwritten in the subclasses
        pass

    def play_reactive_strategies(self):
        # this function is overwritten only in the prisoner's dilemma
        pass


class RPSAgent(GameAgent):

    def __init__(self, model):
        super(RPSAgent, self).__init__(model)
        if self.model.game_mode == "Pure":
            self.strategy = np.random.choice(a=["all_r", "all_p", "all_s"], p=self.model.initial_population_sizes)
            if self.strategy == "all_r":
                self.probabilities = [1, 0, 0]
            elif self.strategy == "all_p":
                self.probabilities = [0, 1, 0]
            elif self.strategy == "all_s":
                self.probabilities = [0, 0, 1]

        elif self.model.game_mode == "Impure":
            self.strategy = "unequal_rps"
            self.probabilities = np.random.dirichlet([10, 10, 10])

        self.moves = np.random.choice(a=["R", "P", "S"], size=self.model.num_moves_per_set, p=self.probabilities)

    def implement_strategies(self):
        self.strategy = self.new_strategy
        self.probabilities = self.new_probabilities

        if self.strategy == "all_r":
            self.probabilities = [1, 0, 0]
        elif self.strategy == "all_p":
            self.probabilities = [0, 1, 0]
        elif self.strategy == "all_s":
            self.probabilities = [0, 0, 1]
        elif self.strategy == "unequal_rps":
            # whilst the linear interpolation should lead to probabilities that sum to 1
            self.probabilities = [prob / sum(self.probabilities) for prob in self.probabilities]

        self.moves = np.random.choice(a=["R", "P", "S"], size=self.model.num_moves_per_set, p=self.probabilities)


class PDAgent(GameAgent):
    def __init__(self, model):
        super(PDAgent, self).__init__(model)
        self.previous_moves = None
        self.strategy = np.random.choice(["all_c", "all_d", "tit_for_tat", "spiteful", "random"],
                                         p=self.model.initial_population_sizes)
        if self.strategy == "all_c":
            self.moves = ["C"] * self.model.num_moves_per_set
        elif self.strategy == "all_d":
            self.moves = ["D"] * self.model.num_moves_per_set
        elif self.strategy == "random":
            self.moves = np.random.choice(a=["C", "D"], size=self.model.num_moves_per_set)

    def play_reactive_strategies(self, i, neighbour):
        if self.strategy == "tit_for_tat":
            if i == 0:
                self.moves = ["C"]
            else:
                self.moves = neighbour.previous_moves[-1]
        elif self.strategy == "spiteful":
            if "D" in neighbour.previous_moves:
                self.moves = ["D"]
            else:
                self.moves = ["C"]

    def implement_strategies(self):
        if self.strategy == "all_c":
            self.moves = ["C"] * self.model.num_moves_per_set
        elif self.strategy == "all_d":
            self.moves = ["D"] * self.model.num_moves_per_set
        elif self.strategy == "random":
            self.moves = np.random.choice(a=["C", "D"], size=self.model.num_moves_per_set)

    def increment_score(self):
        neighbours_moves = [neighbour.moves for neighbour in self.neighbours]
        score = 0
        score += (sum(self.model.payoff[self.moves[i], move] for move in neighbours_moves[j]))
        self.scores = score
        self.total_score = sum(self.scores)

    def step(self):
        self.neighbours = self.model.grid.get_neighbors(self.pos, True, include_center=False)
        for i in range(self.model.num_moves_per_set):
            for neighbour in self.neighbours:
                self.play_reactive_strategies(i, neighbour)
                self.increment_score()
        self.evolve_strategies()
        self.implement_strategies()



