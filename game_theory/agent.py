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
        super().__init__(GameAgent.unique_id, model)
        GameAgent.unique_id += 1
        self.pos = reverse_key(self.unique_id)
        # the length of this array is 8 as their are 8 neighbours for every agent
        self.scores = [0, 0, 0, 0, 0, 0, 0, 0]
        self.total_score = 0
        self.move, self.next_move = None, None
        self.neighbours = []
        self.new_probabilities = [1/3, 1/3, 1/3]
        self.evolved = False
        self.strategy = ""
        self.new_strategy = ""
        # an empty strategy

        if self.model.game_type == "RPS":

            if self.model.game_mode == "Pure Only":
                self.strategy = random.choice(["Pure Rock", "Pure Paper", "Pure Scissors"])
                if self.strategy == "Pure Rock":
                    self.probabilities = [1, 0, 0]
                elif self.strategy == "Pure Paper":
                    self.probabilities = [0, 1, 0]
                elif self.strategy == "Pure Scissors":
                    self.probabilities = [0, 0, 1]
                self.moves = np.random.choice(a=["R", "P", "S"], size=1, p=self.probabilities)

            elif self.model.game_mode == "Pure and Perfect":
                self.strategy = random.choice(["Pure Rock", "Pure Paper", "Pure Scissors", "Perfect Mixed"])
                if self.strategy == "Pure Rock":
                    self.probabilities = [1, 0, 0]
                elif self.strategy == "Pure Paper":
                    self.probabilities = [0, 1, 0]
                elif self.strategy == "Pure Scissors":
                    self.probabilities = [0, 0, 1]
                elif self.strategy == "Perfect Mixed":
                    self.probabilities = [1/3, 1/3, 1/3]

            elif self.model.game_mode == "Imperfect":
                self.strategy = "Imperfect Mixed"
                self.probabilities = np.random.dirichlet([10, 10, 10])

            self.moves = np.random.choice(a=["R", "P", "S"], size=self.model.num_moves_per_set, p=self.probabilities)
            # a small change

        elif self.model.game_type == "PD":

            if self.model.game_mode == "Pure Only":
                self.strategy = random.choice(["Pure Cooperating", "Pure Defecting"])
                if self.strategy == "Pure Cooperating":
                    self.probabilities = [1, 0]
                elif self.strategy == "Pure Defecting":
                    self.probabilities = [0, 1]
            self.moves = np.random.choice(a=["C", "D"], size=self.model.num_moves_per_set, p=self.probabilities)

    def step(self):
        ''' Get the neighbors' moves, and change own move accordingly. '''
        self.neighbours = self.model.grid.get_neighbors(self.pos, True, include_center=False)
        self.increment_score()
        self.evolve_strategies()
        self.implement_strategies()

    def increment_score(self):
        neighbours_moves = [neighbour.moves for neighbour in self.neighbours]
        score = 0
        if self.model.game_mode == "Pure Only":
            score += (sum(self.model.payoff[self.moves[i], move] for move in neighbours_moves[j]))
            self.total_score = score
        else:
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
        if self.total_score <= self.model.cull_threshold and random.random() <= self.model.probability_adoption:
            strongest_neighbour = self.neighbours[np.argmax([neighbour.total_score for neighbour in self.neighbours])]
            if strongest_neighbour.total_score > self.total_score:
                if self.model.game_mode == "Imperfect":
                    for num, i in enumerate(self.probabilities):
                        for j in strongest_neighbour.probabilities:
                            # the bad_agents probabilities will tend towards the probabilities of the strongest_neighbour
                            # with the strength_of_adoption dictating how much it tends towards
                            self.new_probabilities[num] = i + ((j - i) * self.model.strength_of_adoption)
                elif self.model.game_mode == "Pure Only" or self.model.game_mode == "Pure and Perfect":
                    self.new_strategy = strongest_neighbour.strategy
            else:
                self.new_strategy = self.strategy
                self.new_probabilities = self.probabilities
        else:
            self.new_strategy = self.strategy
            self.new_probabilities = self.probabilities

    def implement_strategies(self):
        self.strategy = self.new_strategy
        self.probabilities = self.new_probabilities

        if self.model.game_type == "RPS":
            if self.strategy == "Pure Rock":
                self.probabilities = [1, 0, 0]
            elif self.strategy == "Pure Paper":
                self.probabilities = [0, 1, 0]
            elif self.strategy == "Pure Scissors":
                self.probabilities = [0, 0, 1]
            elif self.strategy == "Perfect Mixed":
                self.probabilities = [1/3, 1/3, 1/3]
            elif self.strategy == "Imperfect Mixed":
                # whilst the linear interpolation should lead to probabilities that sum to 1
                self.probabilities = [prob / sum(self.probabilities) for prob in self.probabilities]

            self.moves = np.random.choice(a=["R", "P", "S"], size=self.model.num_moves_per_set, p=self.probabilities)

        elif self.model.game_type == "PD":
            if self.strategy == "Pure Cooperating":
                self.probabilities = [1, 0]
            elif self.strategy == "Pure Defecting":
                self.probabilities = [0, 1]
            self.moves = np.random.choice(a=["C", "D"], size=self.model.num_moves_per_set, p=self.probabilities)
