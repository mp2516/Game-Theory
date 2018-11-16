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
        self.total_scores = 0
        self.move, self.next_move = None, None
        self.neighbours = []
        self.new_probabilities = [1/3, 1/3, 1/3]
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
        self.increment_score()
        self.evolve_strategies()
        self.implement_strategies()

    def increment_score(self):
        self.neighbours = self.model.grid.get_neighbors(self.pos, True, include_center=False)
        neighbours_moves = [neighbour.moves for neighbour in self.neighbours]
        score = 0
        for j in range(len(neighbours_moves)):
            for i in range(0, self.model.num_moves_per_set):
                score += (sum(self.model.payoff[self.moves[i], move] for move in neighbours_moves[j]))
            self.scores[j] = score
        self.total_scores = sum(self.scores)

    def evolve_strategies(self):
        """
        Identifies the bottom 50% of poorly performing players and eliminates them from the pool.
        The strategies of these weak_players are replaced by the strongest_neighbour (the neighbour with the biggest
        score)
        :return:
        """
        for neighbour in self.neighbours:
            if self.total_scores < neighbour.total_scores:
                strongest_neighbour = self.neighbours[np.argmax([neighbour.total_scores for neighbour in self.neighbours])]
                if random.random() <= self.model.probability_adoption:
                    if self.model.game_mode == "Imperfect":
                        for num, i in enumerate(self.probabilities):
                            for j in strongest_neighbour.probabilities:
                                # the bad_agents probabilities will tend towards the probabilities of the strongest_neighbour
                                # with the strength_of_adoption dictating how much it tends towards
                                self.new_probabilities[num] = i + ((j - i) * self.model.strength_of_adoption)
                    elif self.model.game_mode == "Pure Only" or self.model.game_mode == "Pure and Perfect":
                        self.new_strategy = strongest_neighbour.strategy

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
                self.probabilities = [prob / sum(self.probabilities) for prob in self.probabilities]

            self.moves = np.random.choice(a=["R", "P", "S"], size=self.model.num_moves_per_set, p=self.probabilities)

        elif self.model.game_type == "PD":
            if self.strategy == "Pure Cooperating":
                self.probabilities = [1, 0]
            elif self.strategy == "Pure Defecting":
                self.probabilities = [0, 1]
            self.moves = np.random.choice(a=["C", "D"], size=self.model.num_moves_per_set, p=self.probabilities)
