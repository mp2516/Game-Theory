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
        self.probabilities = []
        # an empty strategy

        if self.model.game_type == "RPS":
            self.probabilities = [0, 0, 0]

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

        elif self.model.game_type == "PD":
            self.probabilities = [0, 0]

            if self.model.game_mode == "Pure Only":
                self.strategy = random.choice(["Pure Cooperating", "Pure Defecting"])
                if self.strategy == "Pure Cooperating":
                    self.probabilities = [1, 0]
                elif self.strategy == "Pure Defecting":
                    self.probabilities = [0, 1]
            self.moves = np.random.choice(a=["C", "D"], size=self.model.num_moves_per_set, p=self.probabilities)

    def step(self):
        ''' Get the neighbors' moves, and change own move accordingly. '''
        self.implement_strategies()
        self.increment_score()

    def implement_strategies(self):
        if self.model.game_type == "RPS":
            self.probabilities = [0, 0, 0]

            if self.model.game_mode == "Pure Only":
                if self.strategy == "Pure Rock":
                    self.probabilities = [1, 0, 0]
                elif self.strategy == "Pure Paper":
                    self.probabilities = [0, 1, 0]
                elif self.strategy == "Pure Scissors":
                    self.probabilities = [0, 0, 1]

            elif self.model.game_mode == "Pure and Perfect":
                if self.strategy == "Pure Rock":
                    self.probabilities = [1, 0, 0]
                elif self.strategy == "Pure Paper":
                    self.probabilities = [0, 1, 0]
                elif self.strategy == "Pure Scissors":
                    self.probabilities = [0, 0, 1]
                elif self.strategy == "Perfect Mixed":
                    self.probabilities = [1/3, 1/3, 1/3]

            self.moves = np.random.choice(a=["R", "P", "S"], size=self.model.num_moves_per_set, p=self.probabilities)

        elif self.model.game_type == "PD":
            self.probabilities = [0, 0]

            if self.model.game_mode == "Pure Only":
                self.strategy = random.choice(["Pure Cooperating", "Pure Defecting"])
                if self.strategy == "Pure Cooperating":
                    self.probabilities = [1, 0]
                elif self.strategy == "Pure Defecting":
                    self.probabilities = [0, 1]
            self.moves = np.random.choice(a=["C", "D"], size=self.model.num_moves_per_set, p=self.probabilities)

    def increment_score(self):
        self.neighbours = self.model.grid.get_neighbors(self.pos, True, include_center=False)
        neighbours_moves = [neighbour.moves for neighbour in self.neighbours]
        score = 0
        for j in range(len(neighbours_moves)):
            for i in range(0, self.model.num_moves_per_set):
                score += (sum(self.model.payoff[self.moves[i], move] for move in neighbours_moves[j]))
            self.scores[j] = score
        self.total_scores = sum(self.scores)
