from mesa import Agent
import random
from itertools import combinations
import math
from utils.logger import logger
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


class Agent(Agent):
    def __init__(self, unique_id, model):
        """
        Play explanation:
        1 - paper (red), 2 - rock (grey), 3 - scissors (cyan)
            1 beats 2 but loses to 3
            2 beats 3 but loses to 1
            3 beats 1 but loses to 2
        Score is the running total of the agents success.
        """
        super().__init__(unique_id, model)
        self.pos = reverse_key(self.unique_id)
        # FIXME: the play should be based on the strategy
        self.score = 0
        self.strategy = random.choice(["Pure Rock", "Pure Paper", "Pure Scissors", "Perfect Mixed", "Imperfect Mixed"])
        # self.play = random.choice("Rock", "Paper", "Scissors")
        self.neighbours = []
        self.play = ""

    def combinations(self):
        for combo in combinations([1, 2, 3], 2):
            pass
            # output is (1, 2), (1, 3), (2, 3)

    def rock_paper_scissors(self, neighbour):
        # TODO: Should be able to shorten code using combinatorics
        # FIXME: the current sum of scores is non-zero
        if self.play == "Paper":
            if neighbour.play == "Rock":
                self.score += 1
            elif neighbour.play == "Scissors":
                self.score -= 1
        elif self.play == "Rock":
            if neighbour.play == "Paper":
                self.score -= 1
            elif neighbour.play == "Scissors":
                self.score += 1
        elif self.play == "Scissors":
            if neighbour.play == "Paper":
                self.score += 1
            elif neighbour.play == "Rock":
                self.score -= 1

    def calculate_scores(self):
        """
        Plays x rounds between agents of the game.
        :return: score of the agents
        """
        # the neighbours need to be calculated here rather than __init__() otherwise the list is incomplete
        self.neighbours = self.model.grid.get_neighbors(pos=self.pos, moore=True,
                                                        # when moore is True, diagonals are included
                                                        include_center=False)
        for neighbour in self.neighbours:
            # the pure strategies have the plays remaining unchanged so go outside the for loop
            if self.strategy == "Pure Rock":
                self.play = "Rock"
            elif self.strategy == "Pure Paper":
                self.play = "Paper"
            elif self.strategy == "Pure Scissors":
                self.play = "Scissors"
            for _ in range(self.model.num_plays_per_set):
                if self.strategy == "Perfect Mixed":
                    self.play = random.choice(["Rock", "Paper", "Scissors"])
                self.rock_paper_scissors(neighbour)
            for _ in range(self.model.num_plays_per_set):
                if self.strategy == "Imperfect Mixed":
                    pr = 0.2 #probability of strategy picking rock
                    pp = 0.3 #probability of strategy picking paper
                    ps = 0.5 #probability of strategy picking scissors
                    rand_weights = np.random.dirichlet(np.ones(3)).tolist() #random probability of given play
                    self.play = random.choice(random.choices(
                                population = ["Rock", "Paper", "Scissors"], 
                                weights = [pr, pp, ps], # rand_weights would give random weightings
                                k = 3
                                ))
                # self.evolution.evolve.mutate()

    def step(self):
        # reset scores
        self.score = 0
        self.calculate_scores()
