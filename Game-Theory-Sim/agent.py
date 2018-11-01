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
    def __init__(self, pos, model):
        super().__init__(pos, model)

        self.pos = pos
        self.score = 0
        self.strategy = None
        self.neighbours = []
        self.move = None

    def advance(self):
        self.move = self.next_move
        self.score += self.increment_score()

    def increment_score(self):
        neighbors = self.model.grid.get_neighbors(self.pos, True)
        if self.model.schedule_type == "Simultaneous":
            moves = [neighbor.next_move for neighbor in neighbors]
        else:
            moves = [neighbor.move for neighbor in neighbors]
        return sum(self.model.payoff[(self.move, move)] for move in moves)


class RPSAgent(GameAgent):
    def __init__(self, pos, model):
        super.__init__(pos, model)
        self.prob_r = 1/3
        self.prob_p = 1/3
        self.prob_s = 1/3

    def create_pure_strategies(self):
        if self.strategy == "Pure Rock":
            self.prob_r = 1
        elif self.strategy == "Pure Paper":
            self.prob_p = 1
        elif self.strategy == "Pure Scissors":
            self.prob_s = 1

    def create_strategies(self):
        if self.RPSmodel.game_type == "Pure Only":
            self.strategy = random.choice("Pure Rock", "Pure Paper", "Pure Scissors")
            self.create_pure_strategies()

        elif self.RPSmodel.game_type == "Pure and Perfect":
            self.strategy = random.choice("Pure Rock", "Pure Paper", "Pure Scissors", "Perfect Mixed")
            self.create_pure_strategies()
            if self.strategy == "Perfect Mixed":
                self.prob_r, self.prob_p, self.prob_s = 1/3, 1/3, 1/3

        elif self.RPSmodel.game_type == "Imperfect":
            self.strategy = "Imperfect Mixed"
            self.prob_r, self.prob_p, self.prob_s = np.random.dirichlet([10, 10, 10])
            rand_weights = np.random.dirichlet(np.ones(3)).tolist()  # random probability of given play
            self.play = random.choice(random.choices(population=["R", "P", "S"], weights=[self.prob_r, self.prob_p, self.prob_s]))



    def calculate_scores(self):
            for _ in range(self.model.num_plays_per_set):
                if self.strategy == "Perfect Mixed":
                    self.play = random.choice(["Rock", "Paper", "Scissors"])
                elif self.strategy == "Imperfect Mixed":
                    pr = 0.2 #probability of strategy picking rock
                    pp = 0.3 #probability of strategy picking paper
                    ps = 0.5 #probability of strategy picking scissors
                    rand_weights = np.random.dirichlet(np.ones(3)).tolist() #random probability of given play
                    self.play = random.choice(random.choices(
                                population = ["Rock", "Paper", "Scissors"], 
                                weights = [pr, pp, ps], # rand_weights would give random weightings
                                k = 3
                                ))
                self.rock_paper_scissors(neighbour)
                # self.evolution.evolve.mutate()

    def increment_score(self):
        neighbors = self.model.grid.get_neighbors(self.pos, True)
        if self.model.schedule_type == "Simultaneous":
            moves = [neighbor.next_move for neighbor in neighbors]
        else:
            moves = [neighbor.move for neighbor in neighbors]
        return sum(self.model.payoff[(self.move, move)] for move in moves)

    def step(self):
        ''' Get the neighbors' moves, and change own move accordingly. '''
        neighbors = self.model.grid.get_neighbors(self.pos, True, include_center=True)
        best_neighbor = max(neighbors, key=lambda a: a.score)
        self.next_move = best_neighbor.move

        if self.model.schedule_type != "Simultaneous":
            self.advance()

    def advance(self):
        self.move = self.next_move
        self.score += self.increment_score()


    class PDAgent(GameAgent):
        ''' Agent member of the iterated, spatial prisoner's dilemma model. '''

        def __init__(self, pos, model, starting_move=None):
            '''
            Create a new Prisoner's Dilemma agent.

            Args:
                pos: (x, y) tuple of the agent's position.
                model: model instance
                starting_move: If provided, determines the agent's initial state:
                               C(ooperating) or D(efecting). Otherwise, random.
            '''
            super().__init__(pos, model)
            self.pos = pos
            self.score = 0
            if starting_move:
                self.move = starting_move
            else:
                self.move = random.choice(["C", "D"])

            self.next_move = None

        @property
        def isCoorperating(self):
            return self.move == "C"

        def step(self):
            ''' Get the neighbors' moves, and change own move accordingly. '''
            neighbors = self.model.grid.get_neighbors(self.pos, True, include_center=True)
            best_neighbor = max(neighbors, key=lambda a: a.score)
            self.next_move = best_neighbor.move

            if self.model.schedule_type != "Simultaneous":
                self.advance()
