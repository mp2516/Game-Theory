from mesa.datacollection import DataCollector
from mesa.space import SingleGrid
from mesa import Model
from mesa.time import RandomActivation
from .agent import RPSAgent, PDAgent
import numpy as np
import random
from .logger import logger

def key(x, y):
    """
    Args:
        x: x value
        y: y value
    Returns:
        Cantor pair function of x and y (maps two integers to one).
    Notes:
        See: https://en.wikipedia.org/wiki/Pairing_function#Cantor_pairing_function
    """
    return int(0.5 * (x + y) * (x + y + 1) + y)


class GameGrid(Model):
    ''' Model class for iterated, spatial prisoner's dilemma model. '''

    # This dictionary holds the payoff for the agent that makes the first move in the key
    # keyed on: (my_move, other_move)

    payoff_PD = {("C", "C"): 1, ("C", "D"): 0, ("D", "C"): 2, ("D", "D"): 0}

    payoff_RPS = {("R", "R"): 0, ("R", "P"): -1, ("R", "S"): 1, ("P", "R"): 1, ("P", "P"): 0,
                  ("P", "S"): -1, ("S", "R"): -1, ("S", "P"): 1, ("S", "S"): 0}

    def __init__(self, config):
        '''
        Create a new Spatial Game Model

        Args:
            self.dimension: GameGrid size. There will be one agent per grid cell.
            self.num_moves_per_set: The number of moves each player makes with each other before evolving
            self.game_type: The type of game to play
            self.game_mode: The mode of that game to play
            self.cull_score: The minimum score a player must achieve in order to survive
        '''
        super().__init__()

        self.dimension = config.dimension
        self.grid = SingleGrid(self.dimension, self.dimension, torus=True)

        self.num_moves_per_set = config.num_moves_per_set
        self.game_type = config.game_type
        self.game_mode = config.game_mode

        self.initial_population_sizes = config.initial_population_sizes
        self.biomes = config.biomes
        self.biome_size = config.biome_percentage_size * self.dimension

        self.cull_score = config.cull_score
        self.probability_adoption = config.probability_adoption
        self.strength_of_adoption = config.strength_of_adoption
        self.probability_mutation = config.probability_mutation
        self.strength_of_mutation = config.strength_of_mutation

        self.schedule = RandomActivation(self)
        self.running = True

        self.datacollector_populations = DataCollector()
        self.datacollector_probabilities = DataCollector()


    def step(self):
        self.schedule.step()
        # collect data
        if self.game_mode == "Pure":
            self.datacollector_populations.collect(self)

    def run(self, n):
        ''' Run the model for n steps. '''
        for _ in range(n):
            self.step()


class RPSModel(GameGrid):
    def __init__(self, config):
        super().__init__(config)
        self.payoff = self.payoff_RPS
        self.num_pure_rock = []
        self.num_pure_paper = []
        self.num_pure_scissors = []
        self.num_perfect = []

        # Create agents
        for x in range(self.dimension):
            for y in range(self.dimension):
                agent = RPSAgent(self)
                self.grid.place_agent(agent, (x, y))
                self.schedule.add(agent)

        #if self.biomes:
        #    x_rand = random.randint(0, self.dimension)
        #    y_rand = random.randint(0, self.dimension)

            # for x in range(x_rand - self.biome_size, x_rand + self.biome_size):
            #     for y in range(y_rand - self.biome_size, y_rand + self.biome_size):

        if self.game_mode == "Pure":
            self.datacollector_populations = DataCollector(
                {"Pure Rock": lambda m: sum(1 for a in m.schedule.agents if a.strategy == "all_r"),
                 "Pure Paper": lambda m: sum(1 for a in m.schedule.agents if a.strategy == "all_p"),
                 "Pure Scissors": lambda m: sum(1 for a in m.schedule.agents if a.strategy == "all_s")})
            self.datacollector_populations.collect(self)

        elif self.game_mode == "Imperfect":
            self.datacollector_probabilities = DataCollector(
                {"Rock Probabilities": lambda m: (a.probabilities[0] for a in m.schedule.agents),
                 "Paper Probabilities": lambda m: (a.probabilities[1] for a in m.schedule.agents),
                 "Scissors Probabilities": lambda m: (a.probabilities[2] for a in m.schedule.agents)})
            self.datacollector_probabilities.collect(self)


class PDModel(GameGrid):
    def __init__(self, config):
        super().__init__(config)
        self.payoff = self.payoff_PD

        # Create agents
        for x in range(self.dimension):
            for y in range(self.dimension):
                agent = PDAgent(self)
                self.grid.place_agent(agent, (x, y))
                self.schedule.add(agent)

        self.datacollector_populations = DataCollector(
            {"Cooperating": lambda m: sum(1 for a in m.schedule.agents if a.strategy == "all_c"),
             "Defecting": lambda m: sum(1 for a in m.schedule.agents if a.strategy == "all_d"),
             "tit_for_tat": lambda m: sum(1 for a in m.schedule.agents if a.strategy == "tit_for_tat"),
             "spiteful": lambda m: sum(1 for a in m.schedule.agents if a.strategy == "spiteful"),
             "Random": lambda m: sum(1 for a in m.schedule.agents if a.strategy == "random")})
        self.datacollector_populations.collect(self)