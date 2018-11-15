from mesa.datacollection import DataCollector
from mesa.space import SingleGrid
from mesa import Model
from mesa.time import BaseScheduler, RandomActivation, SimultaneousActivation
from .agent import GameAgent
import numpy as np
import random
from .logger import logger


class GameGrid(Model):
    ''' Model class for iterated, spatial prisoner's dilemma model. '''

    schedule_types = {"Sequential": BaseScheduler, "Random": RandomActivation, "Simultaneous": SimultaneousActivation}

    # This dictionary holds the payoff for the agent that makes the first move in the key
    # keyed on: (my_move, other_move)

    payoff_PD = {("C", "C"): 1, ("C", "D"): 0, ("D", "C"): 2, ("D", "D"): 0}

    payoff_RPS = {("R", "R"): 0, ("R", "P"): -1, ("R", "S"): 1, ("P", "R"): 1, ("P", "P"): 0,
                  ("P", "S"): -1, ("S", "R"): -1, ("S", "P"): 1, ("S", "S"): 0}

    def __init__(self, config):
        '''
        Create a new Spatial Game Model

        Args:
            height, width: GameGrid size. There will be one agent per grid cell.
            num_moves_per_set: The number of moves each player makes with each other before evolving
            game_type: The type of game to play
            game_mode: The mode of that game to play
            cull_threshold: The percentage of the population that will mutate their strategy to a better one (if it exists) each step
        '''
        self.height = config.height
        self.width = config.width
        self.grid = SingleGrid(self.height, self.width, torus=True)

        self.num_moves_per_set = config.num_moves_per_set
        self.game_type = config.game_type
        self.game_mode = config.game_mode

        self.cull_threshold = config.cull_threshold
        self.num_agents_cull = int(self.cull_threshold * self.height * self.width)
        self.probability_adoption = config.probability_adoption
        self.strength_of_adoption = config.strength_of_adoption
        self.probability_mutation = config.probability_mutation
        self.strength_of_mutation = config.strength_of_mutation

        self.schedule = RandomActivation(self)
        self.running = True

        if self.game_type == "RPS":
            self.payoff = self.payoff_RPS
            self.num_pure_rock = []
            self.num_pure_paper = []
            self.num_pure_scissors = []
            self.num_perfect = []

            # Create agents
            for x in range(self.width):
                for y in range(self.height):
                    agent = GameAgent(self)
                    self.grid.place_agent(agent, (x, y))
                    self.schedule.add(agent)

            if self.game_mode == "Pure Only" or self.game_mode == "Pure and Perfect":
                self.datacollector_populations = DataCollector(
                    {"Pure Rock": lambda m: sum(1 for a in m.schedule.agents if a.strategy == "Pure Rock"),
                     "Pure Paper": lambda m: sum(1 for a in m.schedule.agents if a.strategy == "Pure Paper"),
                     "Pure Scissors": lambda m: sum(1 for a in m.schedule.agents if a.strategy == "Pure Scissors"),
                     "Perfect Mixed": lambda m: sum(1 for a in m.schedule.agents if a.strategy == "Perfect Mixed")})
                self.datacollector_populations.collect(self)

            elif self.game_mode == "Imperfect":
                self.datacollector_probabilities = DataCollector(
                    {"Rock Probabilities": lambda m: (a.probabilities[0] for a in m.schedule.agents),
                     "Paper Probabilities": lambda m: (a.probabilities[1] for a in m.schedule.agents),
                     "Scissors Probabilities": lambda m: (a.probabilities[2] for a in m.schedule.agents)})
                self.datacollector_probabilities.collect(self)

        if self.game_type == "PD":
            self.payoff = self.payoff_PD

            # Create agents
            for x in range(self.width):
                for y in range(self.height):
                    agent = GameAgent(self)
                    self.grid.place_agent(agent, (x, y))
                    self.schedule.add(agent)

            self.datacollector_populations = DataCollector(
                {"Cooperating": lambda m: sum(1 for a in m.schedule.agents if a.strategy == "Pure Cooperating"),
                 "Defecting": lambda m: sum(1 for a in m.schedule.agents if a.strategy == "Pure Defecting")})
            self.datacollector_populations.collect(self)


    def kill_and_reproduce(self):
        """
        Identifies the bottom 50% of poorly performing players and eliminates them from the pool.
        The strategies of these weak_players are replaced by the strongest_neighbour (the neighbour with the biggest
        score)
        :return:
        """
        agents = [player for player in self.schedule.agents]
        # sorts the list in ascending order by the total score of the agent
        agents_sorted = sorted(agents, key=lambda a: a.total_scores)
        worst_agents = agents_sorted[:self.num_agents_cull]

        for bad_agent in worst_agents:
            strongest_neighbour = bad_agent.neighbours[np.argmax(bad_agent.scores)]

            if random.random() <= self.probability_adoption:
                if self.game_mode == "Imperfect":
                    for num, i in enumerate(bad_agent.probabilities):
                        for j in strongest_neighbour.probabilities:
                            # the bad_agents probabilities will tend towards the probabilities of the strongest_neighbour
                            # with the strength_of_adoption dictating how much it tends towards
                            bad_agent.probabilities[num] = i + ((j - i) * self.strength_of_adoption)
                elif self.game_mode == "Pure Only" or self.game_mode == "Pure and Perfect":
                    # logger.debug("Replacing the bad agent {} with strategy {} with the {} strategy of its strongest neighbour".format(bad_agent.unique_id, bad_agent.strategy, strongest_neighbour.strategy))
                    bad_agent.strategy = strongest_neighbour.strategy


    def step(self):
        self.schedule.step()
        # collect data
        if self.game_mode == "Pure Only" or self.game_mode == "Pure and Perfect":
            self.datacollector_populations.collect(self)
        self.kill_and_reproduce()


    def run(self, n):
        ''' Run the model for n steps. '''
        for _ in range(n):
            self.step()