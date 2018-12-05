from mesa.datacollection import DataCollector
from mesa.space import SingleGrid
from mesa import Model
from mesa.time import BaseScheduler, RandomActivation, SimultaneousActivation
from .agent import GameAgent
import numpy as np
import random
from .logger import logger

game_no = []

class GameGrid(Model):
    ''' Model class for iterated, spatial prisoner's dilemma model. '''

    schedule_types = {"Sequential": BaseScheduler, "Random": RandomActivation, "Simultaneous": SimultaneousActivation}

    # This dictionary holds the payoff for the agent that makes the first move in the key
    # keyed on: (my_move, other_move)

    payoff_PD = {("C", "C"): 1, ("C", "D"): 0, ("D", "C"): 2, ("D", "D"): 0}

    payoff_RPS = {("R", "R"): 0, ("R", "P"): -1, ("R", "S"): 1, ("P", "R"): 1, ("P", "P"): 0,
                  ("P", "S"): -1, ("S", "R"): -1, ("S", "P"): 1, ("S", "S"): 0, ("E", "R"): 0,
                  ("E", "P"): 0, ("E", "S"): 0, ("R", "E"): 0, ("P", "E"): 0, ("S", "E"): 0,
                  ("E", "E"): 0}

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
        self.running = True
        self.step_num = 0
        self.height = config.height
        self.width = config.width
        self.grid = SingleGrid(self.height, self.width, torus=True)

        self.num_moves_per_set = config.num_moves_per_set
        self.number_of_steps = config.number_of_steps
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
            self.num_empty = []

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
                     "Perfect Mixed": lambda m: sum(1 for a in m.schedule.agents if a.strategy == "Perfect Mixed"),
                     "Empty": lambda m: sum(1 for a in m.schedule.agents if a.strategy == "Empty")})
                self.datacollector_populations.collect(self)

            elif self.game_mode == "Imperfect":
                self.datacollector_probabilities = DataCollector(
                    {"Rock Probabilities": lambda m: (a.probabilities[0] for a in m.schedule.agents),
                     "Paper Probabilities": lambda m: (a.probabilities[1] for a in m.schedule.agents),
                     "Scissors Probabilities": lambda m: (a.probabilities[2] for a in m.schedule.agents)})
                self.datacollector_probabilities.collect(self)
                
            if self.game_mode == "Pure Only" or self.game_mode == "Pure and Perfect":
                self.datacollector_population_scores = DataCollector(
                        {"Pure Rock Scores": lambda m: [a.total_score for a in m.schedule.agents if a.strategy == "Pure Rock"],
                         "Pure Paper Scores": lambda m: [a.total_score for a in m.schedule.agents if a.strategy == "Pure Paper"],
                         "Pure Scissors Scores": lambda m: [a.total_score for a in m.schedule.agents if a.strategy == "Pure Scissors"],
                         "Perfect Mixed Scores": lambda m: [a.total_score for a in m.schedule.agents if a.strategy == "Perfect Mixed"]})
#                self.datacollector_population_scores.collect(self)
            
            
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
            
            

    def step(self):           
        self.schedule.step()
        self.step_num += 1
        self.num_mutating_agents = 0
#            agent.evolve_strategies()

#        if self.step_num % 2 == 0:
#            for agent in self.schedule.agents:
#                agent.increment_score()
#            for agent in self.schedule.agents:
#                agent.kill_weak()
##        for agent in self.schedule.agents:
#
#        elif self.step_num % 2 == 1:
#            for agent in self.schedule.agents:
#                agent.reproduce()
#        for agent in self.schedule.agents:
#            agent.implement_strategies()
        
        for agent in self.schedule.agents:
            agent.increment_score()
        for agent in self.schedule.agents:
            agent.kill_weak()
        for agent in self.schedule.agents:
            agent.implement_strategies()
        for agent in self.schedule.agents:
            agent.reproduce()
        for agent in self.schedule.agents:
            agent.implement_strategies()
            
#        self.datacollector_scores.collect(self)
#        self.datacollector_mutating_agents.collect(self)
        # collect data
        if self.game_mode == "Pure Only" or self.game_mode == "Pure and Perfect":
            self.datacollector_populations.collect(self)
        if self.game_mode == "Pure Only" or self.game_mode == "Pure and Perfect":
            self.datacollector_population_scores.collect(self)

    def run(self, n):
        ''' Run the model for n steps. '''
        for _ in range(n):
            self.step()