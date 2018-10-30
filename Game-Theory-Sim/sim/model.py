from mesa.datacollection import DataCollector
from mesa.space import SingleGrid
from mesa import Model
from mesa.time import BaseScheduler, RandomActivation, SimultaneousActivation
from .agent import RPSAgent
import numpy as np
from utils.logger import logger
import random

# TODO: Compact this code and work out how arguments work in the datacollector

def population_pure_rock(model):
    agent_strategies = [agent.strategy for agent in model.schedule.agents]
    return agent_strategies.count("Pure Rock")

def population_pure_paper(model):
    agent_strategies = [agent.strategy for agent in model.schedule.agents]
    return agent_strategies.count("Pure Paper")

def population_pure_scissors(model):
    agent_strategies = [agent.strategy for agent in model.schedule.agents]
    return agent_strategies.count("Pure Scissors")


def population_perfect_mixed(model):
    agent_strategies = [agent.strategy for agent in model.schedule.agents]
    return agent_strategies.count("Perfect Mixed")

def population_imperfect_mixed(model):
    agent_strategies = [agent.strategy for agent in model.schedule.agents]
    return agent_strategies.count("Imperfect Mixed")


class RPSGrid(Model):
    """A model with some number of agents."""
    def __init__(self, N, width, height):
        self.num_agents = N
        self.num_plays_per_set = 5
        self.grid = SingleGrid(width, height, True)
        self.schedule = BaseScheduler(self)
        # self.schedule = RandomActivation(self)
        self.running = True


        for x in range(self.grid.width):
            for y in range(self.grid.height):
                # using the Cantor pair function
                unique_id = (0.5 * (x + y) * (x + y + 1)) + y
                a = RPSAgent(unique_id, self)
                self.schedule.add(a)
                self.grid.place_agent(a, (x, y))

        self.datacollector = DataCollector(
            model_reporters={"Pure Rock": population_pure_rock, "Pure Paper": population_pure_paper, "Pure Scissors": population_pure_scissors, "Perfect Mixed": population_perfect_mixed, "Imperfect Mixed": population_imperfect_mixed},  # A function to call
            agent_reporters={"Score": "score"})  # An agent attribute

    def kill_and_reproduce(self):
        """
        Identifies the bottom 50% of poorly performing players and eliminates them from the pool.
        The strategies of these weak_players are replaced by the strongest_neighbour (the neighbour with the biggest
        score)
        :return:
        """
        # player_scores = [player.score for player in self.schedule.agents]
        # logger.info("Player score sum {}".format(sum(player_scores)))
        # num_weak_players = sum(score < -5 for score in player_scores)
        # logger.debug("Player scores {}".format(player_scores))
        #
        # for i in range(num_weak_players):
        #     logger.debug("Number of weak players {}, number of players {}".format(num_weak_players, len(player_scores)))
        #     weakest_player = self.schedule.agents[np.argmin(player_scores)]
        #     neighbour_scores = [neighbour.score for neighbour in weakest_player.neighbours]
        #     strongest_neighbour = weakest_player.neighbours[np.argmax(neighbour_scores)]
        #     # FIXME: Currently the strongest neighbour is not finding the correct answer
        #     logger.debug("Weakest player {} with position {}, Strongest neighbour {}".format(weakest_player.score, weakest_player.pos, strongest_neighbour.score))
        #     logger.debug("Neighbour positions {}".format([neighbour.score for neighbour in weakest_player.neighbours]))
        #     # FIXME: On the second step the simulation crashes and the weakest_player cannot be found
        #     # TODO: Check that this code does indeed remove the worst player
        #     player_scores.remove(weakest_player.score)
        #     weakest_player.strategy = strongest_neighbour.strategy

        weak_player_scores = [player.score for player in self.schedule.agents if player.score < -5]
        weak_players = [player for player in self.schedule.agents if player.score < -5]

        while weak_players:
            weakest_player = weak_players[np.argmin(weak_player_scores)]
            neighbour_scores = [neighbour.score for neighbour in weakest_player.neighbours]
            strongest_neighbour = weakest_player.neighbours[np.argmax(neighbour_scores)]
            logger.debug("Weakest player {} with position {}, Strongest neighbour {}".format(weakest_player.score,
                                                                                             weakest_player.pos,
                                                                                             strongest_neighbour.score))
            logger.debug("Neighbour positions {}".format([neighbour.score for neighbour in weakest_player.neighbours]))
            weakest_player.strategy = strongest_neighbour.strategy
            weak_player_scores.remove(weakest_player.score)
            weak_players.remove(weakest_player)

    def step(self):
        self.kill_and_reproduce()
        self.datacollector.collect(self)
        self.schedule.step()


class Grid(Model):
    ''' Model class for iterated, spatial prisoner's dilemma model. '''

    schedule_types = {"Sequential": BaseScheduler, "Random": RandomActivation, "Simultaneous": SimultaneousActivation}

    # This dictionary holds the payoff for the agent that makes the first move in the key
    # keyed on: (my_move, other_move)

    payoff_PD = {("C", "C"): 1, ("C", "D"): 0, ("D", "C"): 2, ("D", "D"): 0}

    payoff_RPS = {("R", "R"): 0, ("R", "P"): -1, ("R", "S"): 1, ("P", "R"): 1, ("P", "P"): 0,
                  ("P", "S"): -1, ("S", "R"): -1, ("S", "P"): 1, ("S", "S"): 0}

    def __init__(self, height=50, width=50, schedule_type="Random", payoffs=None, seed=None):
        '''
        Create a new Spatial Game Model

        Args:
            height, width: Grid size. There will be one agent per grid cell.
            schedule_type: Can be "Sequential", "Random", or "Simultaneous".
                           Determines the agent activation regime.
            payoffs: (optional) Dictionary of (move, neighbor_move) payoffs.
        '''
        self.grid = SingleGrid(height, width, torus=True)
        self.schedule_type = schedule_type
        self.schedule = self.schedule_types[self.schedule_type](self)
        self.game = "RPS"

        # Create agents
        for x in range(width):
            for y in range(height):
                agent = Agent((x, y), self)
                self.grid.place_agent(agent, (x, y))
                self.schedule.add(agent)

        if self.game == "PD":
            self.datacollector = DataCollector(
                {"Cooperating_Agents": lambda m: len([a for a in m.schedule.agents if a.move == "C"])})
        elif self.game == "RPS":
            self.datacollector = DataCollector(
                {"Pure Rock": lambda m: len([a for a in m.schedule.agents if a.strategy == "Pure Rock"]),
                 "Pure Paper": lambda m: len([a for a in m.schedule.agents if a.strategy == "Pure Paper"]),
                 "Pure Scissors": lambda m: len([a for a in m.schedule.agents if a.strategy == "Pure Scissors"]),
                 "Perfect Mixed": lambda m: len([a for a in m.schedule.agents if a.strategy == "Perfect Mixed"]),
                 "Imperfect Mixed": lambda m: len([a for a in m.schedule.agents if a.strategy == "Imperfect Mixed"])})

        self.running = True
        self.datacollector.collect(self)

    def step(self):
        self.schedule.step()
        # collect data
        self.datacollector.collect(self)

    def run(self, n):
        ''' Run the model for n steps. '''
        for _ in range(n):
            self.step()