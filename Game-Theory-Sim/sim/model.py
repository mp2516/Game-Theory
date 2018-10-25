from mesa.datacollection import DataCollector
from mesa.space import SingleGrid
from mesa import Model
from mesa.time import BaseScheduler, RandomActivation
from sim.agent import Agent
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



class Model(Model):
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
                a = Agent(unique_id, self)
                self.schedule.add(a)
                self.grid.place_agent(a, (x, y))

        self.datacollector = DataCollector(
            model_reporters={"Pure Rock": population_pure_rock, "Pure Paper": population_pure_paper, "Pure Scissors": population_pure_scissors, "Perfect Mixed": population_perfect_mixed},  # A function to call
            agent_reporters={"Score": "score"})  # An agent attribute

    def kill_and_reproduce(self):
        """
        Identifies the bottom 50% of poorly performing players and eliminates them from the pool.
        The strategies of these weak_players are replaced by the strongest_neighbour (the neighbour with the biggest
        score)
        :return:
        """
        player_scores = [player.score for player in self.schedule.agents]
        num_weak_players = sum(score < 0 for score in player_scores)
        logger.debug("Player scores {}".format(player_scores))

        for i in range(num_weak_players):
            weakest_player = self.schedule.agents[np.argmin(player_scores)]
            strongest_neighbour = self.schedule.agents[np.argmax([neighbour.score for neighbour in weakest_player.neighbours])]
            # FIXME: Currently the strongest neighbour is not finding the correct answer
            logger.debug("Weakest player {} with position {}, Strongest neighbour {}".format(weakest_player.score, weakest_player.pos, strongest_neighbour.score))
            # FIXME: On the second step the simulation crashes and the weakest_player cannot be found
            # TODO: Check that this code does indeed remove the worst player
            player_scores.remove(weakest_player.score)
            weakest_player.strategy = strongest_neighbour.strategy

    def step(self):
        self.kill_and_reproduce()
        self.datacollector.collect(self)
        self.schedule.step()