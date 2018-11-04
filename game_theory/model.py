from mesa.datacollection import DataCollector
from mesa.space import SingleGrid
from mesa import Model, Agent
from mesa.time import BaseScheduler, RandomActivation, SimultaneousActivation
from .agent import RPS_Agent, PD_Agent
import numpy as np
from .logger import logger


# # TODO: Compact this code and work out how arguments work in the datacollector
#
# def population_pure_rock(model):
#     agent_strategies = [agent.strategy for agent in model.schedule.agents]
#     return agent_strategies.count("Pure Rock")
#
# def population_pure_paper(model):
#     agent_strategies = [agent.strategy for agent in model.schedule.agents]
#     return agent_strategies.count("Pure Paper")
#
# def population_pure_scissors(model):
#     agent_strategies = [agent.strategy for agent in model.schedule.agents]
#     return agent_strategies.count("Pure Scissors")
#
#
# def population_perfect_mixed(model):
#     agent_strategies = [agent.strategy for agent in model.schedule.agents]
#     return agent_strategies.count("Perfect Mixed")
#
# def population_imperfect_mixed(model):
#     agent_strategies = [agent.strategy for agent in model.schedule.agents]
#     return agent_strategies.count("Imperfect Mixed")
#
#
# class RPS_Model(Model):
#     """A model with some number of agents."""
#     def __init__(self, N, width, height):
#         self.num_agents = N
#         self.num_plays_per_set = 5
#         self.grid = SingleGrid(width, height, True)
#         self.schedule = BaseScheduler(self)
#         # self.schedule = RandomActivation(self)
#         self.running = True
#
#
#         for x in range(self.grid.width):
#             for y in range(self.grid.height):
#                 # using the Cantor pair function
#                 unique_id = (0.5 * (x + y) * (x + y + 1)) + y
#                 a = RPS_Agent(unique_id, self)
#                 self.schedule.add(a)
#                 self.grid.place_agent(a, (x, y))
#
#         self.datacollector = DataCollector(
#             model_reporters={"Pure Rock": population_pure_rock, "Pure Paper": population_pure_paper, "Pure Scissors": population_pure_scissors, "Perfect Mixed": population_perfect_mixed, "Imperfect Mixed": population_imperfect_mixed},  # A function to call
#             agent_reporters={"Score": "score"})  # An agent attribute
#
#     def kill_and_reproduce(self):
#         """
#         Identifies the bottom 50% of poorly performing players and eliminates them from the pool.
#         The strategies of these weak_players are replaced by the strongest_neighbour (the neighbour with the biggest
#         score)
#         :return:
#         """
#         # player_scores = [player.score for player in self.schedule.agents]
#         # logger.info("Player score sum {}".format(sum(player_scores)))
#         # num_weak_players = sum(score < -5 for score in player_scores)
#         # logger.debug("Player scores {}".format(player_scores))
#         #
#         # for i in range(num_weak_players):
#         #     logger.debug("Number of weak players {}, number of players {}".format(num_weak_players, len(player_scores)))
#         #     weakest_player = self.schedule.agents[np.argmin(player_scores)]
#         #     neighbour_scores = [neighbour.score for neighbour in weakest_player.neighbours]
#         #     strongest_neighbour = weakest_player.neighbours[np.argmax(neighbour_scores)]
#         #     # FIXME: Currently the strongest neighbour is not finding the correct answer
#         #     logger.debug("Weakest player {} with position {}, Strongest neighbour {}".format(weakest_player.score, weakest_player.pos, strongest_neighbour.score))
#         #     logger.debug("Neighbour positions {}".format([neighbour.score for neighbour in weakest_player.neighbours]))
#         #     # FIXME: On the second step the simulation crashes and the weakest_player cannot be found
#         #     # TODO: Check that this code does indeed remove the worst player
#         #     player_scores.remove(weakest_player.score)
#         #     weakest_player.strategy = strongest_neighbour.strategy
#
#         weak_player_scores = [player.score for player in self.schedule.agents if player.score < -5]
#         weak_players = [player for player in self.schedule.agents if player.score < -5]
#
#         while weak_players:
#             weakest_player = weak_players[np.argmin(weak_player_scores)]
#             neighbour_scores = [neighbour.score for neighbour in weakest_player.neighbours]
#             strongest_neighbour = weakest_player.neighbours[np.argmax(neighbour_scores)]
#             logger.debug("Weakest player {} with position {}, Strongest neighbour {}".format(weakest_player.score,
#                                                                                              weakest_player.pos,
#                                                                                              strongest_neighbour.score))
#             logger.debug("Neighbour positions {}".format([neighbour.score for neighbour in weakest_player.neighbours]))
#             weakest_player.strategy = strongest_neighbour.strategy
#             weak_player_scores.remove(weakest_player.score)
#             weak_players.remove(weakest_player)
#
#     def step(self):
#         self.kill_and_reproduce()
#         self.datacollector.collect(self)
#         self.schedule.step()

def number_strategy(model, strategy):
    return sum([1 for a in model.schedule.agents if a.strategy is strategy])

def number_cooperating(model):
    return number_strategy(model, "C")

def number_defecting(model):
    return number_strategy(model, "D")

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
            schedule_type: Can be "Sequential", "Random", or "Simultaneous".
                           Determines the agent activation regime.
            payoffs: (optional) Dictionary of (move, neighbor_move) payoffs.
        '''
        self.height = config.system['height']
        self.width = config.system['width']
        self.grid = SingleGrid(self.height, self.width, torus=True)
        self.num_moves_per_set = config.game['num_moves_per_set']
        self.game_type = config.game['game_type']
        self.schedule_type = config.game['scheduler']
        self.schedule = self.schedule_types[self.schedule_type](self)
        self.running = False

    def step(self):
        self.schedule.step()
        # collect data
        self.datacollector.collect(self)

    def run(self, n):
        ''' Run the model for n steps. '''
        for _ in range(n):
            self.step()


class RPS_Model(GameGrid):
    def __init__(self, config):
        super().__init__(config)
        self.payoff = self.payoff_RPS
        self.game_mode = config.game['game_mode']

        # Create agents
        for x in range(self.width):
            for y in range(self.height):
                agent = RPS_Agent((x, y), self, config)
                self.grid.place_agent(agent, (x, y))
                self.schedule.add(agent)

        self.datacollector = DataCollector(
            {"Pure Rock": lambda m: len([a for a in m.schedule.agents if a.strategy == "Pure Rock"]),
             "Pure Paper": lambda m: len([a for a in m.schedule.agents if a.strategy == "Pure Paper"]),
             "Pure Scissors": lambda m: len([a for a in m.schedule.agents if a.strategy == "Pure Scissors"]),
             "Perfect Mixed": lambda m: len([a for a in m.schedule.agents if a.strategy == "Perfect Mixed"]),
             "Imperfect Mixed": lambda m: len([a for a in m.schedule.agents if a.strategy == "Imperfect Mixed"])})

        self.datacollector.collect(self)

class PD_Model(GameGrid):
    def __init__(self, config):
        super().__init__(config)
        self.payoff = self.payoff_PD

        # Create agents
        for x in range(self.width):
            for y in range(self.height):
                agent = PD_Agent((x, y), self)
                self.grid.place_agent(agent, (x, y))
                self.schedule.add(agent)

        self.datacollector = DataCollector(
            {"Cooperating": lambda m: len([a for a in m.schedule.agents if a.strategy == "C"]),
             "Defecting": lambda m: len([a for a in m.schedule.agents if a.strategy == "D"])})

        self.datacollector.collect(self)