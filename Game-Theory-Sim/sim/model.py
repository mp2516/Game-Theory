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

def population_imperfect_mixed(model):
    agent_strategies = [agent.strategy for agent in model.schedule.agents]
    return agent_strategies.count("Imperfect Mixed")

rock_list = []
paper_list = []
scissors_list = []
perf_mix_list = []
imp_mix_list =[]
counter = []


class Model(Model):
    """A model with some number of agents."""
    def __init__(self, N, width, height):
        self.num_agents = N
        self.num_plays_per_set = 5
        self.grid = SingleGrid(width, height, True)
        # self.schedule = BaseScheduler(self)
        self.schedule = RandomActivation(self)
        self.running = True


        for x in range(self.grid.width):
            for y in range(self.grid.height):
                # using the Cantor pair function
                unique_id = (0.5 * (x + y) * (x + y + 1)) + y
                a = Agent(unique_id, self)
                self.schedule.add(a)
                self.grid.place_agent(a, (x, y))

        self.datacollector = DataCollector(
            model_reporters={"Pure Rock": population_pure_rock, "Pure Paper": population_pure_paper, "Pure Scissors": population_pure_scissors, "Perfect Mixed": population_perfect_mixed, "Imperfect Mixed": population_imperfect_mixed},  # A function to call
            agent_reporters={"Score": "score"})  # An agent attribute
        print({"Pure Rock": population_pure_rock, "Pure Paper": population_pure_paper, "Pure Scissors": population_pure_scissors, "Perfect Mixed": population_perfect_mixed, "Imperfect Mixed": population_imperfect_mixed})
        
            
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
# =============================================================================
#             logger.debug("Weakest player {} with position {}, Strongest neighbour {}".format(weakest_player.score,
#                                                                                              weakest_player.pos,
#                                                                                              strongest_neighbour.score))
#             logger.debug("Neighbour positions {}".format([neighbour.score for neighbour in weakest_player.neighbours]))
# =============================================================================
            weakest_player.strategy = strongest_neighbour.strategy #random.choice([neighbour.strategy for neighbour in weakest_player.neighbours])
            weak_player_scores.remove(weakest_player.score)
            weak_players.remove(weakest_player)
    
    def strat_list(self):
        strat_list = [player.strategy for player in self.schedule.agents]
        rock_list.append(strat_list.count("Pure Rock"))
        paper_list.append(strat_list.count("Pure Paper"))
        scissors_list.append(strat_list.count("Pure Scissors"))
        perf_mix_list.append(strat_list.count("Perfect Mixed"))
        imp_mix_list.append(strat_list.count("Imperfect Mixed"))

        
    def fft_analysis(self):
        counter.append(1)
        if len(counter) == 480:
            
            r_ar = np.array(rock_list)            
            N = len(r_ar)
            
            x = np.linspace(0.0, 1.0/(2.0), int(N/2))            
            y = r_ar - np.mean(r_ar)
            yfft = scipy.fftpack.fft(y)
            yf = 2/N * np.abs(yfft[0:np.int(N/8)])
            xf = x[0:np.int(N/8)]
            
            plt.figure(1,)
            plt.plot(xf, yf, label= 'Dominant frequency = ' + str(round(xf[np.argmax(yf)], 4))+ ' set^-1')
            plt.xlabel('Frequency (set^-1)')
            plt.ylabel('FT of Population')
            plt.title('Rock Frequency Domain RPS')     
            plt.legend(loc='best')
            print(np.argmax(yf))
            print("Dominant frequecy >> ", xf[np.argmax(yf)])
            
            plt.figure(2,)
            plt.plot(np.arange(N), r_ar)
            plt.xlabel('Set no')
            plt.ylabel('Rock population')
            plt.title('Rock population with sets')            
            plt.show()
                        
        if len(counter) > 480:
            raise "counter exceeded"
            
        
    def step(self):
        self.kill_and_reproduce()        
        self.datacollector.collect(self)        
        self.strat_list()
        self.fft_analysis()
        self.schedule.step()
