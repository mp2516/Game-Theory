from mesa.datacollection import DataCollector
from mesa.space import SingleGrid
from mesa import Model, Agent
from mesa.time import BaseScheduler, RandomActivation, SimultaneousActivation
from .agent import GameAgent, PD_Agent
import numpy as np
import scipy
import ternary
import random
from .logger import logger
import matplotlib as plt


class GameGrid(Model):
    ''' Model class for iterated, spatial prisoner's dilemma model. '''

    schedule_types = {"Sequential": BaseScheduler, "Random": RandomActivation, "Simultaneous": SimultaneousActivation}

    # This dictionary holds the payoff for the agent that makes the first move in the key
    # keyed on: (my_move, other_move)

    payoff_PD = {("C", "C"): 1, ("C", "D"): 0, ("D", "C"): 2, ("D", "D"): 0}

    payoff_RPS = {("R", "R"): 0, ("R", "P"): -1, ("R", "S"): 1, ("P", "R"): 1, ("P", "P"): 0,
                  ("P", "S"): -1, ("S", "R"): -1, ("S", "P"): 1, ("S", "S"): 0}

    def __init__(self, height, width, num_moves_per_set=5, game_type="RPS", game_mode="Pure Only", cull_threshold = 0.45, probability_adoption = 0.9, strength_of_adoption = 0.1, probability_mutation = 0.05):
        '''
        Create a new Spatial Game Model

        Args:
            height, width: GameGrid size. There will be one agent per grid cell.
            num_moves_per_set: The number of moves each player makes with each other before evolving
            game_type: The type of game to play
            game_mode: The mode of that game to play
            cull_threshold: The percentage of the population that will mutate their strategy to a better one (if it exists) each step
        '''
        self.height = height
        self.width = width
        self.grid = SingleGrid(self.height, self.width, torus=True)
        self.num_moves_per_set = num_moves_per_set
        self.game_type = game_type
        self.schedule = RandomActivation(self)
        self.running = True
        self.game_mode = game_mode
        self.cull_threshold = cull_threshold
        self.probability_adoption = probability_adoption
        self.strength_of_adoption = strength_of_adoption
        self.probability_mutation = probability_mutation
        self.num_agents_cull = int(self.cull_threshold * self.height * self.width)

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
                     "Pure Scissors": lambda m: len([a for a in m.schedule.agents if a.strategy == "Pure Scissors"]),
                     "Perfect Mixed": lambda m: len([a for a in m.schedule.agents if a.strategy == "Perfect Mixed"])})

        if self.game_type == "PD":
            self.payoff = self.payoff_PD

            # Create agents
            for x in range(self.width):
                for y in range(self.height):
                    agent = GameAgent(self)
                    self.grid.place_agent(agent, (x, y))
                    self.schedule.add(agent)

            self.datacollector = DataCollector(
                {"Cooperating": lambda m: len([a for a in m.schedule.agents if a.strategy == "Pure Cooperating"]),
                 "Defecting": lambda m: len([a for a in m.schedule.agents if a.strategy == "Pure Defecting"])})

            self.datacollector.collect(self)

    def kill_and_reproduce(self):
        """
        Identifies the bottom 50% of poorly performing players and eliminates them from the pool.
        The strategies of these weak_players are replaced by the strongest_neighbour (the neighbour with the biggest
        score)
        :return:
        """
        agents = [player for player in self.schedule.agents]
        # sorts the list in ascending order by the total score of the agent
        agents_sorted = sorted(agents, key=lambda a: a.total_score)
        worst_agents = agents_sorted[:self.num_agents_cull]

        for bad_agent in worst_agents:
            strongest_neighbour = bad_agent.neighbours[np.argmax(bad_agent.scores)]

            if random.random <= self.probability_adoption:
                if self.game_mode == "Imperfect":
                    for i in bad_agent.probabilities:
                        for j in strongest_neighbour.probabilities:
                            # the bad_agents probabilities will tend towards the probabilities of the strongest_neighbour
                            # with the strength_of_adoption dictating how much it tends towards
                            bad_agent.probabilities[i] = i + ((j - i) * self.strength_of_adoption)
                elif self.game_mode == "Pure Only" or self.game_mode == "Pure and Perfect":
                    bad_agent.strategy = strongest_neighbour.strategy


    def fft_analysis(self, population_data):
        N = len(population_data)
        t_axis = np.linspace(0.0, 1.0 / (2.0), int(N) / 2)
        y_axis = population_data - np.mean(population_data)
        y_axis_fft = scipy.fftpack.fft(y_axis)
        y_corrected = 2 / N * np.abs(y_axis_fft[0:np.int(N / 4)])
        t_corrected = t_axis[0:np.int(N / 4)]



        plt.figure(1, )
        plt.plot(t_corrected, y_corrected, label='Dominant frequency = ' + str(round(xf[np.argmax(yf)], 4)) + ' $set^(-1)$')
        plt.xlabel('Frequency (set^-1)')
        plt.ylabel('FT of Population')
        plt.title('Rock Frequency Domain RPS')
        plt.legend(loc='best')

        plt.figure(2, )
        plt.plot(np.arange(N), r_ar)
        plt.xlabel('Set no')
        plt.ylabel('Rock population')
        plt.title('Rock population with sets')
        plt.show()

        figure, tax = ternary.figure(scale=1.0)
        tax.boundary()
        tax.gridlines(multiple=0.2, color="black")
        tax.set_title("Populations", fontsize=20)
        tax.left_axis_label("Scissors", fontsize=20)
        tax.right_axis_label("Paper", fontsize=20)
        tax.bottom_axis_label("Rock", fontsize=20)

        r_list_norm = [i / (l * l) for i in rock_list]
        p_list_norm = [i / (l * l) for i in paper_list]
        s_list_norm = [i / (l * l) for i in scissors_list]
        points = list(zip(r_list_norm, p_list_norm, s_list_norm))

        tax.plot(points, linewidth=2.0, label="Curve")
        tax.ticks(axis='lbr', multiple=0.2, linewidth=1)
        tax.legend()
        tax.show()

        print(np.argmax(yf))
        print("Dominant frequecy >> ", xf[np.argmax(yf)])

        raise "counter exceeded"

    def step(self):
        self.schedule.step()
        self.kill_and_reproduce()
        # collect data
        self.datacollector_populations.collect(self)


    def run(self, n):
        ''' Run the model for n steps. '''
        for _ in range(n):
            self.step()


class RPS_Model(GameGrid):
    def __init__(self, game_mode):
        super().__init__()
        self.payoff = self.payoff_RPS
        self.game_mode = game_mode

        # Create agents
        for x in range(self.width):
            for y in range(self.height):
                agent = GameAgent(self)
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