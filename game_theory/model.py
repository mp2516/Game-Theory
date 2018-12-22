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


def biome_boundaries(initial_population_probabilities, width):
    """
    Args:
        initial_population_probabilities: the proportion of the grid that is filled with each probability
        dimension: the length along one edge of the grid
    Returns:
        A list of x values which indicate the boundaries of the biomes with the least possible error from the
        initial_population_probabilities.
    Notes:
        This is a form of allocation problem, here I have used the algorithm called the Hungarian Algorithm
        https://hackernoon.com/the-assignment-problem-calculating-the-minimum-matrix-sum-python-1bba7d15252d
    """
    exact_split = [(prob*width) for prob in initial_population_probabilities]
    probs_round = [int(np.floor(split)) for split in exact_split]
    remainder = [(exact_split[i] - prob_round) for (i, prob_round) in enumerate(probs_round)]
    while sum(probs_round) < width:
        index = int(np.argmax(remainder))
        probs_round[index] += 1
        remainder[index] = 0
    probs_round.append(0)
    probs_round.sort()
    cumulative = np.cumsum(probs_round)
    # this ensures that the strategy generation goes onto the last column in the grid
    cumulative[-1] += 1
    return cumulative


class GameGrid(Model):
    ''' Model class for iterated, spatial prisoner's dilemma model. '''

    # This dictionary holds the payoff for the agent that makes the first move in the key
    # keyed on: (my_move, other_move)

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
        
        self.diagonal_neighbours = config['diagonal_neighbours']
        self.periodic_BC = config['periodic_BC']
        
        if config['square']:
            self.dimension = config['dimension']
            self.grid = SingleGrid(self.dimension, self.dimension, torus=self.periodic_BC)
            self.height = self.dimension
            self.width = self.dimension
        else:
            self.height = config['height']
            self.width = config['width']
            self.grid = SingleGrid(self.width, self.height, torus=self.periodic_BC)
            


        self.step_num = 0
        self.num_mutating = 0
        self.fraction_mutating = 0
        self.crowded_players = []

        self.num_moves_per_set = config['num_moves_per_set']
        self.game_type = config['game_type']
        self.game_mode = config['game_mode']

        self.initial_population_sizes = config['initial_population_sizes']
        self.biomes = config['biomes']
        if self.biomes:
            self.biome_boundaries = biome_boundaries(self.initial_population_sizes, self.width)

        self.cull_score = config['cull_score']
        self.probability_adoption = config['probability_adoption']
        self.strength_of_adoption = config['strength_of_adoption']

        self.probability_mutation = config['probability_mutation']
        self.strength_of_mutation = config['strength_of_mutation']

        self.probability_exchange = config['probability_exchange']
        self.probability_playing = config['probability_playing']

        self.agent_strategies = config['agent_strategies']
        self.agent_moves = config['agent_moves']

        self.probability_cull_score_decrease = config['probability_cull_score_decrease']

        self.schedule = RandomActivation(self)
        self.running = True

        self.datacollector_populations = DataCollector()
        self.datacollector_probabilities = DataCollector()

    def run(self, n):
        ''' Run the model for n steps. '''
        for _ in range(n):
            self.step()


    @staticmethod
    def count_populations(model, agent_strategy):
        """
        Helper method to count agents with a given strategy in a given model.
        """
        count = 0
        for agent in model.schedule.agents:
            if agent.strategy == agent_strategy:
                count += 1
        return count

    @staticmethod
    def count_probabilities(model, agent_probability_index):
        """
        Helper method to count trees in a given condition in a given model.
        """
        count = 0
        for agent in model.schedule.agents:
            count += agent.probabilities[agent_probability_index]
        return count

    @staticmethod
    def count_scores(model, agent_strategy):
        """
        Helper method to count trees in a given condition in a given model.
        """
        count = 0
        for agent in model.schedule.agents:
            if agent.strategy == agent_strategy:
                count += agent.total_score
        return count
    
    @staticmethod
    def count_vortices(model):
        """
        Helper method to count trees in a given condition in a given model.
        """
        positions = []
        vortex_count = 0
        agent_list = [agent for agent in model.schedule.agents]
        if model.periodic_BC:
            for agent in model.schedule.agents:
                neighbor_list = [neighbor for neighbor in agent.neighbors]
                if agent.strategy == neighbor_list[1].strategy and agent.strategy != neighbor_list[0].strategy and agent.strategy != neighbor_list[3].strategy and neighbor_list[0].strategy != neighbor_list[3].strategy:
                    vortex_count += 1
                elif agent.strategy == neighbor_list[3].strategy and agent.strategy != neighbor_list[0].strategy and agent.strategy != neighbor_list[1].strategy and neighbor_list[0].strategy != neighbor_list[1].strategy:
                    vortex_count += 1
                elif agent.strategy == neighbor_list[3].strategy and agent.strategy != neighbor_list[6].strategy and agent.strategy != neighbor_list[5].strategy and neighbor_list[5].strategy != neighbor_list[6].strategy:
                    vortex_count += 1
                elif agent.strategy == neighbor_list[6].strategy and agent.strategy != neighbor_list[3].strategy and agent.strategy != neighbor_list[5].strategy and neighbor_list[3].strategy != neighbor_list[5].strategy:
                    vortex_count += 1
                elif agent.strategy == neighbor_list[6].strategy and agent.strategy != neighbor_list[4].strategy and agent.strategy != neighbor_list[7].strategy and neighbor_list[4].strategy != neighbor_list[7].strategy:
                    vortex_count += 1
                elif agent.strategy == neighbor_list[4].strategy and agent.strategy != neighbor_list[1].strategy and agent.strategy != neighbor_list[2].strategy and neighbor_list[1].strategy != neighbor_list[2].strategy:
                    vortex_count += 1
                elif agent.strategy == neighbor_list[4].strategy and agent.strategy != neighbor_list[6].strategy and agent.strategy != neighbor_list[7].strategy and neighbor_list[6].strategy != neighbor_list[7].strategy:
                    vortex_count += 1
                elif agent.strategy == neighbor_list[1].strategy and agent.strategy != neighbor_list[2].strategy and agent.strategy != neighbor_list[4].strategy and neighbor_list[2].strategy != neighbor_list[4].strategy:
                    vortex_count += 1
                    
        elif not model.periodic_BC:
            for agent in model.schedule.agents:
                if agent.pos[0] != 0 and agent.pos[0] != model.dimension-1 and agent.pos[1] != 0 and agent.pos[1] != model.dimension-1:
                    neighbor_list = [neighbor for neighbor in agent.neighbors]
                    if agent.strategy == neighbor_list[1].strategy and agent.strategy != neighbor_list[0].strategy and agent.strategy != neighbor_list[3].strategy and neighbor_list[0].strategy != neighbor_list[3].strategy:
                        vortex_count += 1
                    elif agent.strategy == neighbor_list[3].strategy and agent.strategy != neighbor_list[0].strategy and agent.strategy != neighbor_list[1].strategy and neighbor_list[0].strategy != neighbor_list[1].strategy:
                        vortex_count += 1
                    elif agent.strategy == neighbor_list[3].strategy and agent.strategy != neighbor_list[6].strategy and agent.strategy != neighbor_list[5].strategy and neighbor_list[5].strategy != neighbor_list[6].strategy:
                        vortex_count += 1
                    elif agent.strategy == neighbor_list[6].strategy and agent.strategy != neighbor_list[3].strategy and agent.strategy != neighbor_list[5].strategy and neighbor_list[3].strategy != neighbor_list[5].strategy:
                        vortex_count += 1
                    elif agent.strategy == neighbor_list[6].strategy and agent.strategy != neighbor_list[4].strategy and agent.strategy != neighbor_list[7].strategy and neighbor_list[4].strategy != neighbor_list[7].strategy:
                        vortex_count += 1
                    elif agent.strategy == neighbor_list[4].strategy and agent.strategy != neighbor_list[1].strategy and agent.strategy != neighbor_list[2].strategy and neighbor_list[1].strategy != neighbor_list[2].strategy:
                        vortex_count += 1
                    elif agent.strategy == neighbor_list[4].strategy and agent.strategy != neighbor_list[6].strategy and agent.strategy != neighbor_list[7].strategy and neighbor_list[6].strategy != neighbor_list[7].strategy:
                        vortex_count += 1
                    elif agent.strategy == neighbor_list[1].strategy and agent.strategy != neighbor_list[2].strategy and agent.strategy != neighbor_list[4].strategy and neighbor_list[2].strategy != neighbor_list[4].strategy:
                        vortex_count += 1


        agent_pos = [agent.pos for agent in model.schedule.agents]
        return vortex_count/2



class RPSModel(GameGrid):
    def __init__(self, config):
        super().__init__(config)
        self.epsilon = config['epsilon']
        self.payoff = {("R", "R"): 0,
                       ("R", "P"): -self.epsilon,
                       ("R", "S"): 1,
                       ("R", "E"): 0,
                       ("P", "R"): 1,
                       ("P", "P"): 0,
                       ("P", "S"): -self.epsilon,
                       ("P", "E"): 0,
                       ("S", "R"): -self.epsilon,
                       ("S", "P"): 1,
                       ("S", "S"): 0,
                       ("S", "E"): 0,
                       ("E", "R"): 0,
                       ("E", "P"): 0,
                       ("E", "S"): 0,
                       ("E", "E"): 0}

        for x in range(self.width):
            for y in range(self.height):
                agent = RPSAgent([x, y], self)
                self.grid.place_agent(agent, (x, y))
                self.schedule.add(agent)

        if self.game_mode == "Pure":
            self.datacollector_populations = DataCollector(
                {"Pure Rock": lambda m: self.count_populations(m, "all_r"),
                 "Pure Paper": lambda m: self.count_populations(m, "all_p"),
                 "Pure Scissors": lambda m: self.count_populations(m, "all_s"),
                 "Empty": lambda m: self.count_populations(m, "empty")})
            self.datacollector_populations.collect(self)

        elif self.game_mode == "Impure":
            self.datacollector_probabilities = DataCollector(
                {"Rock Probabilities": lambda m: (a.probabilities[0] for a in m.schedule.agents),
                 "Paper Probabilities": lambda m: (a.probabilities[1] for a in m.schedule.agents),
                 "Scissors Probabilities": lambda m: (a.probabilities[2] for a in m.schedule.agents)})
            self.datacollector_probabilities.collect(self)

        self.datacollector_scores = DataCollector(
            {"Pure Rock Scores": lambda m: self.count_scores(m, "all_r"),
             "Pure Paper Scores": lambda m: self.count_scores(m, "all_p"),
             "Pure Scissors Scores": lambda m: self.count_scores(m, "all_s")}
        )

        self.datacollector_mutating_agents = DataCollector(
            {"Num Mutating Agents": "fraction_mutating"}
        )

        self.datacollector_no_vortices = DataCollector(
            {"Number of Vortices": lambda m: self.count_vortices(m)}
        )
        
    def step(self):
        self.step_num += 1
        self.num_mutating = 0
        
#        if self.step_num % 2 == 1:
#            for agent in self.schedule.agents:
#                agent.increment_score()
#            for agent in self.schedule.agents:
#                agent.kill_weak()
##            for agent in self.schedule.agents:
##                agent.identify_crowded()
##            for agent in self.schedule.agents:
##                agent.kill_crowded()
##
#        elif self.step_num % 2 == 0:
#            for agent in self.schedule.agents:
#                agent.reproduce()
#        for agent in self.schedule.agents:
#            agent.implement_strategy()

#        for agent in self.schedule.agents:
#            agent.increment_score()
#        for agent in self.schedule.agents:
#            agent.evolve_strategy()
#        for agent in self.schedule.agents:
#            agent.implement_strategy()
#        for agent in self.schedule.agents:
#            agent.exchange()

        for agent in self.schedule.agents:
             agent.increment_score()
        for agent in self.schedule.agents:
             agent.identify_crowded()
        for agent in self.schedule.agents:
             agent.kill_weak()
        for agent in self.schedule.agents:
             agent.implement_strategy()
        for agent in self.schedule.agents:
             agent.reproduce()
        for agent in self.schedule.agents:
             agent.implement_strategy()
        for agent in self.schedule.agents:
             agent.exchange()
        for agent in self.schedule.agents:
             agent.implement_strategy()
        agent_list = [agent for agent in self.schedule.agents]
        agent_list2 = [agent.unique_id for agent in self.schedule.agents]
#        print(agent_list[0].pos, agent_list[1].pos, agent_list[2].pos, agent_list[3].pos, agent_list[24].pos, agent_list[25].pos)
#        print(agent_list2[0], agent_list2[1], agent_list2[2], agent_list2[3], agent_list2[24], agent_list2[25])
##        neighbors = [neighbor for neighbor in agent_list[0].grid.get_neighbors(self.pos, moore=self.model.diagonal_neighbours)]
#        neighbor_list = [neighbor.pos for neighbor in agent_list[262].neighbors]
#        print("Own position ->", agent_list[262].pos, "\n Neighbour positions ->", neighbor_list)
#        print([agent_list[262].neighbors])
#        print(agent[262].pos, "Neighbour positions ->", [neighbor.pos for neighbor in agent_list[262].neighbors])
        
        if self.game_mode == "Pure":
            self.datacollector_populations.collect(self)
        elif self.game_mode == "Impure":
            self.datacollector_probabilities.collect(self)
#        self.fraction_mutating = self.num_mutating / (self.dimension**2)
        self.datacollector_scores.collect(self)
        self.datacollector_no_vortices.collect(self)
#        self.datacollector_mutating_agents.collect(self)
        # logger.error(" " + "\n", color=41)


class PDModel(GameGrid):
    def __init__(self, config):
        super().__init__(config)
        self.payoff = {("C", "C"): 1,
                       ("C", "D"): 0,
                       ("D", "C"): 2,
                       ("D", "D"): 0}

        # Create agents
        for x in range(self.dimension):
            for y in range(self.dimension):
                agent = PDAgent([x,y], self)
                self.grid.place_agent(agent, (x, y))
                self.schedule.add(agent)

        self.datacollector_populations = DataCollector(
            {"cooperating": lambda m: self.count_populations(m, "all_c"),
             "defecting": lambda m: self.count_populations(m, "all_d"),
             "tit_for_tat": lambda m: self.count_populations(m, "tit_for_tat"),
             "spiteful": lambda m: self.count_populations(m, "spiteful"),
             "random": lambda m: self.count_populations(m, "random")})
        self.datacollector_populations.collect(self)

        self.datacollector_scores = DataCollector({"cooperating Scores": lambda m: self.count_scores(m, "all_c"),
                                                   "defecting Scores": lambda m: self.count_scores(m, "all_d"),
                                                   "tit_for_tat Scores": lambda m: self.count_scores(m, "tit_for_tat"),
                                                   "spiteful Scores": lambda m: self.count_scores(m, "spiteful"),
                                                   "random Scores": lambda  m: self.count_scores(m, "random")})

    def step(self):
        self.schedule.step()
        self.step_num += 1
        move_count = 0
        for i in range(self.num_moves_per_set):
            for agent in self.schedule.agents:
                agent.increment_score(move_count)
        for agent in self.schedule.agents:
            agent.evolve_strategy()
        for agent in self.schedule.agents:
            agent.implement_strategy()
        self.datacollector_populations.collect(self)