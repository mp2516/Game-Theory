from mesa.datacollection import DataCollector
from mesa.space import SingleGrid
from mesa import Model
from mesa.time import BaseScheduler, RandomActivation
from sim.agent import Agent
import random


# def compute_gini(model):
#     agent_wealths = [agent.wealth for agent in model.schedule.agents]
#     x = sorted(agent_wealths)
#     N = model.num_agents
#     B = sum( xi * (N-i) for i,xi in enumerate(x) ) / (N*sum(x))
#     return (1 + (1/N) - 2*B)


def calculate_population_1(model):
    agent_values = [agent.value for agent in model.schedule.agents]
    return agent_values.count(1)


def calculate_population_2(model):
    agent_values = [agent.value for agent in model.schedule.agents]
    return agent_values.count(2)


def calculate_population_3(model):
    agent_values = [agent.value for agent in model.schedule.agents]
    return agent_values.count(3)



class Model(Model):
    """A model with some number of agents."""
    def __init__(self, N, width, height):
        self.num_agents = N
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
            model_reporters={"Paper": calculate_population_1, "Rock": calculate_population_2, "Scissors": calculate_population_3},  # A function to call
            agent_reporters={"Value": "value"})  # An agent attribute

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()