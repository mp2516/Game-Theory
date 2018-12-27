from mesa.visualization.modules import CanvasGrid
from game_theory.visualization.ChartVisualization import ChartModule
from mesa.visualization.ModularVisualization import ModularServer
from .config import Config
from .model import RPSModel
from colour import Color
from .logger import logger
from game_theory.visualization.HistogramVisualization import HistogramModule

# file_name = "game_theory/game_configs/prisoners_dilemma.json"
file_name = "game_theory/game_configs/rock_paper_scissors.json"
with open(file_name) as d:
    model_config = Config(d.read()).parameters


def agent_portrayal(agent):
    portrayal = {"Shape": "rect",
                 "w": 1,
                 "h": 1,
                 "Filled": "true",
                 "Layer": 0}

    agent_prob = agent.probabilities
    portrayal["Color"] = Color(rgb=[rgb / max(agent_prob) for rgb in agent_prob]).hex
    if agent.strategy == "empty":
        portrayal["Color"] = "black"

    return portrayal


def chart_populations(labels):
    return ChartModule(labels, data_collector_name='datacollector_populations')


def chart_scores(labels):
    for row in labels:
        row["Label"] += " Scores"
    return ChartModule(labels, data_collector_name='datacollector_scores')


def evolving_agents():
    return ChartModule([{"Label": "Num Evolving Agents", "Color": "black"}],
                       data_collector_name='datacollector_evolving_agents')


def histogram_module():
    return HistogramModule(list(range(-30, 30)), model_config['pixel_dimension'], 200)


def mutating_agents():
    return ChartModule([{"Label": "Num Mutating Agents", "Color": "black"}],
                       data_collector_name='datacollector_mutating_agents')


if model_config['square']:
    model_height = model_config['dimension']
    model_width = model_config['dimension']
    model_pixel_width = model_config['pixel_dimension']
    model_pixel_height = model_config['pixel_dimension']
else:
    model_height = model_config['height']
    model_width = model_config['width']
    model_pixel_width = model_config['pixel_width']
    model_pixel_height = model_config['pixel_height']

grid = CanvasGrid(agent_portrayal,
                  model_width,
                  model_height,
                  model_pixel_width,
                  model_pixel_height)

# it is essential the label matches that collected by the datacollector

model_name = "Rock Paper Scissors Simulator"
model_type = RPSModel

model_labels = [{"Label": "Pure Rock", "Color": "red"},
                {"Label": "Pure Paper", "Color": "green"},
                {"Label": "Pure Scissors", "Color": "blue"},
                {"Label": "Empty", "Color": "black"}]
model_visualisation = [grid,
                       chart_populations(model_labels),
                       chart_scores(model_labels),
                       evolving_agents()]

if model_config['probability_mutation'] > 0:
    model_visualisation.append(mutating_agents())


server = ModularServer(model_cls=model_type,
                       visualization_elements=model_visualisation,
                       name=model_name,
                       model_params={"config": model_config})

server.verbose = False
logger.critical("Started server.")
