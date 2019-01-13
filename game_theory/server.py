from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.ModularVisualization import ModularServer
from .config import Config
from .model import RPSModel
import copy
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

    if agent.strategy == "all_r":
        portrayal["Color"] = "red"
    if agent.strategy == "all_p":
        portrayal["Color"] = "green"
    if agent.strategy == "all_s":
        portrayal["Color"] = "blue"
    if agent.strategy == "empty":
        portrayal["Color"] = "black"

    return portrayal


def chart_populations(labels):
    print(labels)
    return ChartModule(labels, data_collector_name='datacollector_population')


def chart_scores(labels):
    score_labels = copy.deepcopy(labels)
    for row in score_labels:
        row["Label"] += " Scores"
    return ChartModule(score_labels, data_collector_name='datacollector_score')


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
    model_pixel_width = model_config['pixel_dimension']
    model_pixel_height = model_pixel_width / model_width

grid = CanvasGrid(agent_portrayal,
                  model_width,
                  model_height,
                  model_pixel_width,
                  model_pixel_height)

# it is essential the label matches that collected by the datacollector

model_name = "Rock Paper Scissors Simulator"
model_type = RPSModel

model_labels = [{"Label": "Rock", "Color": "red"},
                {"Label": "Paper", "Color": "green"},
                {"Label": "Scissors", "Color": "blue"}]

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
