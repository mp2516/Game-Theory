from mesa.visualization.modules import CanvasGrid
from game_theory.visualization.ChartVisualization import ChartModule
from mesa.visualization.ModularVisualization import ModularServer
from .config import Config
from .model import PDModel, RPSModel
from colour import Color
from .logger import logger
from game_theory.visualization.HistogramVisualization import HistogramModule

# file_name = "game_theory/game_configs/prisoners_dilemma.json"
file_name = "game_theory/game_configs/rock_paper_scissors.json"
with open(file_name) as d:
    model_config = Config(d.read())


def agent_portrayal(agent):
    portrayal = {"Shape": "rect",
                 "w": 1,
                 "h": 1,
                 "Filled": "true",
                 "Layer": 0}

    if model_config.parameters['game_type'] == "RPS":
        portrayal["Color"] = Color(rgb=[rgb / max(agent.probabilities) for rgb in agent.probabilities]).hex
        portrayal["Opacity"] = agent.total_score / 5
        if agent.strategy == "empty":
            portrayal["Color"] = "black"

    elif model_config.parameters['game_type'] == "PD":
        strategy_to_color = {"all_c": "Red",
                             "all_d": "Blue",
                             "tit_for_tat": "Green",
                             "spiteful": "Yellow",
                             "random": "Pink"}
        portrayal["Color"] = strategy_to_color[agent.strategy]
    return portrayal


if model_config.parameters['square']:
    model_height = model_config.parameters['dimension']
    model_width = model_config.parameters['dimension']
    model_pixel_width = model_config.parameters['pixel_dimension']
    model_pixel_height = model_config.parameters['pixel_dimension']
else:
    model_height = model_config.parameters['height']
    model_width = model_config.parameters['width']
    model_pixel_width = model_config.parameters['pixel_width']
    model_pixel_height = model_config.parameters['pixel_height']

grid = CanvasGrid(agent_portrayal,
                  model_width, model_height, model_pixel_width, model_pixel_height)

# it is essential the label matches that collected by the datacollector
if model_config.parameters['game_type'] == "RPS":
    if model_config.parameters['game_mode'] == "Pure":
        chart_populations = ChartModule([{"Label": "Pure Rock", "Color": "red"},
                                         {"Label": "Pure Paper", "Color": "green"},
                                         {"Label": "Pure Scissors", "Color": "blue"}],
                                        data_collector_name='datacollector_populations')
        chart_scores = ChartModule([{"Label": "Pure Rock Scores", "Color": "red"},
                                    {"Label": "Pure Paper Scores", "Color": "green"},
                                    {"Label": "Pure Scissors Scores", "Color": "blue"}],
                                   data_collector_name='datacollector_scores')
        mutating_agents = ChartModule([{"Label": "Num Mutating Agents", "Color": "black"}],
                                      data_collector_name='datacollector_mutating_agents')

        # histogram = HistogramModule(list(range(-30, 30)),
        #                       model_config.pixel_dimension, 200)

        server = ModularServer(RPSModel,
                               [grid, chart_populations, chart_scores, mutating_agents],
                               "Rock Paper Scissors Simulator",
                               {"config": model_config.parameters})
    else:
        # elif the game_mode is Impure
        server = ModularServer(RPSModel,
                               [grid],
                               "Rock Paper Scissors Simulator",
                               {"config": model_config.parameters})

else:
    # elif the game type is Prisoner's Dilemma
    chart_populations = ChartModule([{"Label": "cooperating", "Color": "Red"},
                                     {"Label": "defecting", "Color": "Blue"},
                                     {"Label": "tit_for_tat", "Color": "Green"},
                                     {"Label": "spiteful", "Color": "Yellow"},
                                     {"Label": "random", "Color": "Pink"}],
                                    data_collector_name='datacollector_populations')
    chart_scores = ChartModule([{"Label": "cooperating Scores", "Color": "Red"},
                                {"Label": "defecting Scores", "Color": "Blue"},
                                {"Label": "tit_for_tat Scores", "Color": "Green"},
                                {"Label": "spiteful Scores", "Color": "Yellow"},
                                {"Label": "random Scores", "Color": "Pink"}],
                               data_collector_name='datacollector_scores')
    server = ModularServer(PDModel,
                           [grid, chart_populations, chart_scores],
                           "Prisoners Dilemma Simulator",
                           {"config": model_config.parameters})

server.verbose = False

logger.critical("Started server.")