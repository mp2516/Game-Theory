from mesa.visualization.modules import CanvasGrid
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import ChartModule
from mesa.visualization.UserParam import UserSettableParameter
from .config import Config
from .model import RPS_Model, PD_Model
from colour import Color

file_name = "game_theory/game_configs/rock_paper_scissors.json"

with open(file_name) as d:
    model_config = Config(d.read())


model_height = model_config.system['height']
model_width = model_config.system['width']
game_type = model_config.game['game_type']


def agent_portrayal(agent):
    # opacity should be a number between 0-1
    opacity = agent.score / max([agent.score for agent in agent.schedule.agents])

    portrayal = {"Shape": "circle",
                 "Filled": "true",
                 "Layer": 0,
                 "r": 0.5,
                 "opacity": opacity}

    if game_type == "RPS":
        strategy_to_colour = {"Pure Rock": Color("red"), "Pure Paper": Color("green"), "Pure Scissors": Color("blue")}

    elif game_type == "PD":
        strategy_to_colour = {"Cooperating": Color("red"), "Defecting": Color("green")}

    for strategy, colour in strategy_to_colour:
        if agent.strategy == strategy:
            portrayal["Color"] = colour

    return portrayal


grid = CanvasGrid(agent_portrayal, model_width, model_height, 500, 500)

# it is essential the label matches that collected by the datacollector
if game_type == "RPS":
    chart = ChartModule([{"Label": "Pure Rock",
                          "Color": "Red"},
                         {"Label": "Pure Paper",
                          "Color": "Grey"},
                         {"Label": "Pure Scissors",
                          "Color": "Cyan"},
                         {"Label": "Perfect Mixed",
                          "Color": "Blue"},
                         {"Label": "Imperfect Mixed",
                          "Color": "Green"}],
                        data_collector_name='datacollector')
elif game_type == "PD":
    chart = ChartModule([{"Label": "Cooperating", "Color": "Red"}, {"Label": "Defecting", "Color": "Grey"}],
                        data_collector_name='datacollector')

server = ModularServer(RPS_Model,
                       [grid, chart],
                       "Game Theory Simulator",
                       {"config": model_config})
