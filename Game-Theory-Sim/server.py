from mesa.visualization.modules import CanvasGrid
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import ChartModule
from mesa.visualization.UserParam import UserSettableParameter
from .model import Grid


n_slider = UserSettableParameter('slider', "Number of Agents", 100, 2, 200, 1)


def agent_portrayal(agent):
    portrayal = {"Shape": "circle",
                 "Filled": "true",
                 "Layer": 0,
                 "r": 0.5}

    if Grid.game == "RPS":
        if agent.strategy == "Pure Rock":
            portrayal["Color"] = "red"
        elif agent.strategy == "Pure Paper":
            portrayal["Color"] = "grey"
        elif agent.strategy == "Pure Scissors":
            portrayal["Color"] = "cyan"
        elif agent.strategy == "Perfect Mixed":
            portrayal["Color"] = "blue"
        elif agent.strategy == "Imperfect Mixed":
            portrayal["Color"] = "green"
    return portrayal


grid = CanvasGrid(agent_portrayal, 10, 10, 500, 500)

# it is essential the label matches that collected by the datacollector
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

server = ModularServer(Model,
                       [grid, chart],
                       "Game Theory Simulator",
                       {"N": n_slider, "width": 10, "height": 10})
