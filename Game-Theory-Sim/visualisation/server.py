from mesa.visualization.modules import CanvasGrid
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import ChartModule
from sim.model import Model


def agent_portrayal(agent):
    portrayal = {"Shape": "circle",
                 "Filled": "true",
                 "Layer": 0,
                 "r": 0.5}

    if agent.value == 1:
        portrayal["Color"] = "red"
    if agent.value == 2:
        portrayal["Color"] = "grey"
    if agent.value == 3:
        portrayal["Color"] = "cyan"
    return portrayal


grid = CanvasGrid(agent_portrayal, 2, 2, 1000, 1000)

# it is essential the label matches that collected by the datacollector
chart = ChartModule([{"Label": "Paper",
                      "Color": "Red"},
                     {"Label": "Rock",
                      "Color": "Grey"},
                     {"Label": "Scissors",
                      "Color": "Cyan"}],
                    data_collector_name='datacollector')
server = ModularServer(Model,
                       [grid, chart],
                       "Game Theory Simulator",
                       {"N": 100, "width": 2, "height": 2})