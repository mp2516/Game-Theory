from mesa.visualization.modules import CanvasGrid, ChartModule, TextElement
from mesa.visualization.ModularVisualization import ModularServer, VisualizationElement
from .config import Config
from .model import PDModel, RPSModel
from colour import Color
from .logger import logger
import json
import numpy as np

# file_name = "game_theory/game_configs/prisoners_dilemma.json"
file_name = "game_theory/game_configs/rock_paper_scissors.json"
with open(file_name) as d:
    model_config = Config(d.read())


class HistogramModule(VisualizationElement):
    package_includes = ["jsChart.min.js"]
    local_includes = ["histogram_module.js"]

    def __init__(self, bins, canvas_height, canvas_width):
        super().__init__()
        self.canvas_height = canvas_height
        self.canvas_width = canvas_width
        self.bins = bins
        new_element = "new HistogramModule({}, {}, {})"
        new_element = new_element.format(bins,
                                         canvas_width,
                                         canvas_height)
        self.js_code = "elements.push(" + new_element + ");"

    def render(self, model):
        score_vals = [agent.total_score for agent in model.schedule.agents]
        hist = np.histogram(score_vals, bins=self.bins)[0]
        print(hist)
        print([int(x) for x in hist])
        return [int(x) for x in hist]


class MutatingAgents(TextElement):

    def __init__(self):
        pass

    def render(self, model):
        return "Mutating Agents: " + str(model.num_mutating_agents)



def agent_portrayal(agent):
    # opacity should be a number between 0-1

    portrayal = {"Shape": "rect",
                 "w": 1,
                 "h": 1,
                 "Filled": "true",
                 "Layer": 0}

    if model_config.game_type == "RPS":
        agent_prob = agent.probabilities
#        portrayal["Color"] = Color(rgb=[rgb / max(agent_prob) for rgb in agent_prob]).hex
        
        if agent.strategy == "all_r":
            portrayal["Color"] = "red"
        if agent.strategy == "all_p":
            portrayal["Color"] = "green"
        if agent.strategy == "all_s":
            portrayal["Color"] = "blue"        
        if agent.strategy == "empty":
            portrayal["Color"] = "grey"

    elif model_config.game_type == "PD":
        strategy_to_color = {"all_c": "Red", "all_d": "Blue", "tit_for_tat": "Green", "spiteful": "Yellow", "random": "Pink"}
        portrayal["Color"] = strategy_to_color[agent.strategy]
    return portrayal


grid = CanvasGrid(agent_portrayal, model_config.dimension, model_config.dimension, 500, 500)

# it is essential the label matches that collected by the datacollector
if model_config.game_type == "RPS":
    if model_config.game_mode == "Pure":
        chart_populations = ChartModule([{"Label": "Pure Rock", "Color": "red"}, {"Label": "Pure Paper", "Color": "green"},
                             {"Label": "Pure Scissors", "Color": "blue"}],
                            data_collector_name='datacollector_populations')
        chart_scores = ChartModule([{"Label": "Pure Rock Scores", "Color": "red"}, {"Label": "Pure Paper Scores", "Color": "green"}, {"Label": "Pure Scissors Scores", "Color": "blue"}],
                            data_collector_name='datacollector_scores')
        mutating_agents = ChartModule([{"Label": "Num Mutating Agents", "Color": "red"}], data_collector_name='datacollector_mutating_agents')
        histogram = HistogramModule(list(range(-30, 30)), 500, 500)
        server = ModularServer(RPSModel, [grid, chart_populations, chart_scores, mutating_agents], "Rock Paper Scissors Simulator", {"config": model_config})
    else:
        server = ModularServer(RPSModel, [grid], "Rock Paper Scissors Simulator", {"config": model_config})

elif model_config.game_type == "PD":
    chart_populations = ChartModule([{"Label": "cooperating", "Color": "Red"}, {"Label": "defecting", "Color": "Blue"}, {"Label": "tit_for_tat", "Color": "Green"}, {"Label": "spiteful", "Color": "Yellow"}, {"Label": "random", "Color": "Pink"}],
                        data_collector_name='datacollector_populations')
    chart_scores = ChartModule([{"Label": "cooperating Scores", "Color": "Red"}, {"Label": "defecting Scores", "Color": "Blue"}, {"Label": "tit_for_tat Scores", "Color": "Green"}, {"Label": "spiteful Scores", "Color": "Yellow"}, {"Label": "random Scores", "Color": "Pink"}],
                        data_collector_name='datacollector_scores')
    server = ModularServer(PDModel, [grid, chart_populations, chart_scores], "Prisoners Dilemma Simulator", {"config": model_config})

server.verbose = False

logger.critical("Started server.")