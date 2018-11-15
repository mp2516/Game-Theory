from mesa.visualization.modules import CanvasGrid, ChartModule, ChartVisualization
from mesa.visualization.ModularVisualization import ModularServer, VisualizationElement
from mesa.visualization.UserParam import UserSettableParameter
from .config import Config
from .model import GameGrid
from colour import Color
from .logger import logger

file_name = "game_theory/game_configs/rock_paper_scissors.json"

with open(file_name) as d:
    model_config = Config(d.read())


# model_height = 10
# model_width = 10
# game_type = "RPS"
# game_mode = "Imperfect"
# num_moves_per_set = 1
# cull_threshold = 0.45
# probability_adoption = 0.9
# strength_of_adoption = 0.1
# probability_mutation = 0.05


class SimpleCanvas(VisualizationElement):
    local_includes = ['simple_continuous_canvas.js']
    portrayal_method = None

    def __init__(self, portrayal_method):
        super().__init__()
        self.portrayal_method = portrayal_method

        new_element = ("new Simple_Continuous_Module({})".format(canvas_config))
        self.js_code = "elements.push(" + new_element + ");"

    def render(self, model):
        space_state = []
        self.render_agents(model, space_state)

        return space_state

    def render_agents(self, model, space_state):
        for agent in model.agent_schedule.agents:
            self.render_agent(agent, space_state)

    def render_agent(self, agent, space_state):
        portrayal = self.portrayal_method(agent)
        space_state.append(portrayal)


def agent_portrayal(agent):
    # opacity should be a number between 0-1

    portrayal = {"Shape": "rect",
                 "w": 1,
                 "h": 1,
                 "Filled": "true",
                 "Layer": 0}

    if model_config.game_type == "RPS":
        agent_prob = agent.probabilities

    elif model_config.game_type == "PD":
        # in order to make the list 3 x 1 which ensures it fits into the RBG format
        agent_prob = agent.probabilities.append(0)
        print(agent.probabilities)

    rbg_colours = [rbg / max(agent_prob) for rbg in agent_prob]
    agent_colour = Color(rgb=rbg_colours)
    portrayal["Color"] = agent_colour.hex

    return portrayal


grid = CanvasGrid(agent_portrayal, model_config.width, model_config.height, 500, 500)

# it is essential the label matches that collected by the datacollector
if model_config.game_type == "RPS":
    if model_config.game_mode == "Pure Only" or model_config.game_mode == "Pure and Perfect":
        chart = ChartModule([{"Label": "Pure Rock",
                              "Color": "red"},
                             {"Label": "Pure Paper",
                              "Color": "green"},
                             {"Label": "Pure Scissors",
                              "Color": "blue"},
                             {"Label": "Perfect Mixed",
                              "Color": "black"}],
                            data_collector_name='datacollector_populations')
        server = ModularServer(GameGrid, [grid, chart], "Rock Paper Scissors Simulator", {"config": model_config})
    else:
        server = ModularServer(GameGrid, [grid], "Rock Paper Scissors Simulator", {"config": model_config})

elif model_config.game_type == "PD":
    chart = ChartModule([{"Label": "Cooperating", "Color": "Red"}, {"Label": "Defecting", "Color": "Blue"}],
                        data_collector_name='datacollector')
    server = ModularServer(GameGrid, [grid, chart], "Prisoners Dilemma Simulator", {"config": model_config})

server.verbose = False

logger.critical("Started server.")