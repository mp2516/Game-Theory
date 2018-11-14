from mesa.visualization.modules import CanvasGrid, ChartModule, ChartVisualization
from mesa.visualization.ModularVisualization import ModularServer, VisualizationElement
from mesa.visualization.UserParam import UserSettableParameter
from .config import Config
from .model import RPS_Model, PD_Model, GameGrid
from colour import Color
from .logger import logger

file_name = "game_theory/game_configs/rock_paper_scissors.json"

with open(file_name) as d:
    model_config = Config(d.read())


model_height = 10
model_width = 10
game_type = "RPS"
game_mode = "Imperfect"
num_moves_per_set = 1
cull_threshold = 0.45
probability_adoption = 0.9
strength_of_adoption = 0.1
probability_mutation = 0.05


class SimpleCanvas(VisualizationElement):
    local_includes = ['simple_continuous_canvas.js']
    portrayal_method = None

    def __init__(self, portrayal_method, num_agents_edge, canvas_size):
        super().__init__()
        self.portrayal_method = portrayal_method
        self.num_agents_edge = num_agents_edge
        self.canvas_size = canvas_size

        canvas_config = dict(CANVAS_SIZE=self.canvas_size,
                             NUM_AGENTS_EDGE=self.num_agents_edge)

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
        # opacity should be a number between 0-1

        portrayal = {"Shape": "rect", "w": 1, "h": 1, "Filled": "true", "Layer": 0}

        if game_type == "RPS":
            agent_prob = agent.probabilities

        elif game_type == "PD":
            # in order to make the list 3 x 1 which ensures it fits into the RBG format
            agent_prob = agent.probabilities.append(0)

        rbg_colours = [rbg / max(agent_prob) for rbg in agent_prob]
        agent_colour = Color(rgb=rbg_colours)
        portrayal["Color"] = agent_colour.hex


grid = CanvasGrid(agent_portrayal, model_width, model_height, 500, 500)

# it is essential the label matches that collected by the datacollector
if game_type == "RPS":
    chart = ChartModule([{"Label": "Pure Rock",
                          "Color": "Red"},
                         {"Label": "Pure Paper",
                          "Color": "green"},
                         {"Label": "Pure Scissors",
                          "Color": "blue"},
                         {"Label": "Perfect Mixed",
                          "Color": "purple"},
                         {"Label": "Imperfect Mixed",
                          "Color": "dark purple"}],
                        data_collector_name='datacollector')
elif game_type == "PD":
    chart = ChartModule([{"Label": "Cooperating", "Color": "Red"}, {"Label": "Defecting", "Color": "Blue"}],
                        data_collector_name='datacollector')

model_params = {
    'width': model_width,
    'height': model_height,
    'num_moves_per_set': num_moves_per_set,
    'game_type': game_type,
    'game_mode': game_mode,
    'cull_threshold': cull_threshold,
    'probability_adoption': probability_adoption,
    'strength_of_adoption': strength_of_adoption,
    'probability_mutation': probability_mutation
}

server = ModularServer(GameGrid,
                       [grid],
                       "Game Theory Simulator",
                        model_params)
server.verbose = False

logger.critical("Started server.")