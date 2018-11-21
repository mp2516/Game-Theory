from game_theory.server import server, file_name

import cProfile
import sys

from tqdm import trange
from game_theory.model import GameGrid
from game_theory.config import Config

from game_theory.analysis import fft_analysis, ternary_plot

def run_model(config, n):
    model = GameGrid(config)
    for _ in trange(n):
        model.step()
    print("-" * 10 + "\nSimulation finished!\n" + "-" * 10)

    fft_analysis(model)
    if config.game_mode == "Pure Only":
        labels = ["Pure Rock", "Pure Paper", "Pure Scissors"]
        ternary_plot(model, labels)
    elif config.game_type == "Imperfect":
        labels = ["P(Rock)", "P(Paper)", "P(Scissors)"]
        ternary_plot(model, labels)

    for agent in model.schedule.agents:
        print("ID: {id}\n"
              "Average Score: {average_score}\n"
              "---------------------------".format(
            id=agent.unique_id,
            average_score=agent.total_scores))

with open(file_name) as d:
    model_config = Config(d.read())

if model_config.simulation:
    server.port = 8521  # The default
    server.launch()
else:
    run_model(model_config, model_config.number_of_steps)