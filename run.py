from game_theory.server import server, file_name

import cProfile
import sys

from mesa.batchrunner import BatchRunner
from tqdm import trange
from game_theory.model import RPSModel, PDModel
from game_theory.config import Config
from game_theory.analysis import fft_analysis, ternary_plot


def run_model(config, batchrunning):
    if batchrunning['variable_output']:
        fixed_params = config
        variable_params = {batchrunning['variable']:
                           range(batchrunning['start'], batchrunning['stop'], batchrunning['step'])}
        if config['game_mode'] == "RPS":
            model = RPSModel
        else:
            model = PDModel
        batch_run = BatchRunner(model,
                                fixed_parameters=fixed_params,
                                variable_parameters=variable_params,
                                iterations=batchrunning['num_sims_per_interval'],
                                max_steps=batchrunning['num_steps'],
                                display_progress=True)
        batch_run.run_all()
        print("-" * 10 + "\nSimulation finished!\n" + "-" * 10)
        fft_analysis(model)
        # if config.game_mode == "Pure":
        #     labels = ["Pure Rock", "Pure Paper", "Pure Scissors"]
        #     ternary_plot(model, labels)
        # elif config.game_type == "Impure":
        #     labels = ["P(Rock)", "P(Paper)", "P(Scissors)"]
        #     ternary_plot(model, labels)

        for agent in model.schedule.agents:
            print("ID: {id}\n"
                  "Average Score: {average_score}\n"
                  "---------------------------".format(
                id=agent.unique_id,
                average_score=agent.total_scores))

with open(file_name) as d:
    model_config = Config(d.read())

if model_config.parameters['simulation']:
    server.port = 8521  # The default
    server.launch()
else:
    run_model(model_config.parameters, model_config.batchrunning)