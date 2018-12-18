from tqdm import trange
from game_theory.model import RPSModel, PDModel
from game_theory.config import Config
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from game_theory.analysis import RPSAnalysis

def run_model(config, batchrunning):
    if batchrunning['variable_output']:
        dependent = []
        dependent_error = []
        histogram_data = []
        for variable in range(batchrunning['start'], batchrunning['stop'], batchrunning['step']):
            config[batchrunning['variable_name']] = variable
            dependent_raw_data = []
            for _ in trange(batchrunning['num_sims_per_interval']):
                if config['game_type'] == "RPS":
                    model = RPSModel(config)
                else:
                    model = PDModel(config)
                model.run(batchrunning['num_steps'])
                dependent_raw_data.append(collect_dependent_raw_data(model, batchrunning['dependent_name']))
            dependent.append(calculate_dependent(dependent_raw_data, batchrunning['dependent_name']))
            dependent_error.append(calculate_dependent_error(dependent_raw_data,
                                                             batchrunning['dependent_name'],
                                                             batchrunning['num_sim_batches']))
            if batchrunning['histogram']:
                histogram_data.append(dependent_raw_data)

        if batchrunning['histogram']:
            RPSAnalysis.histogram_results(histogram_data, batchrunning['num_steps'])

        print('\n' + '-' * 30 + '\n Simulation Finished \n' + '-' * 30 + '\n')
        print(tabulate(zip(dependent, dependent_error), headers=[str(batchrunning['dependent_name']), 'error']))
        variable = np.arange(batchrunning['start'], batchrunning['stop'], batchrunning['step'])
        plt.figure()
        plt.scatter(variable, dependent, c='b')
        plt.errorbar(variable, dependent, yerr=dependent_error, elinewidth=0.2, ecolor='b')
        plt.xlabel(batchrunning['variable_name'])
        plt.ylabel(batchrunning['dependent_name'])
        plt.xlim([batchrunning['start'], batchrunning['stop']])
        plt.ylim([0, 1.05])
        plot_name = str(batchrunning['variable_name'])\
                    + '_' + str(batchrunning['dependent_name'])\
                    + '_' + str(batchrunning['num_steps']) + 'n'\
                    + '_' + str(batchrunning['num_sims_per_interval']) + 'sims'
        print(plot_name)
        plt.savefig('graphs/batchruns/' + plot_name + '.png')
        plt.show()

    else:
        if config['game_type'] == "RPS":
            model = RPSModel(config)
        else:
            model = PDModel(config)
        model.run(batchrunning['num_steps'])


file_name = "game_theory/game_configs/rock_paper_scissors.json"
with open(file_name) as d:
    model_config = Config(d.read())

if model_config.parameters['simulation']:
    from game_theory.server import server
    server.port = 8521  # The default
    server.launch()
else:
    run_model(model_config.parameters, model_config.batchrunning)