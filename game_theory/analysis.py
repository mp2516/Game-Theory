import numpy as np
import scipy.fftpack
from matplotlib import pyplot as plt
import ternary
import random
import statistics

from tabulate import tabulate
from tqdm import trange

from game_theory.model import RPSModel, PDModel


class BatchRunner:
    def __init__(self, config_batchrunning, config_model):

        self.num_steps = config_batchrunning['num_steps']
        self.variable_name = config_batchrunning['variable_name']
        self.dependent_name = config_batchrunning['dependent_name']
        self.start = config_batchrunning['start']
        self.stop = config_batchrunning['stop']
        self.step = config_batchrunning['step']
        self.num_sims_per_interval = config_batchrunning['num_sims_per_interval']
        self.num_sim_batches = config_batchrunning['num_sim_batches']

        self.config_model = config_model

        self.dependent = []
        self.dependent_error = []
        self.dependent_raw_data = []

    def run_model(self):
        for variable in range(self.start, self.stop, self.step):
            self.config_model[self.variable_name] = variable
            for _ in trange(self.num_sims_per_interval):
                if self.config_model['game_type'] == "RPS":
                    model = RPSModel(self.config_model)
                else:
                    model = PDModel(self.config_model)
                model.run(self.num_steps)
                self.dependent_raw_data.append(RPSAnalysis.collect_dependent_raw_data(model, self.dependent_name))
            self.dependent.append(RPSAnalysis.calculate_dependent(self.dependent_raw_data, self.dependent_name))
            self.dependent_error.append(RPSAnalysis.calculate_dependent_error(self.dependent_raw_data, self.dependent_name, self.num_sim_batches))


class RPSAnalysis:
    def __init__(self, config_batchrunning, model):
        self.all_population_data = model.datacollector_populations.get_model_vars_dataframe()
        self.all_score_data = model.datacollector_scores.get_model_vars_dataframe()
        self.all_evolving_data = model.datacollector_evolving_agents.get_model_vars_dataframe()

        self.num_steps = config_batchrunning['num_steps']
        self.variable_name = config_batchrunning['variable_name']
        self.dependent_name = config_batchrunning['dependent_name']
        self.transient_threshold = config_batchrunning['transient_threshold']
        self.start = config_batchrunning['start']
        self.stop = config_batchrunning['stop']
        self.step = config_batchrunning['step']
        self.num_sims_per_interval = config_batchrunning['num_sims_per_interval']
        self.num_sim_batches = config_batchrunning['num_sim_batches']
        self.histogram_bool = config_batchrunning['histogram']
        self.line_graph_bool = config_batchrunning['line_graph']

    def run_model(self):
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
                    dependent_raw_data.append(
                        RPSAnalysis.collect_dependent_raw_data(model, batchrunning['dependent_name']))
                dependent.append(RPSAnalysis.alculate_dependent(dependent_raw_data, batchrunning['dependent_name']))
                dependent_error.append(
                    RPSAnalysis.calculate_dependent_error(dependent_raw_data, batchrunning['dependent_name'],
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
            plot_name = str(batchrunning['variable_name']) + '_' + str(batchrunning['dependent_name']) + '_' + str(
                batchrunning['num_steps']) + 'n' + '_' + str(batchrunning['num_sims_per_interval']) + 'sims'
            print(plot_name)
            plt.savefig('graphs/batchruns/' + plot_name + '.png')
            plt.show()

        else:
            if config['game_type'] == "RPS":
                model = RPSModel(config)
            else:
                model = PDModel(config)
            model.run(batchrunning['num_steps'])

    def fft_analysis(self):
        # all_population_data = model.datacollector_populations.get_model_vars_dataframe()
        # surprisingly this iterates over columns, not rows
        for population_data in self.all_population_data:
            N = len(self.all_population_data[population_data])
            print(N)
            print(self.all_population_data[population_data])
            t_axis = np.linspace(0.0, 1.0 / (2.0), (int(N) / 2))
            y_axis = self.all_population_data[population_data] - np.mean(self.all_population_data[population_data])
            y_axis_fft = scipy.fftpack.fft(y_axis)
            y_corrected = 2 / N * np.abs(y_axis_fft[0:np.int(N / 4)])
            t_corrected = t_axis[0:np.int(N / 4)]

            plt.figure(1)
            plt.plot(t_corrected, y_corrected,
                     label='Dominant frequency = '
                           + str(round(t_corrected[np.argmax(y_corrected)], 4))
                           + ' $set^(-1)$'
                           + str(population_data))
            plt.xlabel('Frequency (set^-1)')
            plt.ylabel('FT of Population')
            plt.legend(loc='best')

            plt.figure(2, )
            plt.plot(np.arange(N), self.all_population_data[population_data])
            plt.xlabel('Set no')
            plt.ylabel('Population')

        plt.show()
        print("Dominant frequency >> ", t_corrected[np.argmax(y_corrected)])

def histogram(model):
    all_population_scores = model.datacollector_scores.get_model_vars_dataframe()
    indexes = all_population_scores.columns
    current_population_scores = []
    bins = np.arange((-1) * model.num_moves_per_set * 8, model.num_moves_per_set * 8)
    for a in indexes:
        population_scores = all_population_scores[a]
        current_population_scores.append(population_scores[model.number_of_steps - 1])
    labels = [a for a in all_population_scores.columns]
#    print(labels)
    plt.figure(3,)
    plt.hist(current_population_scores, bins, label=labels, stacked=True)
    plt.xlabel("Score")
    plt.ylabel("Score density")
    plt.legend(loc="best")
#    for population_scores in all_population_scores:
#        print("population scores>>")
#        print(all_population_scores[population_scores])

# =============================================================================
#     for population_scores in all_population_scores:
#         print("population scores>>", all_population_scores[population_scores])
#         plt.hist(all_population_scores[population_scores], bins=4)
# =============================================================================
#bins=model.num_moves_per_set*16+1
#    plt.show()

def ternary_plot(model, labels):
    all_population_data = model.datacollector_populations.get_model_vars_dataframe()
    for index, data in all_population_data.iterrows():



        list_norm = [(i / model.height ** 2) for i in all_population_data[population_data]]
        print(list_norm)
        points = list(zip(list_norm))
        print(points)

        figure, tax = ternary.figure(scale=1.0)
        tax.boundary()
        tax.gridlines(multiple=0.2, color="black")
        tax.set_title("Populations", fontsize=20)
        tax.left_axis_label("Scissors", fontsize=20)
        tax.right_axis_label("Paper", fontsize=20)
        tax.bottom_axis_label("Rock", fontsize=20)

        tax.plot(points, linewidth=2.0, label="Curve")
        tax.ticks(axis='lbr', multiple=0.2, linewidth=1)
        tax.legend()
        tax.show()


    def histogram(self):
        # all_population_scores = model.datacollector_population_scores.get_model_vars_dataframe()
        indexes = self.all_population_scores.columns
        current_population_scores = []
        bins = np.arange((-1) * model.num_moves_per_set * 8, model.num_moves_per_set * 8)
        for a in indexes:
            population_scores = all_population_scores[a]
            current_population_scores.append(population_scores[model.number_of_steps - 1])
        labels = [a for a in all_population_scores.columns]
        print(labels)
        plt.figure(3)
        plt.hist(current_population_scores, bins, label=labels, stacked=True)
        plt.xlabel("Score")
        plt.ylabel("Score density")
        plt.legend(loc="best")
        for population_scores in all_population_scores:
            print("population scores>>")
            print(all_population_scores[population_scores])

    def histogram_results(self, raw_data, num_steps):
        # filtered_raw_data = list(filter(lambda a: a != np.inf, raw_data))
        # print(filtered_raw_data)
        bins = np.arange(num_steps+1)
        # bins = np.linspace(0, num_steps+1, 30)
        plt.hist(raw_data, bins, label=("2", "3", "4"))
        plt.xlabel("Extinction Time")
        plt.ylabel("Number of sims")
        plt.legend(loc="best")


    def pie_chart(self, model):
        current_populations = []
        for population_data in self.all_population_data:
            populations = self.all_population_data[population_data]
            current_populations.append(populations[model.number_of_steps])
        #   current_populations = [a[model.number_of_steps - 1] for a in [population_data for population_data in all_population_data]]
        plt.figure(4, )
        plt.pie(current_populations, labels=[a for a in all_population_data.columns])


    def collect_dependent_raw_data(self, model, dependent_name):
        if dependent_name == "extinction_probability" or dependent_name == "extinction_time":
            return self.calculate_extinction_time(model)
        elif dependent_name == "environment_death":
            step_num = calculate_environment_death(model)
            if step_num == np.inf:
                step_num = calculate_extinction_time(model)
            return step_num


    def calculate_dependent(raw_data, dependent_name):
        if dependent_name == "extinction_probability" or dependent_name == "environment_death":
            return calculate_dependent_prob(raw_data)


    def calculate_dependent_error(raw_data, dependent_name, num_sim_batches):
        if dependent_name == "extinction_probability" or dependent_name == "environment_death":
            return calculate_dependent_prob_sd(raw_data, num_sim_batches)


    def calculate_extinction_time(model):
        """
        :param model:
        :return: np.inf if the population remains stable for the whole simulation, step_num if the population reaches
            extinction after a certain period of time.
        """
        all_population_data = model.datacollector_populations.get_model_vars_dataframe()
        # we only need to loop through one population data set
        for population_data in all_population_data:
            populations = all_population_data[population_data]
            break
        for step_num, i in enumerate(populations):
            if i == (model.dimension ** 2) or i == 0:
                # there is one homogenous population
                # if it is homogenous we can break out of the loop
                return step_num
        return np.inf


    def calculate_stable_transient(model):

        all_population_data = model.datacollector_populations.get_model_vars_dataframe()
        for population_data in all_population_data:
            populations = all_population_data[population_data]
            for step_num, i in enumerate(populations):
                if np.ceil(populations[0] * model.transient_threshold) >= populations[i]\
                        or np.floor(populations[0] * (1 - model.transient_threshold)) <= populations[i]:
                    return step_num


    def calculate_dependent_prob(raw_data):
        dependent_successful = list(filter(lambda a: a != np.inf, raw_data))
        return len(dependent_successful) / len(raw_data)


    def calculate_dependent_prob_sd(raw_data, num_sim_batches):

        def chunk_it(seq, num):
            avg = len(seq) / float(num)
            out = []
            last = 0.0
            while last < len(seq):
                out.append(seq[int(last):int(last + avg)])
                last += avg
            return out

        # raw_data_filtered = list(filter(lambda a: a != np.inf, raw_data))
        split_raw_data = chunk_it(raw_data, num_sim_batches)
        split_extinction_prob = []
        for split in split_raw_data:
            split_extinction_prob.append(calculate_dependent_prob(split))
        return statistics.stdev(split_extinction_prob)


    def calculate_environment_death(model):
        all_mutating_agents = model.datacollector_mutating_agents.get_model_vars_dataframe()
        for mutating_agents in all_mutating_agents:
            mutating = all_mutating_agents[mutating_agents]
            break
        for step_num, i in enumerate(mutating):
            if i == 0 and step_num > 0:
                return step_num
        return np.inf