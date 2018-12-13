import numpy as np
import scipy.fftpack
from matplotlib import pyplot as plt
import ternary
import random
import statistics

def fft_analysis(model):
    all_population_data = model.datacollector_populations.get_model_vars_dataframe()
    # surprisingly this iterates over columns, not rows
    for population_data in all_population_data:
        N = len(all_population_data[population_data])
        print(N)
        print(all_population_data[population_data])
        t_axis = np.linspace(0.0, 1.0 / (2.0), (int(N) / 2))
        y_axis = all_population_data[population_data] - np.mean(all_population_data[population_data])
        y_axis_fft = scipy.fftpack.fft(y_axis)
        y_corrected = 2 / N * np.abs(y_axis_fft[0:np.int(N / 4)])
        t_corrected = t_axis[0:np.int(N / 4)]

        plt.figure(1)
        plt.plot(t_corrected, y_corrected, label='Dominant frequency = ' + str(round(t_corrected[np.argmax(y_corrected)], 4)) + ' $set^(-1)$' + str(population_data))
        plt.xlabel('Frequency (set^-1)')
        plt.ylabel('FT of Population')
        plt.legend(loc='best')

        plt.figure(2, )
        plt.plot(np.arange(N), all_population_data[population_data])
        plt.xlabel('Set no')
        plt.ylabel('Population')

    plt.show()
    print("Dominant frequency >> ", t_corrected[np.argmax(y_corrected)])


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


def histogram(model):
    all_population_scores = model.datacollector_population_scores.get_model_vars_dataframe()
    indexes = all_population_scores.columns
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


def histogram_results(raw_data, num_steps):

    filtered_raw_data = list(filter(lambda a: a != np.inf, raw_data))
    print(filtered_raw_data)
    bins = np.arange(num_steps)
    plt.hist(filtered_raw_data, bins)
    plt.xlabel("Extinction Time")
    plt.ylabel("Number of sims")
    plt.legend(loc="best")




def pie_chart(model):
    all_population_data = model.datacollector_populations.get_model_vars_dataframe()
    current_populations = []
    for population_data in all_population_data:
        populations = all_population_data[population_data]
        current_populations.append(populations[model.number_of_steps])
    #   current_populations = [a[model.number_of_steps - 1] for a in [population_data for population_data in all_population_data]]
    plt.figure(4, )
    plt.pie(current_populations, labels=[a for a in all_population_data.columns])


def collect_dependent_raw_data(model, dependent_name):
    if dependent_name == "extinction_probability" or dependent_name == "extinction_time":
        return calculate_extinction_time(model)


def calculate_dependent(raw_data, dependent_name):
    if dependent_name == "extinction_probability":
        return calculate_extinction_prob(raw_data)


def calculate_dependent_error(raw_data, dependent_name, num_sim_batches):
    if dependent_name == "extinction_probability":
        return calculate_extinction_prob_sd(raw_data, num_sim_batches)

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


def calculate_extinction_prob(raw_data):
    dependent_successful = list(filter(lambda a: a != np.inf, raw_data))
    return len(dependent_successful) / len(raw_data)


def calculate_extinction_prob_sd(raw_data, num_sim_batches):

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
        split_extinction_prob.append(calculate_extinction_prob(split))
    return statistics.stdev(split_extinction_prob)




def cross_sectional_graph():
    pass
