import numpy as np
import scipy.fftpack
from matplotlib import pyplot as plt
import ternary
import random

def fft_analysis(model):
    all_population_data = model.datacollector_populations.get_model_vars_dataframe()
    # surprisingly this iterates over columns, not rows
    for population_data in all_population_data:
        N = len(all_population_data[population_data])
#        print(N)
#        print(all_population_data[population_data])
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
#    print("Dominant frequency >> ", t_corrected[np.argmax(y_corrected)])

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

def pie_chart(model):
     all_population_data = model.datacollector_populations.get_model_vars_dataframe()
     current_populations =[]
     for population_data in all_population_data:
         populations = all_population_data[population_data]
         current_populations.append(populations[model.number_of_steps])
#     current_populations = [a[model.number_of_steps - 1] for a in [population_data for population_data in all_population_data]]
     plt.figure(4,)
     plt.pie(current_populations, labels=[a for a in all_population_data.columns])
     
    
        
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