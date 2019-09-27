import pandas as pd
from classification import *
from perceptron import *
from averaged_perceptron import *
from passive_aggressive import *

#==============================================================================
#                       Load in data
#==============================================================================
print('Loading data...')
images, labels, labels_raw, test_images, tes_labels, test_labels_raw = load_data()
print('\nData loaded')

#==============================================================================
#                       5.1 - Binary Classification
#==============================================================================
iterations_a = 50
iterations_b_d = 20

# a) Learning Curve for Perceptron and PA (50 iterations)
print('5.1a processing...')
w, mistakes = perceptron(images, labels, iterations=iterations_a)
w_pa, mistakes_pa = pa(images, labels, iterations=iterations_a)
plot_data(pd.concat([mistakes, mistakes_pa], axis=1), y_lst = ['perceptron mistakes','passive aggressive mistakes'], filename='./figures/5_1/5_1_a.')
print('5.1a done\n')

# b)
print('5.1b processing...')
percept_scores = acc_scores(iterations_b_d, perceptron, images, test_images, labels, test_labels)
pa_scores = acc_scores(iterations_b_d, pa, images, test_images, labels, test_labels)
plot_data(percept_scores, y_lst = percept_scores.columns, filename='./figures/5_1/5_1_b_perceptron.')
plot_data(pa_scores, y_lst = pa_scores.columns, filename='./figures/5_1/5_1_b_pa.')
print('5.1b done\n')

# c)
print('5.1c processing...')
avg_percept_scores = acc_scores(iterations_b_d, avg_perceptron, images, test_images, labels, test_labels)
plot_data(avg_percept_scores, y_lst = avg_percept_scores.columns, filename='./figures/5_1/5_1_c.')
print('5.1c done\n')

# d)
print('5.1d processing...')
g_learning_df = g_learning_curve(perceptron, images, test_images, labels, test_labels, iterations=iterations_b_d)
plot_data(g_learning_df, y_lst = g_learning_df.columns, filename='./figures/5_1/5_1_d.')
print('5.1d done\n')

#==============================================================================
#                       Output binary class data
#==============================================================================
mistakes.to_csv(r'./output/5_1/5_1_a_perceptron.csv')
mistakes_pa.to_csv(r'./output/5_1/5_1_a_passive.csv')
percept_scores.to_csv(r'./output/5_1/5_1_b_perceptron.csv')
pa_scores.to_csv(r'./output/5_1/5_1_b_pa.csv')
avg_percept_scores.to_csv(r'./output/5_1/5_1_c.csv')
g_learning_df.to_csv(r'./output/5_1/5_1_d.csv')

#==============================================================================
#                       5.2 - Multi-Class Classification
#==============================================================================
# a) Learning Curve for Perceptron and PA (50 iterations)
print('5.2a processing...')
w, mistakes = perceptron_MC(images, labels_raw, iterations=iterations_a)
w_pa, mistakes_pa = pa_MC(images, labels_raw, iterations=iterations_a)
plot_data(pd.concat([mistakes, mistakes_pa], axis=1), y_lst = ['perceptron mistakes','passive aggressive mistakes'], filename='./figures/5_2/5_2_a.')
print('5.2a done\n')

# b)
print('5.2b processing...')
percept_scores = acc_scores(iterations_b_d, perceptron_MC, images, test_images, labels_raw, test_labels_raw, binary=False)
pa_scores = acc_scores(iterations_b_d, pa_MC, images, test_images, labels_raw, test_labels_raw, binary=False)
plot_data(percept_scores, y_lst = percept_scores.columns, filename='./figures/5_2/5_2_b_perceptron.')
plot_data(pa_scores, y_lst = pa_scores.columns, filename='./figures/5_2/5_2_b_pa.')
print('5.2b done\n')

# c)
print('5.2c processing...')
avg_percept_scores = acc_scores(iterations_b_d, avg_perceptron_MC, images, test_images, labels_raw, test_labels_raw, binary=False)
plot_data(avg_percept_scores, y_lst = avg_percept_scores.columns, filename='./figures/5_2/5_2_c.')
print('5.2c done\n')

# d)
print('5.2d processing...')
g_learning_df = g_learning_curve(perceptron_MC, images, test_images, labels_raw, test_labels_raw, iterations=iterations_b_d, binary=False)
plot_data(g_learning_df, y_lst = g_learning_df.columns, filename='./figures/5_2/5_2_d.')
print('5.2d done\n')

#==============================================================================
#                       Output data
#==============================================================================
mistakes.to_csv(r'./output/5_2/5_2_a_perceptron.csv')
mistakes_pa.to_csv(r'./output/5_2/5_2_a_passive.csv')
percept_scores.to_csv(r'./output/5_2/5_2_b_perceptron.csv')
pa_scores.to_csv(r'./output/5_2/5_2_b_pa.csv')
avg_percept_scores.to_csv(r'./output/5_2/5_2_c.csv')
g_learning_df.to_csv(r'./output/5_2/5_2_d.csv')
