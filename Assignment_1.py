import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

#os.chdir(r'C:\Users\josep\python_files\CptS_570\assignment_1_570')
from classification import *
from perceptron import *
from averaged_perceptron import *
from passive_aggressive import *

#==============================================================================
#                       Load in data
#==============================================================================
data = input_data.read_data_sets('./data/fashion')

# training labels
labels_raw = data.train.labels
labels = [-1 if x%2 == 0 else 1 for x in labels_raw]
labels = np.array(labels)

# test labels
test_labels_raw = data.test.labels
test_labels = [-1 if x%2 == 0 else 1 for x in test_labels_raw]
test_labels = np.array(test_labels)

# training features
images = data.train.images

# test feaures
test_images = data.test.images

#==============================================================================
#                       View the Images
#==============================================================================
#for i in range(4):
#    view_image(images[i])

#==============================================================================
#                       5.1 - Binary Classification
#==============================================================================

iterations_a = 50
iterations_b_d = 20

# a) Learning Curve for Perceptron and PA (50 iterations)
w, mistakes = perceptron(images, labels, iterations=iterations_a)
w_pa, mistakes_pa = pa(images, labels, iterations=iterations_a)
#plot_data(pd.concat([mistakes, mistakes_pa], axis=1), y_lst = ['perceptron mistakes','passive aggressive mistakes'], filename='./figures/5_1/5_1_a.')
print('Completed part 5.1 a')

# b)
percept_scores = acc_scores(iterations_b_d, perceptron, images, test_images, labels, test_labels)
pa_scores = acc_scores(iterations_b_d, pa, images, test_images, labels, test_labels)
#plot_data(percept_scores, y_lst = percept_scores.columns, filename='./figures/5_1/5_1_b_perceptron.')
#plot_data(pa_scores, y_lst = pa_scores.columns, filename='./figures/5_1/5_1_b_pa.')
print('Completed part 5.1 b')

# c)
avg_percept_scores = acc_scores(iterations_b_d, avg_perceptron, images, test_images, labels, test_labels)
#plot_data(avg_percept_scores, y_lst = avg_percept_scores.columns, filename='./figures/5_1/5_1_c.')
print('Completed part 5.1 c')

# d)
g_learning_df = g_learning_curve(perceptron, images, test_images, labels, test_labels, iterations=iterations_b_d)
#plot_data(g_learning_df, y_lst = g_learning_df.columns, filename='./figures/5_1/5_1_d.')
print('Completed part 5.1 d')

#==============================================================================
#                       Output binary data
#==============================================================================
mistakes.to_csv(r'./output/5_1/5_1_a_perceptron.csv')
mistakes_pa.to_csv(r'./output/5_1/5_1_a_passive.csv')
g_learning_df.to_csv(r'./output/5_1/5_1_d_accuracy.csv')
avg_percept_scores.to_csv(r'./output/5_1/5_1_c.csv')
pa_scores.to_csv(r'./output/5_1/5_1_b_pa.csv')
percept_scores.to_csv(r'./output/5_1/5_1_b_perceptron.csv')



#==============================================================================
#                       5.2 - Multi-Class Classification
#==============================================================================

# a) Learning Curve for Perceptron and PA (50 iterations)
w, mistakes = perceptron_MC(images, labels_raw, iterations=iterations_a)
w_pa, mistakes_pa = pa_MC(images, labels_raw, iterations=iterations_a)

plot_data(pd.concat([mistakes, mistakes_pa], axis=1), y_lst = ['perceptron mistakes','passive aggressive mistakes'], filename='./figures/5_2/5_2_a.')
print('Completed part 5.2 a')

# b)
percept_scores = acc_scores(iterations_b_d, perceptron_MC, images, test_images, labels_raw, test_labels_raw, binary=False)
pa_scores = acc_scores(iterations_b_d, pa_MC, images, test_images, labels_raw, test_labels_raw, binary=False)
plot_data(percept_scores, y_lst = percept_scores.columns, filename='./figures/5_2/5_2_b_perceptron.')
plot_data(pa_scores, y_lst = pa_scores.columns, filename='./figures/5_2/5_2_b_pa.')
print('Completed part 5.2 b')

# c)
avg_percept_scores = acc_scores(iterations_b_d, avg_perceptron_MC, images, test_images, labels_raw, test_labels_raw, binary=False)
plot_data(avg_percept_scores, y_lst = avg_percept_scores.columns, filename='./figures/5_2/5_2_c.')
print('Completed part 5.2 c')

# d)
d = {}
iterations_b_d = 1
func_names = ['perceptron_MC', 'pa_MC', 'avg_perceptron_MC']
for func, func_name in zip([perceptron_MC, pa_MC, avg_perceptron_MC], func_names):
    g_learning_df = g_learning_curve(func, images, test_images, labels_raw, test_labels_raw, iterations=iterations_b_d, binary=False)
    d[func_name] = g_learning_df

#g_learning_df = g_learning_curve(perceptron_MC, images, test_images, labels_raw, test_labels_raw, iterations=iterations_b_d, binary=False)
plot_data(g_learning_df, y_lst = g_learning_df.columns, filename='./figures/5_2/5_2_d.')
print('Completed part 5.2 d')

#==============================================================================
#                       Output MC data
#==============================================================================
mistakes.to_csv(r'./output/5_2/5_2_a_perceptron.csv')
mistakes_pa.to_csv(r'./output/5_2/5_2_a_passive.csv')
g_learning_df.to_csv(r'./output/5_2/5_2_d_accuracy.csv')
avg_percept_scores.to_csv(r'./output/5_2/5_2_c.csv')
pa_scores.to_csv(r'./output/5_2/5_2_b_pa.csv')
percept_scores.to_csv(r'./output/5_2/5_2_b_perceptron.csv')
