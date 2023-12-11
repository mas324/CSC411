"""
Author: Shawn Merana
Class: CSC 411 - Dr. Hollister
Project Final
Date: 12/10/2023
"""

import os
from matplotlib import pyplot as plt
import numpy as np

np.random.seed(13)
LEARNING_RATE = 0.05
EPOCH = 25

# Number of inputs
# theta in, phi in, theta out, phi out, red
INPUT_NEURON_COUNT = 5
# Number of hidden neurons
HIDDEN_NEURON_COUNT = 10
# Number of outputs
# red
OUTPUT_NEURON_COUNT = 1

# Data to use, as well as an autopath to workaround 'file not found' error
# when the file is in the same directory
DATA_FILE = 'blue-rubber.txt'
LOCAL = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

def read_dataset():
    # Read in as a 2D array 1048576 x 7
    # File must be in format f, f, f, f, f, f, f
    file_read = np.loadtxt(os.path.join(LOCAL, DATA_FILE))
    
    # Shuffle the data
    np.random.shuffle(file_read)
    data_length = len(file_read)
    
    # Will only need 5 inputs 1 output for this assignment (411 class)
    # Organize and clean data, removing green and blue inputs
    data_x = np.empty((data_length, 5))
    data_y = np.empty(data_length)
    for i in range(len(file_read)):
        data_x[i] = file_read[i][:-2] # Ignore last 2 elements (green, blue)
        data_y[i] = file_read[i][4] # Add red to output data
        
    # Split data, half to train and half to test
    x_train, x_test = np.split(data_x, 2)
    
    # Standardize data
    mean = np.mean(x_train)
    stddev = np.std(x_train)
    x_train = (x_train - mean) / stddev
    x_test = (x_test - mean) / stddev
    
    # Create ground truths with red data
    y_train, y_test = np.split(data_y, 2)

    return x_train, x_test, y_train, y_test
    
x_train, x_test, y_train, y_test = read_dataset()
index_list = list(range(len(x_train)))

def neuron_w(neuron_count, input_count):
    weights = np.zeros((neuron_count, input_count + 1))
    for n in range(neuron_count):
        for i in range(1, (input_count + 1)):
            weights[n][i] = np.random.uniform(-1, 1)
    return weights

# Setup neurons/input per neuron
hidden_layer_w = neuron_w(HIDDEN_NEURON_COUNT, INPUT_NEURON_COUNT)
hidden_layer_y = np.zeros(HIDDEN_NEURON_COUNT)
hidden_layer_err = np.zeros(HIDDEN_NEURON_COUNT)

output_layer_w = neuron_w(OUTPUT_NEURON_COUNT, HIDDEN_NEURON_COUNT)
output_layer_y = np.zeros(OUTPUT_NEURON_COUNT)
output_layer_err = np.zeros(OUTPUT_NEURON_COUNT)

# Show data and plot after complete
chart_x = []
chart_y_train = []
chart_y_test = []
def show_learning(epoch, train, test):
    global chart_x, chart_y_train, chart_y_test
    print('epoch:', epoch, ', training acc:',
          '%6.4f' % train, ', testing acc:', '%6.4f' % test)
    chart_x.append(epoch + 1)
    chart_y_train.append(1.0 - train)
    chart_y_test.append(1.0 - test)

def plot_learning():
    plt.plot(chart_x, chart_y_train, 'r-', label='training error')
    plt.plot(chart_x, chart_y_test, 'b-', label='test error')
    plt.axis((0, len(chart_x), 0.0, 1.0))
    plt.xlabel('training epochs')
    plt.ylabel('error')
    plt.title(('Learning rate:' + '%6.4f' % LEARNING_RATE))
    plt.legend()
    plt.savefig(os.path.join(LOCAL, 'pyplot.png'))
    plt.show()

def forward_pass(x):
    global hidden_layer_y, output_layer_y
    hidden_layer_z = np.matmul(hidden_layer_w, x)
    hidden_layer_y = np.tanh(hidden_layer_z)
    
    #for i, w in enumerate(hidden_weight):
    #    z = np.dot(w, x)
    #    hidden_layer_y[i] = np.tanh(z)
    
    hidden_output = np.concatenate((np.array([1.0]), hidden_layer_y))
    
    #for i, w in enumerate(output_weight):
    #    z = np.dot(w, hidden_output)
    #    output_layer_y[i] = 1.0 / (1.0 + np.exp(-z))
    
    output_layer_z = np.matmul(output_layer_w, hidden_output)
    output_layer_y = 1.0 / (1.0 + np.exp(-output_layer_z))
        
def backward_pass(y_truth):
    global hidden_layer_err, output_layer_err
    error_prime = -(y_truth - output_layer_y)
    output_log_prime = output_layer_y * (1.0 - output_layer_y)
    output_layer_err = error_prime * output_log_prime
    hidden_tanh_prime = 1.0 - hidden_layer_y**2
    hidden_weight_error = np.matmul(np.matrix.transpose(output_layer_w[:,1:]), output_layer_err)
    hidden_layer_err = hidden_tanh_prime * hidden_weight_error
    
    '''Changed to use matrix as stated in appendix F of LDL
    
    for i, y in enumerate(output_layer_y):
        error_p = -(y_truth[i] - y)
        derive = y * (1.0 - y)
        output_error[i] = error_p * derive
    for i, y in enumerate(hidden_layer_y):
        error_w = []
        for w in output_weight:
            error_w.append(w[i+1])
        error_w_array = np.array(error_w)
        derive = 1.0 - y**2
        weight_error = np.dot(error_w_array, output_error)
        hidden_error[i] = weight_error * derive
    '''
    
def adjust_weights(x):
    global output_layer_w, hidden_layer_w
    delta_matrix = np.outer(hidden_layer_err, x) * LEARNING_RATE
    hidden_layer_w -= delta_matrix
    hidden_out_arr = np.concatenate((np.array([1.0]), hidden_layer_y))
    delta_matrix = np.outer(output_layer_err, hidden_out_arr) * LEARNING_RATE
    output_layer_w -= delta_matrix
    
    ''' Changed to use matrix as stated in appendix F of LDL
    
    for i, err in enumerate(hidden_error):
        hidden_weight[i] -= (x * LEARNING_RATE * err)
    hidden_array = np.concatenate((np.array([1.0]), hidden_layer_y))
    for i, err in enumerate(output_error):
        output_weight[i] -= (hidden_array * LEARNING_RATE * err)
    '''
        
# Start training and testing
for i in range(EPOCH):
    np.random.shuffle(index_list)
    ACC_TOLERANCE = 0.005
    correct_training = 0
    for j in index_list:
        x = np.concatenate((np.array([1.0]), x_train[j]))
        forward_pass(x)
        # Tolerance of +-ACC_TOL
        if ((output_layer_y > y_train[j] - ACC_TOLERANCE)
            and (output_layer_y < y_train[j] + ACC_TOLERANCE)):
            correct_training += 1
        backward_pass(y_train[j])
        adjust_weights(x)
    correct_testing = 0
    for j in range(len(x_test)):
        x = np.concatenate((np.array([1.0]), x_test[j]))
        forward_pass(x)
        if ((output_layer_y > y_test[j] - ACC_TOLERANCE)
            and (output_layer_y < y_test[j] + ACC_TOLERANCE)):
            correct_testing += 1
    show_learning(i + 1, correct_training/len(x_train), correct_testing/len(x_test))
plot_learning()
with open("weights.txt", "a") as f:
    print('hidden layer stats:\n', hidden_layer_w, hidden_layer_y, file=f)
    print('output layer stats:\n', output_layer_w, output_layer_y, file=f)
    print('\n\nLearning rate:', LEARNING_RATE, file=f)
