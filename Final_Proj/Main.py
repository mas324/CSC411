
from cProfile import label
import os
import re
from matplotlib import pyplot as plt
import numpy as np

np.random.seed(13)
LEARNING_RATE = 0.01
EPOCH = 5

#Number of input types
input_neurons_count = 5
#Number of layers wanted
hidden_neurons_count = 10
#Number of output types
output_neurons_count = 1
DATA_FILE = 'chrome.out'
LOCAL = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

def read_dataset():
    #Read in as a 2D array 1048576 x 7
    #File must be in format f, f, f, f, f, f, f
    file_read = np.loadtxt(os.path.join(LOCAL, DATA_FILE))
    
    #Will only need 5 datapoints for this assignment
    #Iterate over data, removing green and blue outputs
    data_new = np.empty((len(file_read), 5))
    for i in range(len(file_read)):
        data_new[i] = file_read[i][:-2]
    
    #Split data, half to train and half to test, and standardize
    data_split = np.split(data_new, 2)
    x_train = data_split[0]
    mean = np.mean(x_train)
    stddev = np.std(x_train)
    x_train = (x_train - mean) / stddev
    x_test = (data_split[1] - mean) / stddev
    

    #One-hot encode
    y_train = np.zeros((len(x_train), output_neurons_count))
    y_test = np.zeros((len(x_test), output_neurons_count))
    y_train[np.random.randint(len(y_train))][np.random.randint(output_neurons_count)] = 1
    y_test[np.random.randint(len(y_test))][np.random.randint(output_neurons_count)] = 1
    return x_train, x_test, y_train, y_test
    
x_train, x_test, y_train, y_test = read_dataset()
index_list = list(range(len(x_train)))

def neuron_w(neuron_count, input_count):
    weights = np.zeros((neuron_count, input_count + 1))
    for n in range(neuron_count):
        for i in range(1, (input_count + 1)):
            weights[n][i] = np.random.uniform(-0.1, 0.1)
    return weights

hidden_weight = neuron_w(hidden_neurons_count, input_neurons_count)
hidden_layer_y = np.zeros(hidden_neurons_count)
hidden_error = np.zeros(hidden_neurons_count)

output_weight = neuron_w(output_neurons_count, hidden_neurons_count)
output_layer_y = np.zeros(output_neurons_count)
output_error = np.zeros(output_neurons_count)

chart_x = []
chart_y_train = []
chart_y_test = []
def show_learning(epoch, train, test):
    global chart_x, chart_y_train, chart_y_test
    print('epoch:', epoch, ', training:', '%6.4f' % train, ', testing:', '%6.4f' % test)
    chart_x.append(epoch + 1)
    chart_y_train.append(1.0 - train)
    chart_y_test.append(1.0 - test)

def plot_learning():
    plt.plot(chart_x, chart_y_train, 'r-', label='training error')
    plt.plot(chart_x, chart_y_test, 'b-', label='test error')
    plt.axis((0, len(chart_x), 0.0, 1.0))
    plt.xlabel('training epochs')
    plt.ylabel('error')
    plt.legend()
    plt.show()

    
def forward_pass(x):
    global hidden_layer_y, output_layer_y
    for i, w in enumerate(hidden_weight):
        z = np.dot(w, x)
        hidden_layer_y[i] = np.tanh(z)
    hidden_output = np.concatenate((np.array([1.0]), hidden_layer_y))
    for i, w in enumerate(output_weight):
        z = np.dot(w, hidden_output)
        output_layer_y[i] = 1.0 / (1.0 + np.exp(-z))
        
def backward_pass(y_truth):
    global hidden_error, output_error
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
    
def adjust_weights(x):
    global output_weight, hidden_weight
    for i, err in enumerate(hidden_error):
        hidden_weight[i] -= (x * LEARNING_RATE * err)
    hidden_array = np.concatenate((np.array([1.0]), hidden_layer_y))
    for i, err in enumerate(output_error):
        output_weight[i] -= (hidden_array * LEARNING_RATE * err)
        
#Start training and testing
for i in range(EPOCH):
    np.random.shuffle(index_list)
    correct_training = 0
    for j in index_list:
        x = np.concatenate((np.array([1.0]), x_train[j]))
        forward_pass(x)
        if output_layer_y.argmax() == y_train[j].argmax():
            correct_training += 1
        backward_pass(y_train[j])
        adjust_weights(x)
    correct_testing = 0
    for j in range(len(x_test)):
        x = np.concatenate((np.array([1.0]), x_test[j]))
        forward_pass(x)
        if output_layer_y.argmax() == y_test[j].argmax():
            correct_testing += 1
    show_learning(i, correct_training/len(x_train), correct_testing/len(x_test))
plot_learning()