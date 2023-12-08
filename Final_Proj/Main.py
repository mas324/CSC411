
from cProfile import label
import re
from matplotlib import pyplot as plt
import numpy as np

np.random.seed(13)
LEARNING_RATE = 0.01
EPOCH = 5

input_neurons = 64 * 64
hidden_neurons = 16
output_neurons = 8
MASTER_FILE = 'Final_Proj/chrome.out'

def read_dataset():
    #Read in as a 2D array 1048576 x 7
    file_read = np.loadtxt(MASTER_FILE, dtype={'names':('theta_i', 'phi_i', 'theta_o', 'phi_o', 'red', 'green', 'blue'),
                                               'formats':('f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4')})

    x_train = np.split(file_read, 2)[0]
    x_test = file_read
    y_train = np.zeros(len(file_read / 2))
    y_test = np.zeros(len(file_read))
    return x_train, x_test, y_train, y_test
    
x_train, y_train, x_test, y_test = read_dataset()
index_list = list(range(len(x_train)))

def neuron_w(input_count):
    weights = np.zeros(input_count + 1)
    for i in range(1, (input_count + 1)):
        weights[i] = np.random.uniform(-1.0, 1.0)
    return weights

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
    plt.axis([0, len(chart_x), 0.0, 1.0])
    plt.xlabel('training epochs')
    plt.ylabel('error')
    plt.legend()
    plt.show()

    
def forward_pass(x):
    return