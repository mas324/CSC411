
import os
from matplotlib import pyplot as plt
import numpy as np

np.random.seed(13)
LEARNING_RATE = 0.1
EPOCH = 20

#Number of input types
input_neurons_count = 3
#Number of layers wanted
hidden_neurons_count = 6
#Number of output types
output_neurons_count = 3
DATA_FILE = 'chrome.out'
LOCAL = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


def read_dataset():
    #Read in as a 2D array 1048576 x 7
    #File must be in format f, f, f, f, f, f, f
    file_read = np.loadtxt(os.path.join(LOCAL, DATA_FILE))
    
    #Will only need 3 datapoints for this assignment
    #Iterate over data, removing green and blue outputs
    #Also compute the vectors of incident and existent
    data_new = np.empty((len(file_read), 3))
    for i in range(len(file_read)):
        data_new[i] = [file_read[i][0] * file_read[i][1],
                       file_read[i][2] * file_read[i][3],
                       file_read[i][4]]
    
    #Split data, half to train and half to test, and standardize
    np.random.shuffle(data_new)
    data_split = np.split(data_new, 2)
    x_train = data_split[0]
    mean = np.mean(x_train)
    stddev = np.std(x_train)
    x_train = (x_train - mean) / stddev
    x_test = (data_split[1] - mean) / stddev
    

    #One-hot encode
    y_train = np.zeros((len(x_train), output_neurons_count))
    y_test = np.zeros((len(x_test), output_neurons_count))
    for i in range(len(y_train)):
        theta_in = np.pi / (np.random.randint(32) + 1)
        phi_in = (np.pi * 2) / (np.random.randint(64) + 1)
        theta_out = np.pi / (np.random.randint(32) + 1)
        phi_out = (np.pi * 2) / (np.random.randint(64) + 1)
        y_train[i][0] = theta_in * phi_in
        y_train[i][1] = theta_out * phi_out
        y_train[i][2] = 0.01
    for i in range(len(y_test)):
        theta_in = np.pi / (np.random.randint(32) + 1)
        phi_in = (np.pi * 2) / (np.random.randint(64) + 1)
        theta_out = np.pi / (np.random.randint(32) + 1)
        phi_out = (np.pi * 2) / (np.random.randint(64) + 1)
        y_test[i][0] = theta_in * phi_in
        y_test[i][1] = theta_out * phi_out
        y_test[i][2] = 0.01
    return x_train, x_test, y_train, y_test
    
x_train, x_test, y_train, y_test = read_dataset()
index_list = list(range(len(x_train)))

def neuron_w(neuron_count, input_count):
    weights = np.zeros((neuron_count, input_count + 1))
    for n in range(neuron_count):
        for i in range(1, (input_count + 1)):
            weights[n][i] = np.random.uniform(-1.0, 1.0)
    return weights

hidden_layer_w = neuron_w(hidden_neurons_count, input_neurons_count)
hidden_layer_y = np.zeros(hidden_neurons_count)
hidden_layer_err = np.zeros(hidden_neurons_count)

output_layer_w = neuron_w(output_neurons_count, hidden_neurons_count)
output_layer_y = np.zeros(output_neurons_count)
output_layer_err = np.zeros(output_neurons_count)

chart_x = []
chart_y_train = []
chart_y_test = []
def show_learning(epoch, train, test):
    global chart_x, chart_y_train, chart_y_test
    print('epoch:', epoch, ', accuracy: training:', '%6.4f' % train, ', testing:', '%6.4f' % test)
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
    
    '''
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
    
    '''for i, err in enumerate(hidden_error):
        hidden_weight[i] -= (x * LEARNING_RATE * err)
    hidden_array = np.concatenate((np.array([1.0]), hidden_layer_y))
    for i, err in enumerate(output_error):
        output_weight[i] -= (hidden_array * LEARNING_RATE * err)
    '''
        
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
