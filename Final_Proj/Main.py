
import numpy as np

np.random.seed(3)
LEARNING_RATE = 0.01
EPOCH = 20

MASTER_FILE = 'chrome.out'

def read_dataset():
    file_read = np.loadtxt(MASTER_FILE)
    for i in file_read:
        print(i)