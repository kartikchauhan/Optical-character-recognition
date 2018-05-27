
import pickle
import gzip
import numpy as np

def load_data():

    objects = []

    PIK = "/home/killwithme/Desktop/projects/final_year_project/custom-data-pickle.dat"

    with (open(PIK, "rb")) as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break

    training_set, test_set = objects[0]

    return (training_set, test_set)


def load_data_wrapper():
    tr_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, test_data)

def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
