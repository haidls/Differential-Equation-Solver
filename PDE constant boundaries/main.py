"""
A program that trains neural networks to solve one dimensional partial differential equations of order one with a
specified right hand side (currently the heat equation) and constant boundaries.
author: Linus Haid
version: 22.06.2023
"""

import time

import numpy.linalg
from tensorflow import keras
import matplotlib.pyplot as plt
import os

import save_data

input_length = 3
input_width = 1
test_data_amount = 30
test_repetitions = 5
interval_amount = 100
coefficient_value_amount = 7


def load_data():
    """
    Loads the data if it already exists, generates new data otherwise
    :return: the data necessary for building a model
    """
    file_path = 'data_sets/data_set_lwdri_%d_%d_%d_%d_%d.npy' % (input_length, input_width, test_data_amount,
                                                                 test_repetitions, interval_amount)
    if not os.path.exists(file_path):
        save_data.generate_data(coefficient_value_amount, input_length, input_width, test_data_amount, test_repetitions,
                                interval_amount, file_path)
    with open(file_path, 'rb') as f:
        training_input = numpy.load(f)
        training_output = numpy.load(f)
        testing_input = numpy.load(f)
        testing_output = numpy.load(f)
    return training_input, training_output, testing_input, testing_output


def build_model(training_input, training_output):
    """
    Creates a model and fits it to the data
    :param training_input: input data for the model
    :param training_output: expected output data of the model
    :return: a model fitted to the data
    """
    model = keras.Sequential(
        [
            keras.layers.Dense(16, activation="relu", name="layer1"),
            keras.layers.Dense(10, activation="relu", name="layer2"),
            keras.layers.Dense(1, name="layer3"),
        ]
    )
    model.compile(loss=keras.losses.MeanSquaredError(),
                  optimizer=keras.optimizers.Adam(),
                  metrics=[keras.metrics.MeanAbsoluteError()])
    model.fit(training_input, training_output, batch_size=32, epochs=50)
    return model


def test_model(testing_input, model):
    """
    Uses the model to approximate the solutions of PDEs.
    :param model: model fitted to the data
    :param testing_input: input data for the model
    :return: approximation of the PDEs solution
    """
    results = numpy.zeros((test_data_amount, input_length + test_repetitions, interval_amount + 1))
    results[:, 0:input_length, :] = testing_input
    shape = (test_data_amount, input_length * (2 * input_width + 1))
    for i in range(0, test_repetitions):
        results[:, input_length + i, [0, interval_amount]] = results[:, input_length + i - 1, [0, interval_amount]]
        for j in range(input_width, interval_amount - input_width + 1):
            input = numpy.reshape(results[:, i:input_length + i, j - input_width:j + input_width + 1], shape)
            results[:, input_length + i, j] = (model(input).numpy())[:, 0]
    return results


def print_error(results, testing_output):
    """
    Calculates and plots the error of the model.
    :param results: actual output data of the model
    :param testing_output: expected output data of the model
    """
    total_error = results[:, -1, input_width:interval_amount - input_width + 1] \
                  - testing_output[:, -1, input_width:interval_amount - input_width + 1]
    error_norm = numpy.linalg.norm(total_error, 2)
    error = error_norm / total_error.size
    print('the approximation error is %f' % error)


def plot_results(testing_input, testing_output, results):
    """
    Plots the results of the test.
    :param testing_input: input data for the model
    :param testing_output: expected output data of the model
    :param results: actual output data of the model
    """
    x_values = numpy.array(range(0, interval_amount + 1)) / interval_amount
    plt.ioff()
    for i in range(0, test_data_amount):
        for j in range(0, input_length):
            figure = plt.figure()
            plt.plot(x_values, testing_input[i, j, :])
            plt.legend(["input"])
            plt.title("Test number " + str(i) + ", t = " + str(j / 10))
            plt.show()
            plt.close(figure)
            time.sleep(2)
        for k in range(0, test_repetitions):
            figure = plt.figure()
            plt.plot(x_values, testing_output[i, k, :])
            plt.plot(x_values, results[i, input_length + k, :])
            plt.legend(["exact solution", "approximate solution"])
            plt.title("Test number " + str(i) + ", t = " + str((input_length + k) / 10))
            plt.show()
            plt.close(figure)
            time.sleep(2)


if __name__ == '__main__':
    training_input, training_output, testing_input, testing_output = load_data()
    model = build_model(training_input, training_output)
    results = test_model(testing_input, model)
    print_error(results, testing_output)
    plot_results(testing_input, testing_output, results)


