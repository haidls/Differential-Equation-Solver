"""
A program that trains neural networks to solve multi dimensional ordinary differential equations of order one with a
specified right hand side (currently the lorenz attractor).
author: Linus Haid
version: 22.06.2023
"""

import os
import time

import numpy.linalg
from tensorflow import keras

import data
import save_data
import matplotlib.pyplot as plt

input_length = 3
test_data_amount = 100
test_repetitions = 28
coeff_value_amount = 7


def plot_3d(testing_input, testing_output, results):
    """
    Plots the results of the test for 3D ODEs.
    :param testing_input: input data for the model
    :param testing_output: expected output data of the model
    :param results: actual output data of the model
    """
    for i in range(0, len(testing_output)):
        figure = plt.figure()
        ax = figure.add_subplot(projection='3d')
        output_plot = []
        results_plot = []
        for j in range(0, 3):
            input_val = testing_input[i, j:3*input_length+j:3]
            output_plot.append(numpy.append(input_val, testing_output[i, j, :]))
            results_plot.append(numpy.append(input_val, results[i, j, :]))

        ax.plot(output_plot[0], output_plot[1], output_plot[2])
        ax.plot(results_plot[0], results_plot[1], results_plot[2])
        ax.legend(["exact solution", "approximate solution"])
        plt.title("Plot number " + str(i))
        plt.show()
        plt.close(figure)
        time.sleep(2)


def load_data():
    """
    Loads the data if it already exists, generates new data otherwise
    :return: the data necessary for building a model
    """
    file_path = 'data_sets/data_set_lwdri_%d_%d_%d_%d.npy' % (
        coeff_value_amount, input_length, test_repetitions, test_data_amount)
    if not os.path.exists(file_path):
        save_data.generate_data(coeff_value_amount, input_length, test_data_amount, test_repetitions, file_path)
    with open(file_path, 'rb') as f:
        training_input = numpy.load(f)
        training_output = numpy.load(f)
        testing_input = numpy.load(f)
        testing_output = numpy.load(f)
        t_eval = numpy.load(f)
        test_coeff = numpy.load(f)
    return training_input, training_output, testing_input, testing_output, t_eval, test_coeff


def build_model(dim, training_input, training_output):
    """
    Creates a model and fits it to the data
    :param dim: dimension of the ODE
    :param training_input: input data for the model
    :param training_output: expected output data of the model
    :return: a model fitted to the data
    """
    model = keras.Sequential(
        [
            keras.layers.Dense(16, activation="relu", name="layer1"),
            keras.layers.Dense(12, activation="relu", name="layer2"),
            keras.layers.Dense(dim, name="layer3"),
        ]
    )
    model.compile(loss=keras.losses.MeanSquaredError(),
                  optimizer=keras.optimizers.Adam(),
                  metrics=[keras.metrics.MeanAbsoluteError()])
    model.fit(training_input, training_output, batch_size=32, epochs=20)
    return model


def test_model(model, dim, testing_input, test_coeff):
    """
    Uses the model to approximate the solutions of ODEs.
    :param model: model fitted to the data
    :param dim: dimension of the ODE
    :param testing_input: input data for the model
    :param test_coeff: coefficients of the right hand sides of the ODEs
    :return: approximation of the ODEs solution
    """
    results = numpy.zeros((len(testing_input), dim, test_repetitions))
    new_values = numpy.zeros((len(testing_input), dim))
    new_diff = numpy.zeros((len(testing_input), dim))
    testing_input_modified = numpy.array(testing_input, copy=True)
    for i in range(0, test_repetitions - 1):
        results[:, :, i] = (model(testing_input_modified).numpy())
        testing_input_values = testing_input_modified[:, dim:input_length * dim]
        testing_input_diff = testing_input_modified[:, (input_length + 1) * dim:]
        new_values[:, :] = results[:, :, i]
        for j in range(0, len(new_values)):
            F = data.function(test_coeff[j])
            new_diff[j, :] = F(t_eval[input_length - 1 + i], new_values[j, :])
        testing_input_modified = numpy.concatenate((testing_input_values, new_values, testing_input_diff, new_diff),
                                                   axis=1)
    results[:, :, test_repetitions - 1] = (model(testing_input_modified).numpy())
    return results


def print_error(results, testing_output):
    """
    Calculates and plots the error of the model.
    :param results: actual output data of the model
    :param testing_output: expected output data of the model
    """
    total_error = results[:, :, -1] - testing_output[:, :, -1]
    error_norm = numpy.linalg.norm(total_error, 2)
    error = error_norm / len(testing_output)
    print('the approximation error is %f' % error)


if __name__ == '__main__':
    dim = data.dimensions
    training_input, training_output, testing_input, testing_output, t_eval, test_coeff = load_data()
    model = build_model(dim, training_input, training_output)
    results = test_model(model, dim, testing_input, test_coeff)
    print_error(results, testing_output)
    if dim == 3:
        plot_3d(testing_input, testing_output, results)
