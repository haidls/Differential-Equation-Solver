import math

import numpy.linalg
from tensorflow import keras

import data
from data import get_data
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

input_length = 3
test_data_amount = 100
test_repetitions = 8


def test_function(F, model, start, name):
    interval = [0, 0.1 * (input_length + test_repetitions - 0.9)]
    t_eval_func = numpy.linspace(0, (input_length + test_repetitions - 1) * 0.1, input_length + test_repetitions)
    solution = solve_ivp(F, interval, [start], t_eval=t_eval_func)
    func_input = numpy.zeros((1, 2 * input_length))
    func_input[0, :], s = data.format_input(input_length, solution, F, t_eval_func)
    func_output = solution.y[0]

    results = numpy.zeros(input_length + test_repetitions)
    results[0:input_length] = func_output[0:input_length]
    for i in range(0, test_repetitions-1):
        results[input_length + i] = (model(func_input).numpy())[:, 0]
        func_input[0, 0:input_length] = results[i + 1:input_length + i + 1]
        func_input[0, input_length:-1] = func_input[0, input_length+1:]
        func_input[-1, 0] = F(0, func_input[0, input_length-1])
    results[input_length + test_repetitions-1] = (model(func_input).numpy())[:, 0]
    plt.figure()
    plt.plot(t_eval_func, func_output)
    plt.plot(t_eval_func, results)
    plt.legend(["exact solution", "approximate solution"])
    plt.title("ODE solution for right hand side " + str(name) + " with starting value " + str(start))


def plot_3d(testing_input, testing_output, results):
    for i in range(0, len(testing_output)):
        ax = plt.figure().add_subplot(projection='3d')
        output_plot = []
        results_plot = []
        for j in range(0,3):
            input_val = testing_input[i, j:3:3*input_length+j]
            output_plot.append(numpy.append(input_val, testing_output[:, j, :]))
            results_plot.append(numpy.append(input_val, results[:, j, :]))

        ax.plot(output_plot[0], output_plot[1], output_plot[2])
        ax.plot(results_plot[0], results_plot[1], results_plot[2])
        ax.legend(["exact solution", "approximate solution"])
        plt.title("Plot number " + str(i))


if __name__ == '__main__':
    training_input, training_output, testing_input, testing_output, t_eval, test_coeff = \
        get_data(coefficient_value_amount=9, input_length=input_length, test_repetitions=test_repetitions,
                 test_data_amount=test_data_amount)

    dim = data.dimensions
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
    results = numpy.zeros((len(testing_output), dim, test_repetitions))
    new_values = numpy.zeros((len(testing_output), dim))
    new_diff = numpy.zeros((len(testing_output), dim))
    testing_input_modified = numpy.array(testing_input, copy=True)
    for i in range(0, test_repetitions-1):
        results[:, :, i] = (model(testing_input_modified).numpy())
        testing_input_values = testing_input_modified[:, dim:input_length*dim]
        testing_input_diff = testing_input_modified[:, (input_length+1)*dim:]
        new_values[:, :] = results[:, :, i]
        for j in range(0, len(new_values)):
            F = data.function(test_coeff[j])
            new_diff[j, :] = F(t_eval[input_length - 1 + i], new_values[j, :])
        testing_input_modified = numpy.concatenate((testing_input_values, new_values, testing_input_diff, new_diff),
                                                   axis=1)
    results[:, :, test_repetitions-1] = (model(testing_input_modified).numpy())
    total_error = results[:, :, -1] - testing_output[:, :, -1]
    error_norm = numpy.linalg.norm(total_error, 2)
    error = error_norm / len(testing_output)
    print('the approximation error is %f' % error)

    plot_3d(testing_input, testing_output, results)

    """f = lambda t, x: math.exp(x/10)/10
    test_function(f, model, 1, "exp(x/10)")

    g = lambda t, x: math.sqrt(abs(x))/5
    test_function(g, model, 1, "sqrt(abs(x))") """
    plt.show()
