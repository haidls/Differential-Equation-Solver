import numpy.linalg
from tensorflow import keras

import data
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os

import save_data

input_length = 2
input_width = 1
test_data_amount = 10
test_repetitions = 1


"""def test_function(F, model, start, name):
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
    plt.title("ODE solution for right hand side " + str(name) + " with starting value " + str(start))"""


if __name__ == '__main__':
    file_path = 'data_sets/data_set_lwdr_%d_%d_%d_%d.npy' % (input_length, input_width, test_data_amount,
                                                             test_repetitions)
    if not os.path.exists(file_path):
        save_data.generate_data(input_length, input_width, test_data_amount, test_repetitions)

    with open(file_path, 'rb') as f:
        training_input = numpy.load(f)
        training_output = numpy.load(f)
        testing_input = numpy.load(f)
        testing_output = numpy.load(f)

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

    model.fit(training_input, training_output, batch_size=32, epochs=10)

    """results = numpy.zeros((len(testing_output), test_repetitions))
    new_values = numpy.zeros((len(testing_output), 1))
    new_diff = numpy.zeros((len(testing_output), 1))
    testing_input_modified = numpy.array(testing_input, copy=True)
    for i in range(0, test_repetitions-1):
        results[:, i] = (model(testing_input_modified).numpy())[:, 0]
        testing_input_values = testing_input_modified[:, 1:input_length]
        testing_input_diff = testing_input_modified[:, input_length+1:]
        new_values[:, 0] = results[:, i]
        for j in range(0, len(new_values)):
            F = data.function(test_coeff[j])
            new_diff[j, 0] = F(t_eval[input_length - 1 + i], new_values[j, 0])
        testing_input_modified = numpy.concatenate((testing_input_values, new_values, testing_input_diff, new_diff),
                                                   axis=1)
    results[:, test_repetitions-1] = (model(testing_input_modified).numpy())[:, 0]"""

    for i in range(0, (testing_output.shape())[0]):


    results = (model(testing_input).numpy())[:, 0]

    total_error = results - testing_output
    error_norm = numpy.linalg.norm(total_error, 2)
    error = error_norm / len(testing_output)
    print('the approximation error is %f' % error)

    x_values = numpy.array(range(0, 11)) / 10
    for i in range(0, int(len(testing_output)/9)):
        plt.figure()
        exact = numpy.zeros(11)
        exact[0] = testing_input[9 * i, 0]
        exact[1:10] = testing_output[9 * i: 9 * i + 9]
        exact[10] = testing_input[9 * i + 8, -1]
        approximation = numpy.zeros(11)
        approximation[0] = testing_input[9 * i, 0]
        approximation[1:10] = results[9 * i: 9 * i + 9]
        approximation[10] = testing_input[9 * i + 8, -1]
        plt.plot(x_values, exact)
        plt.plot(x_values, approximation)
        plt.legend(["exact solution", "approximate solution"])
        plt.title("Plot number " + str(i))

    """f = lambda t, x: math.exp(x/10)/10
    test_function(f, model, 1, "exp(x/10)")

    g = lambda t, x: math.sqrt(abs(x))/5
    test_function(g, model, 1, "sqrt(abs(x))")"""
    plt.show()

