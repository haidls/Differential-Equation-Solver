import numpy.linalg
from tensorflow import keras
import matplotlib.pyplot as plt
import os

import save_data
import pde

right_hand_side = "(6 * u * d_dx(u) - laplace(d_dx(u)))/10000"
input_length = 2
input_width = 1
test_data_amount = 30
test_repetitions = 7
interval_amount = 70


def test_function(expression, model):
    input_data, output_data = test_solve(expression)
    results = test_apply_model(input_data, model)
    test_plot(expression, input_data, output_data, results)


def test_plot(expression, input_data, output_data, results):
    x_values = (numpy.array(range(0, interval_amount)) + 0.5) / interval_amount
    for j in range(0, input_length):
        plt.figure()
        plt.plot(x_values, input_data[j, :])
        plt.legend(["input"])
        plt.title("Extra test: F(x) = " + expression + " , t = " + str(j / 10))
    for k in range(0, test_repetitions):
        plt.figure()
        plt.plot(x_values, output_data[k, :])
        plt.plot(x_values, results[input_length + k, :])
        plt.legend(["exact solution", "approximate solution"])
        plt.title("Extra test: F(x) = " + expression + " , t = " + str((input_length + k) / 10))
    plt.show()


def test_apply_model(input_data, model):
    width_sym = (2 * input_width + 1)
    results = numpy.zeros((input_length + test_repetitions, interval_amount))
    results[0:input_length, :] = input_data
    for i in range(0, test_repetitions):
        for j in range(0, interval_amount):
            input = numpy.zeros((1, input_length * width_sym))
            for k in range(0, input_length):
                for l in range(0, width_sym):
                    input[0, k * width_sym + l] = results[i + k, (j + l - input_width) % interval_amount]
            results[input_length + i, j] = (model(input).numpy())[:, 0]
    return results


def test_solve(expression):
    grid = pde.CartesianGrid([[0, 1]], [interval_amount], periodic=True)
    field = pde.ScalarField.from_expression(grid, expression)
    equation = pde.PDE({"u": right_hand_side}, bc='periodic')
    storage = pde.MemoryStorage()
    t_range = 0.1 * (input_length + test_repetitions - 1)
    equation.solve(field, t_range=t_range, dt=1e-4, tracker=storage.tracker(0.1))
    data = numpy.vstack(storage.data)
    input_data = (data[0:input_length, :])
    output_data = (data[input_length:, :])
    return input_data, output_data


def load_data():
    file_path = 'data_sets/data_set_lwdri_%d_%d_%d_%d_%d.npy' % (input_length, input_width, test_data_amount,
                                                                 test_repetitions, interval_amount)
    if not os.path.exists(file_path):
        save_data.generate_data(input_length, input_width, test_data_amount, test_repetitions, interval_amount,
                                right_hand_side)
    with open(file_path, 'rb') as f:
        training_input = numpy.load(f)
        training_output = numpy.load(f)
        testing_input = numpy.load(f)
        testing_output = numpy.load(f)
    return training_input, training_output, testing_input, testing_output


def build_model(training_input, training_output):
    model = keras.Sequential(
        [
            keras.layers.Dense(12, activation="relu", name="layer1"),
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
    width_sym = (2 * input_width + 1)
    results = numpy.zeros((test_data_amount, input_length + test_repetitions, interval_amount))
    results[:, 0:input_length, :] = testing_input
    shape = (test_data_amount, input_length * width_sym)
    for i in range(0, test_repetitions):
        for j in range(0, interval_amount):
            input = numpy.zeros(shape)
            for k in range(0, input_length):
                for l in range(0, width_sym):
                    input[:, k * width_sym + l] = results[:, i + k, (j + l - input_width) % interval_amount]
            results[:, input_length + i, j] = (model(input).numpy())[:, 0]
    return results


def print_error(testing_output, results):
    total_error = results[:, -1, :] - testing_output[:, -1, :]
    error_norm = numpy.linalg.norm(total_error, 2)
    error = error_norm / total_error.size
    print('the approximation error is %f' % error)


def additional_tests(model):
    exp_func = "exp(sin(2*pi*x)+cos(2*pi*x))/(sin(4*pi*x) + 1.5)"
    test_function(exp_func, model)


def plot_results(testing_input, testing_output, results):
    x_values = (numpy.array(range(0, interval_amount)) + 0.5) / interval_amount
    for i in range(4, test_data_amount):
        for j in range(0, input_length):
            plt.figure()
            plt.plot(x_values, testing_input[i, j, :])
            plt.legend(["input"])
            plt.title("Test number " + str(i) + ", t = " + str(j / 10))
        for k in range(0, test_repetitions):
            plt.figure()
            plt.plot(x_values, testing_output[i, k, :])
            plt.plot(x_values, results[i, input_length + k, :])
            plt.legend(["exact solution", "approximate solution"])
            plt.title("Test number " + str(i) + ", t = " + str((input_length + k) / 10))
        plt.show()


if __name__ == '__main__':
    training_input, training_output, testing_input, testing_output = load_data()
    model = build_model(training_input, training_output)
    results = test_model(testing_input, model)
    print_error(testing_output, results)
    additional_tests(model)
    plot_results(testing_input, testing_output, results)
