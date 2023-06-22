import random

import numpy
import pde

delta_time = 0.1
coef_max_value = 1
coef_min_value = -1
coefficient_amount = 3
dt = 1e-7


def get_data(coefficient_value_amount, input_length, input_width, right_hand_side, test_repetitions=1,
             test_data_amount=50, interval_amount=10):
    """
    Generates and returns data for the model.
    :param coefficient_value_amount: the amount of different values per coefficient
    :param input_length: the amount of time steps the input consists of
    :param input_width: the amount of spacial steps the input consists of
    :param right_hand_side: right hand side of the PDE
    :param test_repetitions: the amount of output time steps
    :param test_data_amount: the amount of testing data
    :param interval_amount: the amount of spacial intervals the numerical approximation uses
    :return: the data necessary for building a model
    """
    coefficient_range = numpy.linspace(coef_min_value, coef_max_value, coefficient_value_amount)
    grid = pde.CartesianGrid([[0, 1]], [interval_amount], periodic=True)

    training_input, training_output = create_training_data(coefficient_range, coefficient_value_amount, grid,
                                                           input_length, input_width, interval_amount, right_hand_side)

    testing_input, testing_output = create_testing_data(grid, input_length, right_hand_side, test_data_amount,
                                                        test_repetitions)

    return numpy.vstack(training_input), numpy.array(training_output), \
           numpy.stack(testing_input), numpy.stack(testing_output)


def create_testing_data(grid, input_length, right_hand_side, test_data_amount, test_repetitions):
    """
    Creates data for testing the model
    :param grid: grid used for the numerical solving of the PDE
    :param input_length: the amount of time steps the input consists of
    :param right_hand_side: right hand side of the PDE
    :param test_data_amount: the amount of testing data
    :param test_repetitions: the amount of output time steps
    :return: data for testing a model
    """
    testing_input = []
    testing_output = []
    random.seed(22)
    coeff = numpy.zeros(coefficient_amount)
    test_coeff = numpy.zeros((test_data_amount, coefficient_amount))
    counter = 0
    for i in range(0, test_data_amount):
        for j in range(0, coefficient_amount):
            coeff[j] = random.uniform(coef_min_value, coef_max_value)
        test_coeff[i, :] = coeff
        solution = solve_pde(coeff, grid, input_length + test_repetitions - 1, right_hand_side)
        input = (solution[0:input_length, :])
        output = (solution[input_length:, :])
        counter = counter + 1
        print("solved equation " + str(counter))
        testing_input.append(input)
        testing_output.append(output)
    return testing_input, testing_output


def create_training_data(coefficient_range, coefficient_value_amount, grid, input_length, input_width, interval_amount,
                         right_hand_side):
    """
    Creates data for training the model
    :param coefficient_range: possible coefficient values
    :param coefficient_value_amount: the amount of different values per coefficient
    :param grid: grid used for the numerical solving of the PDE
    :param input_length: the amount of time steps the input consists of
    :param input_width: the amount of spacial steps the input consists of
    :param interval_amount: the amount of spacial intervals the numerical approximation uses
    :param right_hand_side: right hand side of the PDE
    :return: data for creating a model
    """
    training_input = []
    training_output = []
    coeff = numpy.zeros(coefficient_amount)
    counter = 0
    for i in range(0, coefficient_value_amount ** coefficient_amount):
        div = i
        for j in range(0, coefficient_amount):
            index = div % coefficient_value_amount
            coeff[j] = coefficient_range[index]
            div //= coefficient_value_amount
        solution = solve_pde(coeff, grid, input_length, right_hand_side)
        input, output = format_input(input_length, input_width, solution, interval_amount)
        counter = counter + 1
        print("solved equation " + str(counter))
        training_input.extend(input)
        training_output.extend(output)
    return training_input, training_output


def solve_pde(coeff, grid, length, right_hand_side):
    """
    Solves a PDE numerically.
    :param coeff: coefficients of the right hand side
    :param grid: grid used for the numerical solving of the PDE
    :param length: the amount of time steps
    :param right_hand_side: right hand side of the PDE
    :return: numerical solution of the PDE
    """
    expression = function_string(coeff)
    field = pde.ScalarField.from_expression(grid, expression)
    equation = pde.PDE({"u": right_hand_side}, bc='periodic')
    storage = pde.MemoryStorage()
    t_range = delta_time * length
    equation.solve(field, t_range=t_range, dt=dt, tracker=storage.tracker(delta_time))
    data = numpy.vstack(storage.data)
    return data


def function_string(coeff):
    """
    generates a string representation of the function.
    :param coeff: coefficients of the right hand side
    :return: string representation
    """
    return str(coeff[0]) + "*sin(2*pi*x)**3+" + str(coeff[1]) + "*cos(2*pi*x)**3+" + str(coeff[2]) + "*cos(4*pi*x)**3"


def format_input(input_length, input_width, solution, interval_amount):
    """
       Extracts input and output data for training a model from the numerical solution .
       :param input_length: the amount of time steps the input consists of
       :param input_width: the amount of spacial steps the input consists of
       :param solution: numerical solution of the PDE
       :param interval_amount: the amount of spacial intervals the numerical approximation uses
       :return: input and output data
       """
    input = []
    output = []
    width_sym = (2 * input_width + 1)
    current_input = numpy.zeros(input_length * width_sym)
    for i in range(0, interval_amount):
        for j in range(0, input_length):
            for k in range(0, width_sym):
                current_input[j*width_sym+k] = solution[j, (i + k - input_width) % interval_amount]
        input.append(numpy.copy(current_input))
        output.append(solution[-1, i])
    return input, output