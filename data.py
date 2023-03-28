import math
import random

import numpy
import pde

delta_time = 0.1
coef_max_value = 1
coef_min_value = -1
coefficient_amount = 3

def get_data(coefficient_value_amount, input_length, input_width, test_repetitions=1, test_data_amount=50):
    training_input = []
    training_output = []
    testing_input = []
    testing_output = []

    coefficient_range = numpy.linspace(coef_min_value, coef_max_value, coefficient_value_amount)
    grid = pde.CartesianGrid([[0, 1]], [15])

    coeff = numpy.zeros(coefficient_amount)
    counter = 0
    for i in range(0, coefficient_value_amount**coefficient_amount):
        div = i
        for j in range(0, coefficient_amount):
            index = div % coefficient_value_amount
            coeff[j] = coefficient_range[index]
            div //= coefficient_value_amount
        solution = solve_pde(coeff, grid, input_length)
        input, output = format_input(input_length, input_width, solution)
        counter = counter + 1
        print("solved equation " + str(counter))
        training_input.extend(input)
        training_output.extend(output)

    random.seed(22)
    test_coeff = numpy.zeros((test_data_amount, coefficient_amount))
    for i in range(0, test_data_amount):
        for j in range(0, coefficient_amount):
            coeff[j] = random.uniform(coef_min_value, coef_max_value)
        test_coeff[i, :] = coeff
        solution = solve_pde(coeff, grid, input_length, test_repetitions-1)
        input = (solution[0:input_length, :])
        output = (solution[input_length:, :])
        counter = counter + 1
        print("solved equation " + str(counter))
        testing_input.append(input)
        testing_output.append(output)


        """solution = solve_ivp(F, [0, delta_time * (input_length + test_repetitions - 0.9)], [coeff[0]],
                             t_eval=t_eval_test)
        if solution.success:
            input, output = format_input(input_length, solution, F, t_eval_test)
            testing_input.append(input)
            testing_output.append(output)
        else:
            failed_counter_testing += 1"""

    return numpy.vstack(training_input), numpy.array(training_output), \
           numpy.stack(testing_input), numpy.stack(testing_output)


def solve_pde(coeff, grid, input_length, elongation=0):
    F = function(coeff)
    expression = function_string(coeff)
    field = pde.ScalarField.from_expression(grid, expression)
    boundary_conditions = {"value_expression": expression}
    equation = pde.PDE({"u": "laplace(u)/5"}, bc=boundary_conditions)
    storage = pde.MemoryStorage()
    t_range = delta_time * (input_length + elongation)
    equation.solve(field, t_range=t_range, dt=1e-2, tracker=storage.tracker(delta_time))
    data = numpy.vstack(storage.data)
    solution = pick_out_values(data, F, input_length+elongation+1)
    return solution


def function(coeff):
    return lambda t, s: coeff[0] * math.sin(2 * math.pi * s) + coeff[1] * s + coeff[2] * math.cos(2 * math.pi * s)


def function_string(coeff):
    return str(coeff[0]) + "*sin(2*pi*x)+" + str(coeff[1]) + "*x+" + str(coeff[2]) + "*cos(2*pi*x)"


def pick_out_values(data, F, length):
    solution = numpy.zeros((length, 11))
    solution[:, 0] = numpy.ones(length) * F(0, 0)
    index = 1
    for i in range(0, 4):
        solution[:, 2 * i + 1] = data[:, index]
        solution[:, 2 * i + 2] = (data[:, index + 1] + data[:, index + 2]) / 2
        index = index + 3
    solution[:, 9] = data[:, index]
    solution[:, 10] = numpy.ones(length) * F(0, 1)
    return solution


def format_input(input_length, input_width, solution):
    input = []
    output = []
    for i in range(input_width, solution.shape[1] - input_width):
        input.append((solution[-1-input_length:-1, i-input_width:i+input_width+1]).flatten())
        output.append(solution[-1, i])

    return input, output