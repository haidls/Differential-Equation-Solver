import math
import random

import numpy
from scipy.integrate import solve_ivp


delta_time = 0.1
dimensions = 3
coef_max_value = 3
coef_min_value = 1
start_max_value = 3
start_min_value = -3
coefficient_amount = 6


def get_data(coefficient_value_amount, input_length, test_repetitions=1, test_data_amount=50):
    training_input = []
    training_output = []
    testing_input = []
    testing_output = []

    coefficient_range = numpy.linspace(coef_min_value, coef_max_value, coefficient_value_amount)
    start_range = numpy.linspace(start_min_value, start_max_value, coefficient_value_amount)
    t_eval_test = numpy.linspace(0, (input_length + test_repetitions - 1) * delta_time, input_length + test_repetitions)
    t_eval_training = t_eval_test[0:input_length+1]
    failed_counter_training = 0

    coeff = numpy.zeros(coefficient_amount)
    for i in range(0, coefficient_amount**coefficient_value_amount):
        div = i
        for j in range(0, dimensions):
            index = div % coefficient_value_amount
            coeff[j] = start_range[index]
            div //= coefficient_value_amount
        for j in range(dimensions, coefficient_amount):
            index = div % coefficient_value_amount
            coeff[j] = coefficient_range[index]
            div //= coefficient_value_amount

        F = function(coeff[dimensions:])
        solution = solve_ivp(F, [0, delta_time * (input_length + 0.1)], coeff[0:dimensions], t_eval=t_eval_training)
        if solution.success:
            input, output = format_input(input_length, solution, F, t_eval_training)
            training_input.append(input)
            training_output.append(output)
        else:
            failed_counter_training += 1
    print('The training data generation failed %d times' % failed_counter_training)

    failed_counter_testing = 0
    random.seed(22)
    test_coeff = numpy.zeros((test_data_amount, coefficient_amount-dimensions))
    for i in range(0, test_data_amount):
        for j in range(0, coefficient_amount):
            coeff[j] = random.uniform(coef_min_value, coef_max_value)
        F = function(coeff[dimensions:])
        test_coeff[i, :] = coeff[dimensions:]

        solution = solve_ivp(F, [0, delta_time * (input_length + test_repetitions - 0.9)], coeff[0:dimensions],
                             t_eval=t_eval_test)
        if solution.success:
            input, output = format_input(input_length, solution, F, t_eval_test)
            testing_input.append(input)
            testing_output.append(output)
        else:
            failed_counter_testing += 1
    print('The testing data generation failed %d times' % failed_counter_testing)

    return numpy.vstack(training_input), numpy.array(training_output), \
           numpy.vstack(testing_input), numpy.array(testing_output), t_eval_test, test_coeff


def function(coeff):
    return lambda t, s: numpy.array([coeff[0] * (s[1] - s[0]),
                                     s[0] * (coeff[1] - s[2]) - s[1],
                                     s[0] * s[1] - coeff[2] * s[2]])


def format_input(input_length, solution, F, t_eval):
    input = numpy.zeros(2 * input_length * dimensions)
    input[0:input_length*dimensions] = (solution.y[:, 0:input_length]).flatten('F')
    for i in range(0, input_length):
        input[dimensions*(input_length+i):dimensions*(input_length+i+1)] = \
            F(t_eval[i], input[dimensions*i:dimensions*(i+1)])
    return input, (solution.y[:, input_length:]).flatten('F')
