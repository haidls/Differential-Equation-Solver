import math
import random

import numpy
from scipy.integrate import solve_ivp


def get_data(coefficient_value_amount, input_length, test_repetitions=1, test_data_amount=50):
    delta_time = 0.1
    coef_max_value = 1
    coef_min_value = -1
    start_value_scalar = 3
    coefficient_amount = 5

    training_input = []
    training_output = []
    testing_input = []
    testing_output = []

    coefficient_range = numpy.linspace(coef_min_value, coef_max_value, coefficient_value_amount)
    t_eval_test = numpy.linspace(0, (input_length + test_repetitions - 1) * delta_time, input_length + test_repetitions)
    t_eval_training = t_eval_test[0:input_length+1]
    failed_counter_training = 0

    coeff = numpy.zeros(coefficient_amount)
    for i in range(0, coefficient_amount**coefficient_value_amount):
        div = i
        for j in range(0, coefficient_amount):
            index = div % coefficient_value_amount
            coeff[j] = coefficient_range[index]
            div //= coefficient_value_amount
        coeff[0] = coeff[0] * start_value_scalar
        F = function(coeff[1:5])
        solution = solve_ivp(F, [0, delta_time * (input_length + 0.1)], [coeff[0]], t_eval=t_eval_training)
        if solution.success:
            input, output = format_input(input_length, solution, F, t_eval_training)
            training_input.append(input)
            training_output.append(output)
        else:
            failed_counter_training += 1
    print('The training data generation failed %d times' % failed_counter_training)

    failed_counter_testing = 0
    random.seed(22)
    test_coeff = numpy.zeros((test_data_amount, coefficient_amount-1))
    for i in range(0, test_data_amount):
        for j in range(0, coefficient_amount):
            coeff[j] = random.uniform(coef_min_value, coef_max_value)
        F = function(coeff[1:5])
        test_coeff[i, :] = coeff[1:5]

        solution = solve_ivp(F, [0, delta_time * (input_length + test_repetitions - 0.9)], [coeff[0]],
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
    return lambda t, s: coeff[0] / 8 * s**3 + coeff[1] / 4 * s**2 + coeff[2] / 2 * s + coeff[3]


def format_input(input_length, solution, F, t_eval):
    input = numpy.zeros(2 * input_length)
    input[0:input_length] = solution.y[0, 0:input_length]
    for i in range(0, input_length):
        input[input_length + i] = F(t_eval[i], input[i])
    return input, solution.y[0, input_length:]
