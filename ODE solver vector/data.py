"""
This file contains a function for creating data for a neural network
author: Linus Haid
version: 22.06.2023
"""

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
    """
    Generates and returns data for the model.
    :param coefficient_value_amount: the amount of different values per coefficient
    :param input_length: the amount of time steps the input consists of
    :param test_repetitions: the amount of output time steps
    :param test_data_amount: the amount of testing data
    :return: the data necessary for building a model
    """
    coefficient_range, start_range, t_eval_test, t_eval_training = data_setup(coefficient_value_amount, input_length,
                                                                              test_repetitions)
    training_input, training_output = create_training_data(coefficient_range, coefficient_value_amount,
                                                                  input_length, start_range, t_eval_training)

    test_coeff, testing_input, testing_output = create_testing_data(input_length, t_eval_test, test_data_amount,
                                                                    test_repetitions)

    return numpy.stack(training_input, axis=0), numpy.array(training_output), \
           numpy.stack(testing_input, axis=0), numpy.stack(testing_output, axis=0), t_eval_test, test_coeff


def data_setup(coefficient_value_amount, input_length, test_repetitions):
    """
    Prepares data for the numerical solving of the ODEs.
    :param coefficient_value_amount: the amount of different values per coefficient
    :param input_length: the amount of time steps the input consists of
    :param test_repetitions: the amount of output time steps
    :return: data for the numerical solving of the ODE
    """
    coefficient_range = numpy.linspace(coef_min_value, coef_max_value, coefficient_value_amount)
    start_range = numpy.linspace(start_min_value, start_max_value, coefficient_value_amount)
    t_eval_test = numpy.linspace(0, (input_length + test_repetitions - 1) * delta_time, input_length + test_repetitions)
    t_eval_training = t_eval_test[0:input_length + 1]
    return coefficient_range, start_range, t_eval_test, t_eval_training


def create_testing_data(input_length, t_eval_test, test_data_amount, test_repetitions):
    """
    Creates data for testing the model
    :param input_length: the amount of time steps the input consists of
    :param t_eval_test: evaluation points for the approximation
    :param test_data_amount: the amount of testing data
    :param test_repetitions: the amount of output time steps
    :return: data for testing a model
    """
    testing_input = []
    testing_output = []
    failed_counter_testing = 0
    coeff = numpy.zeros(coefficient_amount)
    random.seed(22)
    test_coeff = numpy.zeros((test_data_amount, coefficient_amount - dimensions))
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
    return test_coeff, testing_input, testing_output


def create_training_data(coefficient_range, coefficient_value_amount, input_length, start_range, t_eval_training):
    """
    Creates data for training the model
    :param coefficient_range: possible coefficient values
    :param coefficient_value_amount: the amount of different values per coefficient
    :param input_length: the amount of time steps the input consists of
    :param t_eval_training: evaluation points for the approximation
    :return: data for creating a model
    """
    training_input = []
    training_output = []
    failed_counter_training = 0
    coeff = numpy.zeros(coefficient_amount)
    for i in range(0, coefficient_value_amount ** coefficient_amount):
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
    return training_input, training_output


def function(coeff):
    """
    right hand side of the ODE
    :param coeff: coefficients for the function
    :return: a derivative value
    """
    return lambda t, s: numpy.array([coeff[0] * (s[1] - s[0]),
                                     s[0] * (coeff[1] - s[2]) - s[1],
                                     s[0] * s[1] - coeff[2] * s[2]])


def format_input(input_length, solution, F, t_eval):
    """
    Extracts input and output data for training a model from the numerical solution .
    :param input_length: the amount of time steps the input consists of
    :param solution: numerical solution of the ODE
    :param F: right hand side of the ODE
    :param t_eval: evaluation points for the approximation
    :return: input and output data
    """
    input = numpy.zeros(2 * input_length * dimensions)
    input[0:input_length*dimensions] = (solution.y[:, 0:input_length]).flatten('F')
    for i in range(0, input_length):
        input[dimensions*(input_length+i):dimensions*(input_length+i+1)] = \
            F(t_eval[i], input[dimensions*i:dimensions*(i+1)])
    return input, (solution.y[:, input_length:])
