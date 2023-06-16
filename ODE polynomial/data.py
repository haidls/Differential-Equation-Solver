import random
import numpy
from scipy.integrate import solve_ivp


def get_data(coefficient_value_amount, starting_value_amount, test_data_amount = 50):
    eval_points = [1]
    coefficient_bound = 1
    starting_value_bound = 2
    coefficient_values = numpy.linspace(-coefficient_bound, coefficient_bound, coefficient_value_amount)
    starting_values = numpy.linspace(-starting_value_bound, starting_value_bound, starting_value_amount)

    training_data = []
    training_coefficients = []
    failed = []

    for start in starting_values:
        for a in coefficient_values:
            for b in coefficient_values:
                for c in coefficient_values:
                    for d in coefficient_values:
                        F = lambda t, s: a * (s ** 3) / 6 + b * (s ** 2) * 0.5 + c * s + d
                        solution = solve_ivp(F, [0, 1], [start], t_eval=eval_points)
                        if solution.success:
                            training_data.append(solution.y[0, 0])
                            training_coefficients.append(numpy.array([start, a, b, c, d]))
                        else:
                            failed.append([start, a, b, c, d])

    test_data = []
    test_coefficients = []

    random.seed(22)
    for i in range(0, test_data_amount):
        curr_coefficients = numpy.zeros(5)
        curr_coefficients[0] = random.uniform(-1, 1)
        for j in range(1, 5):
            curr_coefficients[j] = random.uniform(-coefficient_bound, coefficient_bound)

        F = lambda t, s: curr_coefficients[1] * (s ** 3) / 6 + curr_coefficients[2] * (s ** 2) * 0.5 \
                         + curr_coefficients[3] * s + curr_coefficients[4]

        solution = solve_ivp(F, [0, 1], [curr_coefficients[0]], t_eval=numpy.array(range(0, 10)) / 10)
        if solution.success:
            test_data.append(solution.y[0])
            test_coefficients.append(curr_coefficients)
        else:
            failed.append(curr_coefficients)

    print("failed amount = " + str(len(failed)))

    return numpy.array(training_data), numpy.array(test_data), numpy.array(training_coefficients), \
           numpy.array(test_coefficients)


