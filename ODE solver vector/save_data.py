import numpy
import data


def generate_data(coeff_value_amount, input_length, test_data_amount, test_repetitions, file_path):
    training_input, training_output, testing_input, testing_output, t_eval, test_coeff = \
        data.get_data(coefficient_value_amount=coeff_value_amount, input_length=input_length, test_repetitions=test_repetitions,
                 test_data_amount=test_data_amount)

    with open(file_path, 'wb') as f:
        numpy.save(f, training_input)
        numpy.save(f, training_output)
        numpy.save(f, testing_input)
        numpy.save(f, testing_output)
        numpy.save(f, t_eval)
        numpy.save(f, test_coeff)