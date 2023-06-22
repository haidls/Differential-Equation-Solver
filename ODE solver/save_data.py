import numpy
import data


def generate_data(coeff_value_amount, input_length, test_data_amount, test_repetitions, file_path):
    """
    Generates and saves data for creating a model.
    :param coeff_value_amount: the amount of different values per coefficient
    :param input_length: the amount of time steps the input consists of
    :param test_data_amount: the amount of testing data
    :param test_repetitions: the amount of output time steps
    :return: the data necessary for building a model
    :param file_path: path of the file in which the data will be saved
    """
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