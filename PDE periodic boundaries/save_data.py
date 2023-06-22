import numpy
import data


def generate_data(coefficient_value_amount, input_length, input_width, test_data_amount, test_repetitions,
                  interval_amount, right_hand_side, file_path):
    """
    Generates and saves data for creating a model.
    :param coefficient_value_amount: the amount of different values per coefficient
    :param input_length: the amount of time steps the input consists of
    :param input_width: the amount of spacial steps the input consists of
    :param right_hand_side: right hand side of the PDE
    :param test_repetitions: the amount of output time steps
    :param test_data_amount: the amount of testing data
    :param interval_amount: the amount of spacial intervals the numerical approximation uses
    :param file_path: path of the file in which the data will be saved
    """
    training_input, training_output, testing_input, testing_output = \
        data.get_data(coefficient_value_amount=coefficient_value_amount, input_length=input_length,
                      input_width=input_width, test_repetitions=test_repetitions, test_data_amount=test_data_amount,
                      interval_amount=interval_amount, right_hand_side=right_hand_side)

    with open(file_path, 'wb') as f:
        numpy.save(f, training_input)
        numpy.save(f, training_output)
        numpy.save(f, testing_input)
        numpy.save(f, testing_output)
