import numpy
import data

right_hand_side = "(6 * u * d_dx(u) - laplace(d_dx(u)))/10000"
input_length = 3
input_width = 1
test_data_amount = 30
test_repetitions = 7
interval_amount = 100

def generate_data(input_length, input_width, test_data_amount, test_repetitions, interval_amount, right_hand_side):
    training_input, training_output, testing_input, testing_output = \
        data.get_data(coefficient_value_amount=5, input_length=input_length, input_width=input_width,
                      test_repetitions=test_repetitions, test_data_amount=test_data_amount,
                      interval_amount=interval_amount, right_hand_side=right_hand_side)

    file_path = 'data_sets/data_set_lwdri_%d_%d_%d_%d_%d.npy' % (input_length, input_width, test_data_amount,
                                                             test_repetitions, interval_amount)
    with open(file_path, 'wb') as f:
        numpy.save(f, training_input)
        numpy.save(f, training_output)
        numpy.save(f, testing_input)
        numpy.save(f, testing_output)


if __name__ == '__main__':
    generate_data(input_length, input_width, test_data_amount, test_repetitions, interval_amount, right_hand_side)