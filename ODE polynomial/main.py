import numpy
import pandas
from data import get_data
import stackabuse_res
import matplotlib.pyplot as plt

coefficient_value_amount = 5
starting_value_amount = 6
test_data_amount = 1
training_data, test_data, training_coefficients, test_coefficients = get_data(coefficient_value_amount,
                                                                              starting_value_amount, test_data_amount)

t_training_data_frame = pandas.DataFrame({
    "start": training_coefficients[:, 0],
    "a": training_coefficients[:, 1],
    "b": training_coefficients[:, 2],
    "c": training_coefficients[:, 3],
    "d": training_coefficients[:, 4]
})

y_training_data_frame = pandas.DataFrame({
    "y": training_data
})

t_testing_data_frame = pandas.DataFrame({
    "start": test_coefficients[:, 0],
    "a": test_coefficients[:, 1],
    "b": test_coefficients[:, 2],
    "c": test_coefficients[:, 3],
    "d": test_coefficients[:, 4]
})

y_testing_data_frame = pandas.DataFrame({
    "y": test_data
})

# Run the training for 3 different network architectures: (4-5-3) (4-10-3) (4-20-3)

# Plot the loss function over iterations
num_hidden_nodes = [5, 10, 20]
loss_plot = {5: [], 10: [], 20: []}
weights1 = {5: None, 10: None, 20: None}
weights2 = {5: None, 10: None, 20: None}
bias1 = {5: None, 10: None, 20: None}
bias2 = {5: None, 10: None, 20: None}
num_iters = 300

plt.figure()
for hidden_nodes in num_hidden_nodes:
    weights1[hidden_nodes], weights2[hidden_nodes], bias1[hidden_nodes], bias2[hidden_nodes], loss_plot[hidden_nodes] \
        = stackabuse_res.create_train_model(hidden_nodes, num_iters, coefficient_value_amount ** 4 *
                                            starting_value_amount, t_training_data_frame, y_training_data_frame)
    print("amount of hidden nodes: " + str(hidden_nodes))
    print(weights1[hidden_nodes])
    print(weights2[hidden_nodes])
    print(bias1[hidden_nodes])
    print(str(bias2[hidden_nodes]) + '\n')
    plt.plot(range(num_iters), loss_plot[hidden_nodes], label="nn: 4-%d-1" % hidden_nodes)

plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend(fontsize=12)
plt.show()

stackabuse_res.evaluate(num_hidden_nodes, weights1, weights2, bias1, bias2, t_testing_data_frame, y_testing_data_frame,
                        test_data_amount)
