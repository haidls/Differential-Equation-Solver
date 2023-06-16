import numpy
import tensorflow._api.v2.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt


# Create and train a tensorflow model of a neural network
from scipy.integrate import solve_ivp


def create_train_model(hidden_nodes, num_iters, num_input, Xtrain, ytrain):
    tf.disable_eager_execution()

    # Reset the graph
    tf.reset_default_graph()

    # Placeholders for input and output data
    X = tf.placeholder(shape=(num_input, 5), dtype=tf.float64, name='X')
    y = tf.placeholder(shape=(num_input, 1), dtype=tf.float64, name='y')

    # Variables for two group of weights between the three layers of the network
    W1 = tf.Variable(np.random.rand(5, hidden_nodes), dtype=tf.float64)
    W2 = tf.Variable(np.random.rand(hidden_nodes, 1), dtype=tf.float64)

    B1 = tf.Variable(np.random.rand(hidden_nodes), dtype=tf.float64)
    B2 = tf.Variable(np.random.rand(1), dtype=tf.float64)

    # Create the neural net graph
    A1 = tf.sigmoid(tf.add(tf.matmul(X, W1), B1))
    y_est = tf.sigmoid(tf.add(tf.matmul(A1, W2), B2))

    # Define a loss function
    deltas = tf.square(y_est - y)
    loss = tf.reduce_sum(deltas)

    # Define a train operation to minimize the loss
    optimizer = tf.train.GradientDescentOptimizer(0.005)
    train = optimizer.minimize(loss)

    # Initialize variables and run session
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    loss_plot = []

    # Go through num_iters iterations
    for i in range(num_iters):
        sess.run(train, feed_dict={X: Xtrain, y: ytrain})
        loss_plot.append(sess.run(loss, feed_dict={X: Xtrain.to_numpy(), y: ytrain.to_numpy()}) / num_input)
        weights1 = sess.run(W1)
        weights2 = sess.run(W2)
        bias1 = sess.run(B1)
        bias2 = sess.run(B2)

    print("loss (hidden nodes: %d, iterations: %d): %.2f" % (hidden_nodes, num_iters, loss_plot[-1]))
    sess.close()
    return weights1, weights2, loss_plot


def evaluate(num_hidden_nodes, weights1, weights2, Xtest, ytest, test_data_amount = 50):
    # Evaluate models on the test set
    X = tf.placeholder(shape=(test_data_amount, 5), dtype=tf.float64, name='X')
    y = tf.placeholder(shape=(test_data_amount, 1), dtype=tf.float64, name='y')

    for hidden_nodes in num_hidden_nodes:
        # Forward propagation
        W1 = tf.Variable(weights1[hidden_nodes])
        W2 = tf.Variable(weights2[hidden_nodes])
        A1 = tf.sigmoid(tf.matmul(X, W1))
        y_est = tf.sigmoid(tf.matmul(A1, W2))

        # Calculate the predicted outputs
        init = tf.global_variables_initializer()
        results = []
        start = numpy.array(Xtest["start"])
        for i in range(0, 10):
            with tf.Session() as sess:
                sess.run(init)
                y_est_np = sess.run(y_est, feed_dict={X: Xtest, y: ytest})
                Xtest["start"] = y_est_np
                results.append(y_est_np[0])

        Xtest["start"] = start
        eval_points = numpy.linspace(0, 1, 40)
        F = lambda t, s: Xtest["a"][0] * (s ** 3) / 6 + Xtest["b"][0] * (s ** 2) * 0.5 + Xtest["c"][0] * s + \
                         Xtest["d"][0]
        solution = solve_ivp(F, [0, 1], [start[0]], t_eval=eval_points).y[0, :]
        plt.figure()
        plt.plot(numpy.linspace(0.1, 1, 10), results)
        plt.plot(eval_points, solution)
        plt.show()

        # Calculate the prediction accuracy

        error = [(estimate - target) ** 2
                   for estimate, target in zip(y_est_np, ytest.to_numpy())]
        accuracy = sum(error) / len(error)
        print('Network architecture 4-%d-3, accuracy: %.2f' % (hidden_nodes, accuracy))
