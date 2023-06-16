import numpy
from data import get_data
from tensorflow import keras
import tensorflow
import matplotlib.pyplot as plt

coefficient_value_amount = 10
starting_value_amount = 10
test_data_amount = 20
training_data, test_data, training_coefficients, test_coefficients = get_data(coefficient_value_amount,
                                                                              starting_value_amount, test_data_amount)
model = keras.Sequential(
    [
        keras.layers.Dense(20, activation="exponential", name="layer1", input_shape=(5,)),
        keras.layers.Dense(16, activation="exponential", name="layer2"),
        keras.layers.Dense(1, name="layer3"),
    ]
)

model.compile(loss=keras.losses.MeanSquaredError(),
                optimizer=keras.optimizers.Adam(),
                metrics=[keras.metrics.MeanAbsoluteError()])

model.fit(training_coefficients, training_data, batch_size=32, epochs=40)
results = model(test_coefficients).numpy()


t_val = numpy.array(range(0, 10)) / 10

solution_approx = numpy.zeros([len(test_data),2]) #10
solution_approx[:, 0] = test_coefficients[:, 0]

input = test_coefficients
for i in range(1, 2): #10
    #input[:, 0] = solution_approx[:, i-1]
    input_tensor = tensorflow.constant(input)
    solution_approx[:, i] = model(input_tensor).numpy()[0]

offset = solution_approx[:, 1] - test_data[:, -1]
error = numpy.sum(numpy.absolute(offset)) / len(test_data)
print('average error: %f' % error)
print('average exact abs result: %f' % (numpy.sum(numpy.absolute(test_data[:, -1])) / len(test_data)))
print('average approximate abs result: %f' % (numpy.sum(numpy.absolute(results[:, 0])) / len(test_data)))

for i in range(0, len(test_data)):
    plt.figure()
    plt.plot(t_val, test_data[i])
    #plt.plot(t_val, solution_approx[i])
    plt.plot([0,1], solution_approx[i])
    plt.title('start = %f , a = %f , b = %f , c = %f , d = %f' % (test_coefficients[i, 0], test_coefficients[i, 1],
                                                                  test_coefficients[i, 2], test_coefficients[i, 3],
                                                                  test_coefficients[i, 4]))
    plt.legend(["exact solution", "approximate solution"])
plt.show()