import matplotlib.pyplot as plt
import numpy as np


def step_function(x):
    return 1 if x > 0 else 0


def plot_decision_boundary(weight, inputs):
    vectors = inputs[:, 1:]
    if len(vectors[0]) == 2:
        xx = np.arange(0, 1.5)
        if weight[2] != 0:
            yy = - weight[0] / weight[2] - xx * weight[1] / weight[2]
            plt.scatter(
                [vectors[0][0], vectors[1][0], vectors[2][0], vectors[3][0]],
                [vectors[0][1], vectors[1][1], vectors[2][1], vectors[3][1]]
            )
            plt.plot(xx, yy)
            plt.show()


def perceptron(weight, inputs, expected_outputs):
    actual_outputs = np.array([-1, -1, -1, -1])
    for i in range(0, 10000):
        for index, (x, d) in enumerate(zip(inputs, expected_outputs)):
            actual_output = step_function(np.dot(weight, x))
            if actual_output != d:
                weight = np.add(weight, (d - actual_output) * x)
                plot_decision_boundary(weight, inputs)

            actual_outputs.put(index, actual_output)

        if np.array_equal(actual_outputs, expected_outputs):
            print("Perceptron finished with weight: ", weight)
            break


def bupa(weight, inputs, expected_outputs):
    actual_outputs = np.array([-1, -1, -1, -1])
    for i in range(0, 50000):
        z = np.zeros(len(weight))
        for index, (x, d) in enumerate(zip(inputs, expected_outputs)):
            actual_output = step_function(np.dot(weight, x))
            actual_outputs.put(index, actual_output)

        for (x, d, y) in zip(inputs, expected_outputs, actual_outputs):
            if d != y:
                z = np.add((d - y) * x, z)
        weight = np.add(weight, z)
        plot_decision_boundary(weight, inputs)

        if np.array_equal(actual_outputs, expected_outputs):
            print("BUPA finished with weight: ", weight)
            break


def kernel(x, y):
    return np.exp(-((x-y)**2)/2)


if __name__ == '__main__':

    # AND
    weight = np.array([0.5, 0, 1])
    # weight = np.array([0, 0, 0])
    inputs = np.array([
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1]
    ])
    expected_outputs = np.array([0, 0, 0, 1])

    plot_decision_boundary(weight, inputs)
    perceptron(weight, inputs, expected_outputs)
    bupa(weight, inputs, expected_outputs)


    # XOR
    weight = np.array([0, 0, 0, 0])
    inputs = np.array([
        [1, 0, 0, kernel(0, 0)],
        [1, 0, 1, kernel(0, 1)],
        [1, 1, 0, kernel(1, 0)],
        [1, 1, 1, kernel(1, 1)]
    ])
    expected_outputs = np.array([0, 1, 1, 0])

    plot_decision_boundary(weight, inputs)
    # bupa(weight, inputs, expected_outputs)
    # perceptron(weight, inputs, expected_outputs)
