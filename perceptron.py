import matplotlib.pyplot as plt
import numpy as np


def step_function(x):
    return 1 if x > 0 else 0


def perceptron(weight, inputs, expected_outputs):
    actual_outputs = np.array([-1, -1, -1, -1])
    xx = np.arange(0, 1.5)
    for i in range(0, 1000):
        for index, (x, d) in enumerate(zip(inputs, expected_outputs)):
            actual_output = step_function(np.dot(weight, x))
            if actual_output != d:
                weight = np.add(weight, (d - actual_output) * x)
                if weight[2] != 0:
                    yy = - weight[0] / weight[2] - xx * weight[1] / weight[2]
                    plt.scatter([0, 0, 1, 1], [0, 1, 0, 1])
                    plt.plot(xx, yy)
                    plt.show()
            actual_outputs.put(index, actual_output)

        # print(actual_outputs, weight)
        if np.array_equal(actual_outputs, expected_outputs):
            print("Perceptron finished with weight: ", weight)
            break


def bupa(weight, inputs, expected_outputs):
    actual_outputs = np.array([-1, -1, -1, -1])
    xx = np.arange(0, 1.5)
    for i in range(0, 1000):
        z = np.array([0, 0, 0])
        for index, (x, d) in enumerate(zip(inputs, expected_outputs)):
            actual_output = step_function(np.dot(weight, x))
            actual_outputs.put(index, actual_output)

        for (x, d, y) in zip(inputs, expected_outputs, actual_outputs):
            if d != y:
                z = np.add((d - y) * x, z)
        weight = np.add(weight, z)
        if weight[2] != 0:
            yy = - weight[0] / weight[2] - xx * weight[1] / weight[2]
            plt.scatter([0, 0, 1, 1], [0, 1, 0, 1])
            # plt.scatter([0, 0, 1, 1], [1, 0.6, 0.6, 1])
            plt.plot(xx, yy)
            plt.show()

        # print(actual_outputs, weight)
        if np.array_equal(actual_outputs, expected_outputs):
            print("BUPA finished with weight: ", weight)
            break


def kernel(x, y):
    return np.exp(-((x-y)**2)/2)


def rbf(vector):
    sigma = 1
    return np.exp(- ((np.linalg.norm(vector) ** 2) / 2 * sigma ** 2))


if __name__ == '__main__':

    # AND
    weight = np.array([0.5, 0, 1])
    inputs = np.array([
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1]
    ])
    expected_outputs = np.array([0, 0, 0, 1])

    perceptron(weight, inputs, expected_outputs)
    bupa(weight, inputs, expected_outputs)

    # XOR
    weight = np.array([0.5, 0, 1])
    inputs = np.array([
        [0, 0, kernel(0, 0)],
        [0, 1, kernel(0, 1)],
        [1, 0, kernel(1, 0)],
        [1, 1, kernel(1, 1)]
    ])
    expected_outputs = np.array([0, 1, 1, 0])

    bupa(weight, inputs, expected_outputs)
    # perceptron(weight, inputs, expected_outputs)
