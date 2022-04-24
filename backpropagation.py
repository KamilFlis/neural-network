import numpy as np
import matplotlib.pyplot as plt


def error(d, y):
    return (d - y) ** 2


def delta_error(d, y):
    return 2 * (d - y)


def f(t):
    return 1 / (1 + np.exp(-t))


# tutaj wersja bez wchodzenia jeszcze raz w funkcje bo jako argument dajemy x juz po przejsciu przez funkcje aktywacji,
# czyli f(x * waga)
def delta_f(t):
    return t * (1 - t)
    # return f(t) * (1 - f(t))


if __name__ == '__main__':

    epochs = 20000
    learning_rate = 0.5
    epsilon = 10 ** -12  # ?
    inputs = np.array([
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1]
    ])

    expected_output = np.array([0, 1, 1, 0])

    weight = np.array([
        [0.86, -0.16, 0.28],
        [0.82, -0.51, -0.89],
        [0.04, -0.43, 0.48]
    ])

    for it in range(epochs):
        energy = 0
        for X, d in zip(inputs, expected_output):
            # feed forward
            x2_1 = f(X.dot(weight[0]))
            x2_2 = f(X.dot(weight[1]))
            output_vector_1 = np.array([1, x2_1, x2_2])
            x3_1 = f(output_vector_1.dot(weight[2]))

            print(f"X: {X}, d: {d}, y: {x3_1}")

            err = error(d, x3_1)
            # if err < epsilon:
            #     break

            # back propagate
            d3_1 = delta_f(x3_1) * delta_error(d, x3_1)
            d2_1 = delta_f(x2_1) * weight[2][1] * d3_1
            d2_2 = delta_f(x2_2) * weight[2][2] * d3_1

            delta_w = np.array([
                X * learning_rate * d2_1,
                X * learning_rate * d2_2,
                output_vector_1 * learning_rate * d3_1
            ])

            weight = np.add(weight, delta_w)

    print(weight)
