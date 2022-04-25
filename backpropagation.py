import numpy as np
import matplotlib.pyplot as plt


def error(d, y):
    return (d - y) ** 2


def delta_error(d, y):
    return 2 * (d - y)


def f(t):
    return 1 / (1 + np.exp(-t))


def delta_f(t):
    return t * (1 - t)


def plot_energy_histogram(energy_history, title_prefix):
    for input_vector in energy_history:
        plt.plot(energy_history[input_vector], marker='o')
        plt.title(f"{title_prefix} - X{input_vector}")
        plt.ylabel('Wartość Energii')
        plt.xlabel('Liczba przetworzonych wektorów wejściowych')
        plt.show()


def process_using_partial_energy_method():

    weights = np.copy(INITIAL_WEIGHTS)
    print("\n===================== PROCESSING USING PARTIAL ENERGY =====================\n")

    energy_history = dict()
    for X in INPUTS:
        energy_history[tuple(X)] = []

    for i in range(ITERATION_LIMIT):
        print(f"Iteration number : {i}")
        network_learning_finished_correctly = True
        for X, d in zip(INPUTS, EXPECTED_OUTPUTS):
            # feed forward
            x2_1 = f(X.dot(weights[0]))
            x2_2 = f(X.dot(weights[1]))
            output_vector_1 = np.array([1, x2_1, x2_2])
            x3_1 = f(output_vector_1.dot(weights[2]))

            energy = error(d, x3_1)
            energy_history[tuple(X)].append(energy)
            if energy > TOLERATED_ERROR:
                network_learning_finished_correctly = False
                print(f"X: {X}, d: {d}, y: {x3_1} - ERROR NOT TOLERATED")
            else:
                print(f"X: {X}, d: {d}, y: {x3_1} - ERROR TOLERATED")
                continue

            # back propagate
            d3_1 = delta_f(x3_1) * delta_error(d, x3_1)
            d2_1 = delta_f(x2_1) * weights[2][1] * d3_1
            d2_2 = delta_f(x2_2) * weights[2][2] * d3_1

            delta_w = np.array([
                X * LEARNING_RATE * d2_1,
                X * LEARNING_RATE * d2_2,
                output_vector_1 * LEARNING_RATE * d3_1
            ])

            weights = np.add(weights, delta_w)

        if network_learning_finished_correctly:
            print(f"Network has learned with success!")
            break

    print("Final Weights : ")
    print(weights)

    plot_energy_histogram(energy_history, "Metoda energii cząstkowej")


def process_using_cumulative_energy_method():
    print("\n===================== PROCESSING USING CUMULATIVE ENERGY =====================\n")
    weights = np.copy(INITIAL_WEIGHTS)

    energy_history = dict()
    for X in INPUTS:
        energy_history[tuple(X)] = []

    for i in range(ITERATION_LIMIT):
        print(f"Iteration number : {i}")
        network_learning_finished_correctly = True
        cumulative_delta = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]
        ])
        for X, d in zip(INPUTS, EXPECTED_OUTPUTS):
            # feed forward
            x2_1 = f(X.dot(weights[0]))
            x2_2 = f(X.dot(weights[1]))
            output_vector_1 = np.array([1, x2_1, x2_2])
            x3_1 = f(output_vector_1.dot(weights[2]))

            energy = error(d, x3_1)
            energy_history[tuple(X)].append(energy)
            if energy > TOLERATED_ERROR:
                network_learning_finished_correctly = False
                print(f"X: {X}, d: {d}, y: {x3_1} - ERROR NOT TOLERATED")
            else:
                print(f"X: {X}, d: {d}, y: {x3_1} - ERROR TOLERATED")
                continue

            # back propagate
            d3_1 = delta_f(x3_1) * delta_error(d, x3_1)
            d2_1 = delta_f(x2_1) * weights[2][1] * d3_1
            d2_2 = delta_f(x2_2) * weights[2][2] * d3_1

            delta_w = np.array([
                X * LEARNING_RATE * d2_1,
                X * LEARNING_RATE * d2_2,
                output_vector_1 * LEARNING_RATE * d3_1
            ])

            cumulative_delta = np.add(cumulative_delta, delta_w)

        if network_learning_finished_correctly:
            print(f"Network has learned with success!")
            break
        else:
            weights = np.add(weights, cumulative_delta)

    print("Final Weights : ")
    print(weights)

    plot_energy_histogram(energy_history, "Metoda energii całkowitej")


def main():
    input("Press any key to start network processing using partial energy method : ")
    process_using_partial_energy_method()
    input("Press any key to start network processing using cumulative energy method : ")
    process_using_cumulative_energy_method()


if __name__ == '__main__':
    ITERATION_LIMIT = 30000
    LEARNING_RATE = 0.5
    TOLERATED_ERROR = 10 ** -4 # Epsilon
    INPUTS = np.array([
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1]
    ])
    EXPECTED_OUTPUTS = np.array([0, 1, 1, 0])
    INITIAL_WEIGHTS = np.array([
        [0.86, -0.16, 0.28],
        [0.82, -0.51, -0.89],
        [0.04, -0.43, 0.48]
    ])

    main()
