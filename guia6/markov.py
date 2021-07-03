import math
import numpy as np


def transitionPower(p, n):
    """p^n

    Args:
        p (np.array): Transition matrix
        n (int): Power
    """

    if not isinstance(p, np.ndarray):
        raise RuntimeError("Not a valid matrix :(")
    if type(n) is not int:
        raise RuntimeError("n must be an integer.")

    if (len(p.shape) != 2 or p.shape[0] != p.shape[1]):
        raise RuntimeError("This is not a square matrix!")

    return np.linalg.matrix_power(p, n)


def stationaryDist(p):
    """Stationary distribution using eigenvectors

    Args:
        p (np.array): Transition matrix
    """

    if not isinstance(p, np.ndarray):
        raise RuntimeError("Not a valid matrix :(")

    if (len(p.shape) != 2 or p.shape[0] != p.shape[1]):
        raise RuntimeError("This is not a square matrix!")

    statDistArr = np.zeros((1, p.shape[0]))

    w, v = np.linalg.eig(np.transpose(p))

    i = 0
    eigval = 0
    while i < len(w):
        eigval = w[i]
        if math.isclose(eigval, 1):
            statDistArr = np.divide(
                np.transpose(v[:, i]),
                np.sum(v[:, i])
                )
            break

        i += 1
    else:
        print("No eigenvalue equal to 1 was found.")

    return statDistArr


if __name__ == "__main__":
    print()
    # Transition matrix
    markovP = np.array(
        [
            [0.3, 0.4, 0.3],
            [1.0, 0.0, 0.0],
            [0.0, 0.3, 0.7]
        ]
    )

    matrixPower = transitionPower(markovP, 2)
    statDistributionArr = stationaryDist(markovP)

    # Pretty print
    np.set_printoptions(formatter={'float': '{: 0.5f}'.format})

    print("Stationary distribution array: ", statDistributionArr)
    print()
    print("Matrix ^ 2:")
    for row in matrixPower:
        print('|', end='  ')
        for col in row:
            print('{:0.5f}'.format(round(col, 5)), end='  ')
        print('|')

    print()
