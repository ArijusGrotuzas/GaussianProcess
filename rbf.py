"""
RBF interpolation in 2D space.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def theta(points):
    x = points[:, 0]
    y = points[:, 1]
    return (x ** 2 + 3 * y ** 2) * np.e ** (-x ** 2 - y ** 2)


def linear(r, _=None):
    """
    Returns a linear distance, given a distance between two points.

    :param r: Float
        Distance between points x_n and x_m, r = || x_n - x_m ||.
    :param _: None
        Dummy argument.
    :return: Float
        Linear distance, in this case r.
    """

    return r


def cubic(r, _=None):
    """
    Returns cubic distance, given a distance between two points.

    :param r: Float
        Distance between points x_n and x_m, r = || x_n - x_m ||.
    :param _: None
        Dummy argument.
    :return: Float
        Cubic distance, in this case r^3.
    """

    return r ** 3


def thin_plate(r, k=2, _=None):
    """
    Returns a thin plate spline distance, given a distance between two points.

    :param r: Float
        Distance between points x_n and x_m, r = || x_n - x_m ||.
    :param k: Int
        Exponent of r.
    :param _: None
        Dummy argument.
    :return: Float
        Thin plate spline distance, in this case r^(k-1) * log(r^r).
    """

    res = r ** (k - 1) * np.log(r ** r)
    return res


def multiquadric(r, epsilon=1.0):
    """
    Returns a multiquadric distance, given a distance between two points.

    :param r: Float
        Distance between points x_n and x_m, r = || x_n - x_m ||.
    :param epsilon: Float
        Shape parameter.
    :return: Float
        Multiquadric distance, in this case sqrt(1 + (e * r)^2).
    """

    res = np.sqrt(1 + (epsilon * r) ** 2)
    return res


def inverse_quadratic(r, epsilon=1.0):
    """
    Returns an inverse quadratic distance, given a distance between two points.

    :param r: Float
        Distance between points x_n and x_m, r = || x_n - x_m ||.
    :param epsilon: Float
        Shape parameter.
    :return: Float
        Inverse quadratic distance, in this case 1 / (1 + (e * r)^2).
    """

    res = 1 / (1 + (epsilon * r) ** 2)
    return res


def gaussian(r, epsilon=1.0):
    """
    Returns a Gaussian distance, given a distance between two points.

    :param r: Float
        Distance between points x_n and x_m, r = || x_n - x_m ||.
    :param epsilon: Float
        Shape parameter.
    :return: Float
        Gaussian distance, in this case exp(-1 * (e * r)^2).
    """

    res = np.exp(-1 * (epsilon * r) ** 2)
    return res


def inverse_multiquadric(r, epsilon=1.0):
    """
    Returns an inverse multiquadric, given a distance between two points.

    :param r: Float
        Distance between points x_n and x_m, r = || x_n - x_m ||.
    :param epsilon: Float
        Shape parameter.
    :return: Float
        Inverse multiquadric, in this case 1 / sqrt(1 + (e * r)^2).
    """

    res = 1 / np.sqrt(1 + (epsilon * r) ** 2)
    return res


def kernel_matrix(x, epsilon=2.0, function=gaussian):
    """
    Compute a kernel matrix given a set of input points n and a Radial Basis Function (RBF).

    :param x: np.ndarray
        An array of input points, with size of n.
    :param epsilon: Float
        Shape parameter.
    :param function: Function
        An RBF.
    :return: np.ndarray
        A kernel matrix of size n x n
    """

    G = np.zeros((x.shape[0], x.shape[0]))

    for i in range(x.shape[0]):
        diff = np.linalg.norm(x - x[i], axis=1)
        res = function(diff, epsilon)
        G[i] = res

    return G


def rbf_model(G, d):
    """
    Calculate model weights given a kernel matrix and a set of input values for input points x.

    :param G: np.ndarray
        A n x n kernel matrix.
    :param d: np.ndarray
        A list of n input values.
    :return: np.ndarray
        A list of model weights [m_1, m_2, ..., m_n].
    """

    m = np.dot(np.linalg.inv(G), d)
    return m


def rbf_predict(y, x, m, epsilon=2.0, function=gaussian):
    """
    Interpolate m number of points y, given n input points x, and n model weights.

    :param y: np.ndarray
        A list of points that should be interpolated.
    :param x: np.ndarray
        A list of input points, that have an observed value.
    :param m: np.ndarray
        A set of model weights.
    :param epsilon:
        Shape parameter.
    :param function: Function
        An RBF.
    :return: np.ndarray
        A list of interpolated values.
    """

    output = np.zeros(y.shape[0])

    for i in range(y.shape[0]):
        diff = np.linalg.norm(x - y[i], axis=1)
        res = np.sum(m * function(diff, epsilon))
        output[i] = res

    return output


def main():
    # Load a set of randomly sampled points
    data = np.load('points.npz')
    points = data['points']
    values = theta(points)

    # Create a set of points to interpolate for
    shape = np.array([500, 500])
    x = np.linspace(-1.5, 1.5, shape[0])
    y = np.linspace(-1.5, 1.5, shape[1])
    xx, yy = np.meshgrid(x, y)
    interPoints = np.column_stack((xx.flatten(), yy.flatten()))

    # Create kernel matrix and find the model weights
    epsilon = 2.0
    G = kernel_matrix(points, epsilon, inverse_multiquadric)
    m = rbf_model(G, values)

    # Interpolate for points whose values are unknown
    zz = rbf_predict(interPoints, points, m, epsilon, inverse_multiquadric).reshape(shape[0], shape[1], order='F')

    # Display the results
    plt.contourf(xx, yy, zz, cmap='jet', levels=40)
    plt.scatter(points[:, 1], points[:, 0], c=values, cmap='jet', edgecolors='black')
    plt.ylabel('y')
    plt.xlabel('x')
    plt.title('Interpolation of $f(x, y) = (x^2 + 3y^2) \cdot e^{(-x^2 -y^2)}$', fontsize=20)
    plt.show()


if __name__ == '__main__':
    main()
