"""
Nothing
"""

import numpy as np
import matplotlib.pyplot as plt


def theta(x, scalar=0.0):
    noise = np.random.normal(size=x.shape[0]) * scalar
    return x ** 2 + noise


def constant(xi, xj, epsilon=1.0, value=1.0):
    return value


def quadratic_exponential(xi, xj, epsilon=1.0, alpha=1.0):
    K = alpha ** 2 * np.exp(-np.linalg.norm(xi - xj) ** 2 / (2 * epsilon ** 2))
    return K


def rational_quadratic_kernel(xi, xj, epsilon=1.0, weight=1.0):
    K = (1 + np.linalg.norm(xi - xj) ** 2 / (2 * weight * epsilon) ** 2) ** -weight
    return K


def exponential_sine_squared(xi, xj, epsilon=1.0, period=1.0):
    K = np.exp(-2 / epsilon ** 2 * (np.sin(np.pi * np.linalg.norm(xi - xj) ** 2 / period)) ** 2)
    return K


def get_inner_variance(x, epsilon=1.0, sigma=0.5, function=quadratic_exponential):
    if len(x.shape) > 1:
        raise ValueError("Input array of points x should be 1-dimensional. Got higher dimension instead...")

    K = np.zeros([x.shape[0], x.shape[0]])

    for i in range(x.shape[0]):
        for j in range(x.shape[0]):
            K[i, j] = function(x[i], x[j], epsilon)

    return K + sigma * np.eye(K.shape[0])


def get_covariance_vector(x, x_new, epsilon=1.0, sigma=0.5, function=quadratic_exponential):
    k = np.zeros(x.shape[0])

    for i in range(x.shape[0]):
        k[i] = function(x[i], x_new, epsilon)

    k[-1] = k[-1] + sigma
    return k


def gaussian_process_predict_mean(x, y, x_new, K, epsilon=1.0, sigma=0.0, function=quadratic_exponential):
    k = get_covariance_vector(x, x_new, epsilon=epsilon, sigma=sigma, function=function)
    return np.dot(np.dot(k, np.linalg.inv(K)), y)


def gaussian_process_predict_std(x, x_new, K, epsilon=1.0, sigma=0.0, function=quadratic_exponential):
    k = get_covariance_vector(x, x_new, epsilon=epsilon, sigma=sigma, function=function)
    return k[-1] - np.dot(np.dot(k, np.linalg.inv(K)), k)


def gaussian_predict(x, y, x_new, K=None, epsilon=1.0, sigma=0.0, function=quadratic_exponential):
    predictions = list()
    error = list()

    # If no kernel is given construct one from a given function
    if not type(K) == np.ndarray:
        K = get_inner_variance(x, epsilon=epsilon, sigma=sigma, function=function)

    for i in x_new:
        point = np.array([i])

        # Estimate the mu and sigma parameters
        prediction = gaussian_process_predict_mean(x, y, point, K, epsilon=epsilon, sigma=sigma, function=function)
        std = gaussian_process_predict_std(x, point, K, epsilon=epsilon, sigma=sigma, function=function)

        predictions.append(prediction)
        error.append(std)

    return np.array(predictions), np.array(error)


def main():
    # Randomly sample a set of points
    x = np.linspace(-5, 5, 1000)
    rng = np.random.RandomState(1)
    x_train = rng.choice(np.linspace(-3, 3, 1000), size=6, replace=False)
    y_train = theta(x_train)

    # Predict values with a given kernel, we assume there is no noise in the samples
    epsilon = 16.9
    y, error = gaussian_predict(x_train, y_train, x, epsilon=epsilon, sigma=0.0, function=quadratic_exponential)
    print(f'Mean error: {np.mean(error)}')

    # Plot the results
    plt.figure(figsize=(10, 7))
    plt.plot(x, theta(x), linewidth=8, c='b', label='$f(x) = x^2$')
    plt.plot(x, y, linewidth=4, label="Predictions", linestyle='dashed', c='orange')
    plt.scatter(x_train, y_train, c='r', marker='o', zorder=10, label="Samples", s=100)
    plt.legend(fontsize=14)
    plt.ylabel('y')
    plt.xlabel('x')
    plt.title('Extrapolation of $f(x) = x^2$, $\epsilon = 16.9$', fontsize=20)
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()
