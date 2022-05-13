# Gaussian Process regression and RBF interpolation

Gaussian process and RBF interpolation implemented in `Python`.

## Gaussian Process regression

Given a random set of samples `x`, and their respective values `y`, which arise from some function `f(x)`, we can construct a kernel and predict for values of `x` not in present the input. When trying to predict values for `x` with gaussian a kernel needs to be constructed with great care, often a mix of different kernel functions gives the best performance.

![gaussian_proc](https://user-images.githubusercontent.com/50104866/168268451-7ada21a5-d947-4062-a1d1-148111e6e625.png)


## RBF interpolation

![interp](https://user-images.githubusercontent.com/50104866/168268460-01549a0a-a511-4134-91bc-c199546efb39.png)
