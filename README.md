# Gaussian Process regression and RBF interpolation

Gaussian process and RBF interpolation implemented in `Python`. Based on C. M. Bishop's book [Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf) (2006).

## Gaussian Process regression

Given a random set of samples `x`, and their respective values `y`, which arise from some function `f(x)`, we can construct a kernel and predict for values of `x` not present in the input. When trying to predict values for `x` with gaussian process the kernel needs to be constructed with great care, often a mix of different kernel functions gives the best performance.

![gaussian_proc](https://user-images.githubusercontent.com/50104866/168268451-7ada21a5-d947-4062-a1d1-148111e6e625.png)


## RBF interpolation

![interp](https://user-images.githubusercontent.com/50104866/168268460-01549a0a-a511-4134-91bc-c199546efb39.png)

A set of 2D points uniformly generated using f(x,y), between the range of {-1.0, 1.0}, are interpolated using RBF interpolation.
