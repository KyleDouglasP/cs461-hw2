# CS461 HW2 Samples
Some code samples from Machine Learning Principles HW2, relating to Regression, Principal Component Analysis, and Cross-Fold Validation.

## P1: MMSE Regression

`p1.py` computes the MMSE solution for a small linear regression problem using two methods.

### Normal Equation and Rank Check

The design matrix $\Phi$ is constructed with a bias term, and its rank is checked with `numpy.linalg.matrix_rank`. Since $\Phi^\top \Phi$ is rank deficient, the exact inverse does not exist, so an approximate solution is required.

### Pseudoinverse Solution

Using `numpy.linalg.pinv`, the script computes the weight vector $w=(\Phi^t\Phi)^{+}\Phi^ty$ and prints the result.

### Spectral Decomposition

The file also implements a custom solution using eigen-decomposition. Both solutions are compared with `np.allclose`, confirming they match up to floating point error.

## P3: Learning Sinusoidal Functions

This section includes three main scripts: `ols_regression.py`, `ridge_regression.py`, and `ols_vs_ridge.py`, which work to test regression models on fitting a sinusoidal function.

### Ordinary Least Squares

`ols_regression.py` runs MMSE regression on the sinusoid dataset with 25 training points, using 5-fold validation. It computes the average validation error to be $MSE\approx0.688$.

### Ridge Regression

`ridge_regression.py` evaluates the following lambda values: $\lambda=[1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1]$

Each value is trained and evaluated on the same 25-point training set with 5-fold cross-validation. The script plots average validation error and selects the lambda that minimizes it.

<p align="center">
  <img src="https://raw.githubusercontent.com/KyleDouglasP/cs461-hw2/refs/heads/main/img/Ridge_Regression_Plot.PNG" />
</p>

In this case $\lambda=10^{-4}$ minimized the MSE.

### Model Comparison

`ols_vs_ridge.py` loads both sets of weights and maps them over a dense x-range. It produces a comparison plot that highlights how ridge reduces overfitting and smooths the function.

<p align="center">
  <img src="https://github.com/KyleDouglasP/cs461-hw2/blob/main/img/Ridge_VS_OLS.PNG" />
</p>

On the edges of the range, it can be seen that the model with $\lambda=0$ fails to generalize, and goes way off of a normal sinusoidal function. Also, for both models an averaged weight was found for the 5 folds, running a test to find the average MSE for both:

$w(\lambda=0): MSE=0.4441293406629071$

$w(\lambda*): MSE = 0.037604624960812676$

### Base Model

I also tested a model that had no regularization or cross validation using 100 training points instead of the earlier 25 points, plotting the results:

<p align="center">
  <img src="https://github.com/KyleDouglasP/cs461-hw2/blob/main/img/OLS_Regression_100_Plot.PNG" />
</p>

It can be seen that another way to effectively reduce error is to introduce more training points, reducing the necessity of regularization.
