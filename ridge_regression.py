import numpy as np
import matplotlib.pyplot as plt

train_x=np.load("P3_data/train.npz")["x"]# 25 data points
train_y=np.load("P3_data/train.npz")["y"]

test_x=np.load("P3_data/test.npz")["x"]#100 data points
test_y=np.load("P3_data/test.npz")["y"]

Phi_x = np.hstack([(train_x**d).reshape(-1,1) for d in range(10)]) # Transform dataset with polynomial basis functions

folds_x = np.array_split(Phi_x, 5)
folds_y = np.array_split(train_y, 5)

# Array of lambdas to test in our validation
lambdas = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, .5, 1]
lambda_error = []
lambda_weights = []

# Runs a for loop to test each possible regularization parameter
for lambda_test in lambdas:
    average_error = 0
    weights = []
    # Runs a for loop where in each iteration you pick a new validation set
    for i in range(5):

        val_x = folds_x[i]  # Validation set for x values
        val_y = folds_y[i]  # Validation set for y values

        train_set_x = np.concatenate([folds_x[j] for j in range(5) if j != i])  # Remaining x values as training data
        train_set_y = np.concatenate([folds_y[j] for j in range(5) if j != i])  # Remaining x values as training data

        # Performs ridge regression with the test lambda
        Identity = np.eye(train_set_x.shape[1])
        W = np.linalg.inv(train_set_x.T @ train_set_x + lambda_test*Identity) @ train_set_x.T @ train_set_y # Using normal equation, compute optimal W with the train set
        
        weights.append(W)
        # Now evaluate the loss with the validation set
        predicted_y = val_x @ W
        MSE = np.mean((val_y-predicted_y)**2)

        # Update average error
        average_error*=i
        average_error+=MSE
        average_error/=(i+1)

    print(f'Average error for lambda = {lambda_test} is {average_error}')
    lambda_error.append(average_error)
    lambda_weights.append(np.mean(weights, axis=0))

# Finds the minimum error in the lambda_error array and matches that to the lambda
print(f'The lambda that produced the minimum error in validation was lambda = {lambdas[np.argmin(lambda_error)]}')
optimal_weight = lambda_weights[np.argmin(lambda_error)]
print(f'The average weight for the optimal lambda is then:\n{optimal_weight}')

Phi_x_test = np.hstack([(test_x**d).reshape(-1,1) for d in range(10)]) # Transform dataset with polynomial basis functions
predicted_y_test = Phi_x_test @ optimal_weight
Test_MSE = np.mean((test_y-predicted_y_test)**2)
print(f'The average error found in testing for this model with lambda = 0.0001 was MSE = {Test_MSE}')

import matplotlib.pyplot as plt

# Plot the lambda values against the corresponding error values
plt.figure(figsize=(8, 6))
plt.plot(lambdas, lambda_error, marker='o', linestyle='-', color='b')

# Adding labels and title
plt.xlabel('Lambda (Regularization Parameter)')
plt.ylabel('Average Error (MSE)')
plt.title('Ridge Regression: Average MSE vs. Lambda')

# Logarithmic scale to better show the scaling of lambda
plt.xscale('log')
plt.grid(True)
plt.show()