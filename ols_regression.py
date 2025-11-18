import numpy as np
import matplotlib.pyplot as plt

train_x=np.load("P3_data/train.npz")["x"]# 25 data points
train_y=np.load("P3_data/train.npz")["y"]

test_x=np.load("P3_data/test.npz")["x"]#100 data points
test_y=np.load("P3_data/test.npz")["y"]

Phi_x = np.hstack([(train_x**d).reshape(-1,1) for d in range(10)]) # Transform dataset with polynomial basis functions
# print(Phi_x)

folds_x = np.array_split(Phi_x, 5)
folds_y = np.array_split(train_y, 5)

average_error = 0
Weights = []
# Runs a for loop where in each iteration you pick a new validation set

for i in range(5):

    val_x = folds_x[i]  # Validation set for x values
    val_y = folds_y[i]  # Validation set for y values

    train_set_x = np.concatenate([folds_x[j] for j in range(5) if j != i])  # Remaining x values as training data
    train_set_y = np.concatenate([folds_y[j] for j in range(5) if j != i])  # Remaining x values as training data

    W = np.linalg.inv(train_set_x.T @ train_set_x) @ train_set_x.T @ train_set_y # Using normal equation, compute optimal W with the train set

    # Now evaluate the loss with the validation set

    Weights.append(W)

    predicted_y = val_x @ W
    MSE = np.mean((val_y-predicted_y)**2)
    average_error*=i
    average_error+=MSE
    average_error/=(i+1)
    print(f'Error for iteration {i+1} is {MSE}')

print(f'Average error of the 5 folds is {average_error}')
average_weight = np.mean(Weights, axis=0)
print(f'Average weights computed:\n{average_weight}')

Phi_x_test = np.hstack([(test_x**d).reshape(-1,1) for d in range(10)]) # Transform dataset with polynomial basis functions
predicted_y_test = Phi_x_test @ average_weight
Test_MSE = np.mean((test_y-predicted_y_test)**2)
print(f'The average error found in testing for this model was MSE = {Test_MSE}')