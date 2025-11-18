import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

train_logit=np.load("P5_data/vgg16_train.npz")["logit"]
train_year=np.load("P5_data/vgg16_train.npz")["year"]

mean = np.sum(train_logit, axis=0, keepdims=True) / train_logit.shape[0]

train_logit_centered = train_logit-mean
cov = np.matmul(train_logit_centered.T, train_logit_centered) / (train_logit_centered.shape[0] - 1)

# Eigen decomposition of the covariance matrix
eigenvalues, eigenvectors = np.linalg.eigh(cov)

# Sort eigenvalues and eigenvectors for PCA
sorted_indices = np.argsort(eigenvalues)[::-1]  # Sort in descending order
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

two_dimensional_data = np.dot(train_logit_centered, eigenvectors[:, :2]) * (1 / np.sqrt(eigenvalues[:2]))

def polynomial(data, degree=4):
    Phi_x = np.ones((data.shape[0], 1)) # Bias term
    for d in range(1, degree + 1):
        for i in range(d + 1):
            j = d - i  # To ensure we generate terms like x^2, y^2, xy, etc.
            new_feature = (data[:, 0]**i) * (data[:, 1]**j)
            Phi_x = np.hstack([Phi_x, new_feature.reshape(-1, 1)])
    return Phi_x

# Creating the polynomial basis functions
Phi_x = polynomial(two_dimensional_data)

folds_x = np.array_split(Phi_x, 10)
folds_y = np.array_split(train_year, 10)

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
        average_error+=MSE

    average_error/=(i+1)
    print(f'Average error for lambda = {lambda_test} is {average_error}')
    lambda_error.append(average_error)
    lambda_weights.append(np.mean(weights, axis=0))

# Finds the minimum error in the lambda_error array and matches that to the lambda
print(f'The lambda that produced the minimum error in validation was lambda = {lambdas[np.argmin(lambda_error)]}')
optimal_weight = lambda_weights[np.argmin(lambda_error)]
print(f'The average weight for the optimal lambda is then:\n{optimal_weight}')

# fig = plt.figure(figsize=(12, 6))

# cmap = mpl.cm.viridis
# norm = mpl.colors.Normalize(vmin=1148, vmax=2012)

# ax = fig.add_subplot(121, projection='3d')
# ax.set_title("Predicted Years")
# scatter2 = ax.scatter(val_logit[:, 0], -val_logit[:, 1], y_predicted, c=y_predicted, cmap=cmap, s=2, picker=4)
# ax2 = fig.add_subplot(122, projection='3d')
# ax2.set_title("Actual Years")
# scatter2 = ax2.scatter(val_logit[:, 0], -val_logit[:, 1], val_year, c=val_year, cmap=cmap, s=2, picker=4)

# # Set prediction to the same viewbox
# ax.set_xlim(ax2.get_xlim())
# ax.set_ylim(ax2.get_ylim())
# ax.set_zlim(ax2.get_zlim())

# plt.show()