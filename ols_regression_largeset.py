import numpy as np
import matplotlib.pyplot as plt

train_x_100=np.load("P3_data/train_100.npz")["x"] #100 data points
train_y_100=np.load("P3_data/train_100.npz")["y"]

Phi_x = np.hstack([(train_x_100**d).reshape(-1,1) for d in range(10)]) # Transform dataset with polynomial basis functions

W = np.linalg.inv(Phi_x.T @ Phi_x) @ Phi_x.T @ train_y_100 # Using normal equation, compute optimal W with the train set

# Define a range [0,1] and transform it to the feature space
test_range = np.linspace(0, 1, 100)
Phi_test = np.hstack([(test_range**d).reshape(-1,1) for d in range(10)]) # Transform dataset with polynomial basis functions

# Predict the y value over the range
y_predicted = Phi_test @ W

# Plot the two models
plt.figure(figsize=(8, 6))
plt.plot(test_range, y_predicted, label="Regression Model", color='blue', linestyle='-', linewidth=2)
plt.scatter(train_x_100, train_y_100, label="Original Data", color='red', linestyle='-', linewidth=2)

# Adding labels and title
plt.xlabel("x")
plt.ylabel("Predicted y")
plt.title("Regression Model with 100 Data Points")

# Add a legend
plt.legend()

# Display grid
plt.grid(True)

# Show the plot
plt.show()