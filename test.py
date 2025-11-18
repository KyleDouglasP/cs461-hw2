import numpy as np
import matplotlib.pyplot as plt

train_x_100=np.load("P3_data/train_100.npz")["x"] #100 data points
train_y_100=np.load("P3_data/train_100.npz")["y"]

plt.figure(figsize=(8, 6))
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
# Show the plot