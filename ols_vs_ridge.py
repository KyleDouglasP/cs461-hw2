import numpy as np
import matplotlib.pyplot as plt

W_0 = np.array([1.68937617e+00, -5.73601930e+01,  8.92068743e+02, -6.58561302e+03, 2.79037866e+04, -7.23796763e+04,  1.15976277e+05, -1.11676469e+05, 5.91992752e+04, -1.32777181e+04])

W_lambda = np.array([0.05588899, 6.71593665, -12.7471026, -7.25949516, 3.91699304, 8.5414802, 6.9767911, 2.45489886,  -2.50676441, -6.57519645])

test_range = np.linspace(0, 1, 100)
Phi_test = np.hstack([(test_range**d).reshape(-1,1) for d in range(10)]) # Transform dataset with polynomial basis functions

y_w0 = Phi_test @ W_0
y_lambda_star = Phi_test @ W_lambda

# Plot the two models
plt.figure(figsize=(8, 6))
plt.plot(test_range, y_w0, label="Model (λ = 0) - MMSE", color='blue', linestyle='-', linewidth=2)
plt.plot(test_range, y_lambda_star, label="Model (λ*) - Ridge", color='red', linestyle='-', linewidth=2)

# Adding labels and title
plt.xlabel("x")
plt.ylabel("Predicted y")
plt.title("Comparison of MMSE (λ = 0) and Ridge Regression (λ*)")

# Add a legend
plt.legend()

# Display grid
plt.grid(True)

# Show the plot
plt.show()