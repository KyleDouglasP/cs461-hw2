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

np.savez("pca_params.npz",
         mean=mean,
         eigenvectors=eigenvectors,
         eigenvalues=eigenvalues)

two_dimensional_data = np.dot(train_logit_centered, eigenvectors[:, :2]) * (1 / np.sqrt(eigenvalues[:2]))

# Cut the data into training and validation sets
train_logit = two_dimensional_data[200:]
val_logit = two_dimensional_data[0:200]
val_year = train_year[0:200]
train_year = train_year[200:]

def polynomial(data, degree=5):
    Phi_x = np.ones((data.shape[0], 1)) # Bias term
    for d in range(1, degree + 1):
        for i in range(d + 1):
            j = d - i  # To ensure we generate terms like x^2, y^2, xy, etc.
            new_feature = (data[:, 0]**i) * (data[:, 1]**j)
            Phi_x = np.hstack([Phi_x, new_feature.reshape(-1, 1)])
    return Phi_x

# Creating the polynomial basis functions
Phi_x = polynomial(train_logit)

W = np.linalg.inv(Phi_x.T @ Phi_x) @ Phi_x.T @ train_year  # Using normal equation, compute optimal W with the train set
print(f'Weight vector:')
print("[", end="")
for weight in W:
    print(f'{weight}, ', end="")
print("]")

Phi_test = polynomial(val_logit)
y_predicted = Phi_test @ W

MSE = np.mean((val_year-y_predicted)**2)
print(f'Average error is {MSE}')

fig = plt.figure(figsize=(12, 6))

cmap = mpl.cm.viridis
norm = mpl.colors.Normalize(vmin=1148, vmax=2012)

ax = fig.add_subplot(121, projection='3d')
ax.set_title("Predicted Years")
scatter2 = ax.scatter(val_logit[:, 0], -val_logit[:, 1], y_predicted, c=y_predicted, cmap=cmap, s=2, picker=4)
ax2 = fig.add_subplot(122, projection='3d')
ax2.set_title("Actual Years")
scatter2 = ax2.scatter(val_logit[:, 0], -val_logit[:, 1], val_year, c=val_year, cmap=cmap, s=2, picker=4)

# Set prediction to the same viewbox
ax.set_xlim(ax2.get_xlim())
ax.set_ylim(ax2.get_ylim())
ax.set_zlim(ax2.get_zlim())

plt.show()