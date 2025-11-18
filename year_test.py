import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

W = np.array([ 1.79651288e+03,  4.61254049e+01 , 1.14196321e+02,  7.36370978e+01,
 -4.24082643e+01,  2.32477998e+01, -9.71219833e+00 , 1.67766542e+01,
 -2.61039572e+00, -1.32098262e-01])
print(W.shape)

def polynomial(data, degree=3):
    Phi_x = np.ones((data.shape[0], 1)) # Bias term
    for d in range(1, degree + 1):
        for i in range(d + 1):
            j = d - i  # To ensure we generate terms like x^2, y^2, xy, etc.
            new_feature = (data[:, 0]**i) * (data[:, 1]**j)
            Phi_x = np.hstack([Phi_x, new_feature.reshape(-1, 1)])
    return Phi_x

test_logit=np.load("P5_data/vgg16_test.npz")["logit"]
test_year=np.load("P5_data/vgg16_test.npz")["year"]
test_filename=np.load("P5_data/vgg16_test.npz", allow_pickle=True)["filename"] 

mean = np.sum(test_logit, axis=0, keepdims=True) / test_logit.shape[0]

test_logit_centered = test_logit-mean
cov = np.matmul(test_logit_centered.T, test_logit_centered) / (test_logit_centered.shape[0] - 1)

# Eigen decomposition of the covariance matrix
eigenvalues, eigenvectors = np.linalg.eigh(cov)

# Sort eigenvalues and eigenvectors for PCA
sorted_indices = np.argsort(eigenvalues)[::-1]  # Sort in descending order
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

two_dimensional_data = np.dot(test_logit_centered, eigenvectors[:, :2]) * (1 / np.sqrt(eigenvalues[:2]))

Phi_test = polynomial(two_dimensional_data)
print(Phi_test.shape)
y_predicted = Phi_test @ W

SE = (test_year-y_predicted)**2
MSE = np.mean(SE)
print(f'Average error is {MSE}')
print(f'Most accurate prediction was {test_filename[np.argmin(SE)]} with SE={np.min(SE)}')
print(f'Least accurate prediction was {test_filename[np.argmax(SE)]} with SE={np.max(SE)}')


fig = plt.figure(figsize=(12, 6))

cmap = mpl.cm.viridis
norm = mpl.colors.Normalize(vmin=1148, vmax=2012)

ax = fig.add_subplot(121, projection='3d')
ax.set_title("Predicted Years")
scatter2 = ax.scatter(two_dimensional_data[:, 0], -two_dimensional_data[:, 1], y_predicted, c=y_predicted, cmap=cmap, s=2, picker=4)
ax2 = fig.add_subplot(122, projection='3d')
ax2.set_title("Actual Years")
scatter2 = ax2.scatter(two_dimensional_data[:, 0], -two_dimensional_data[:, 1], test_year, c=test_year, cmap=cmap, s=2, picker=4)

# Set prediction to the same viewbox
ax.set_xlim(ax2.get_xlim())
ax.set_ylim(ax2.get_ylim())
ax.set_zlim(ax2.get_zlim())

plt.show()