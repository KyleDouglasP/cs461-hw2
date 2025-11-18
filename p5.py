import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

train_logit=np.load("P5_data/vgg16_train.npz")["logit"]
train_year=np.load("P5_data/vgg16_train.npz")["year"]
test_logit=np.load("P5_data/vgg16_test.npz")["logit"]
test_year=np.load("P5_data/vgg16_test.npz")["year"]

train_logit = np.vstack([train_logit,test_logit])
train_year = np.hstack([train_year,test_year])

mean = np.sum(train_logit, axis=0, keepdims=True) / train_logit.shape[0]

train_logit_centered = train_logit-mean
cov = np.matmul(train_logit_centered.T, train_logit_centered) / (train_logit_centered.shape[0] - 1)

# Eigen decomposition of the covariance matrix
eigenvalues, eigenvectors = np.linalg.eigh(cov)

# Sort eigenvalues and eigenvectors for PCA
sorted_indices = np.argsort(eigenvalues)[::-1]  # Sort in descending order
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

one_dimensional_data = np.dot(train_logit_centered, eigenvectors[:, :1]) * (1 / np.sqrt(eigenvalues[0]))
two_dimensional_data = np.dot(train_logit_centered, eigenvectors[:, :2]) * (1 / np.sqrt(eigenvalues[:2]))

#maybe useful for problem 5.1

fig = plt.figure(figsize=(12, 6))

cmap = mpl.cm.viridis
norm = mpl.colors.Normalize(vmin=1148, vmax=2012)

ax = fig.add_subplot(121)
fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
scatter = ax.scatter(one_dimensional_data, train_year, c=train_year, cmap=cmap, norm=norm, s=2, picker=4)

ax2 = fig.add_subplot(122, projection='3d')
scatter2 = ax2.scatter(two_dimensional_data[:, 0], -two_dimensional_data[:, 1], train_year, c=train_year, cmap=cmap, s=2, picker=4)

plt.show()