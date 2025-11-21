import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

W = np.array([ 1786.9698684523012, 121.512865758543, 135.0613692698496, 109.9214405007493, -108.09920026487828, 23.25854543684881, -62.52672358038889, 8.01631295025755, -55.156723681489495, -4.970100573598966, -9.881991369284515, -21.474129987758236, -4.569036585525209, -10.118143720608252, -1.4675213513741237, 0.7918527585952612, -2.113037296666221, -3.1717326951392533, -0.5853961561820902, -0.4361064396921838, -0.0665283169550251])
print(W.shape)

def polynomial(data, degree=5):
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

# Load PCA parameters
pca_params = np.load("pca_params.npz")
mean = pca_params["mean"]
eigenvectors = pca_params["eigenvectors"]
eigenvalues = pca_params["eigenvalues"]

test_logit_centered = test_logit-mean

two_dimensional_data = np.dot(test_logit_centered, eigenvectors[:, :2]) * (1 / np.sqrt(eigenvalues[:2]))

Phi_test = polynomial(two_dimensional_data)
print(Phi_test.shape)
y_predicted = Phi_test @ W

SE = (test_year-y_predicted)**2
MSE = np.mean(SE)
print(f'Average error is {MSE}')
min_idx = np.argmin(SE)
max_idx = np.argmax(SE)
print(f'Most accurate prediction was {test_filename[min_idx]} with SE={np.min(SE)}')
print(f'Least accurate prediction was {test_filename[max_idx]} with SE={np.max(SE)}')

# Try to find image files for the most/least accurate examples inside P5_data
def find_image_file(fname):
    try:
        fname = str(fname)
    except Exception:
        pass
    base = os.path.basename(fname)
    candidates = [
        os.path.join("P5_data", fname),
        os.path.join("P5_data", base),
        os.path.join("P5_data", "test", fname),
        os.path.join("P5_data", "train", fname),
        os.path.join("P4_data", "test", fname),
        os.path.join("P4_data", "train", fname),
        os.path.join("img", fname),
        fname,
    ]
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None

min_file = find_image_file(test_filename[min_idx])
max_file = find_image_file(test_filename[max_idx])

fig_imgs = plt.figure(figsize=(8, 4))
ax1 = fig_imgs.add_subplot(1, 2, 1)
ax2 = fig_imgs.add_subplot(1, 2, 2)

if min_file is not None:
    img_min = mpimg.imread(min_file)
    ax1.imshow(img_min)
    ax1.set_title(f"Most accurate\nPred={y_predicted[min_idx]:.1f}, True={test_year[min_idx]}, SE={SE[min_idx]:.2f}")
    ax1.axis('off')
else:
    ax1.text(0.5, 0.5, f"File not found:\n{test_filename[min_idx]}", ha='center', va='center')
    ax1.axis('off')

if max_file is not None:
    img_max = mpimg.imread(max_file)
    ax2.imshow(img_max)
    ax2.set_title(f"Least accurate\nPred={y_predicted[max_idx]:.1f}, True={test_year[max_idx]}, SE={SE[max_idx]:.2f}")
    ax2.axis('off')
else:
    ax2.text(0.5, 0.5, f"File not found:\n{test_filename[max_idx]}", ha='center', va='center')
    ax2.axis('off')

plt.tight_layout()

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