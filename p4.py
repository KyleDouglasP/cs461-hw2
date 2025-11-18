import os
import numpy as np
from PIL import Image
from PIL import UnidentifiedImageError
import matplotlib.pyplot as plt

images = []
for file in os.listdir("P4_data/train"):
    try:
        img_path = os.path.join("P4_data/train", file)
        img = Image.open(img_path)
        images.append(np.array(img).flatten())
    except UnidentifiedImageError:
        pass

mean = np.mean(images, axis=0)
print("The mean vector shape is ", mean.shape)

centered_images = images - mean

cov = np.matmul(centered_images.T, centered_images) / (centered_images.shape[0] - 1)
print("The covariance matrix shape is ", cov.shape)

eigenvalues, eigenvectors = np.linalg.eigh(cov)
print("There are ", eigenvalues.size, " eigenvalues")

sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

# Load a random test image
test_image = Image.open("P4_data/test/subject14.surprised")
test_image = np.array(test_image).flatten()

test_image_centered = test_image - mean

def reconstruct_image(test_image_centered, mean, eigenvectors, M):
    top_eigenvectors = eigenvectors[:, :M]  
    projection = np.dot(test_image_centered, top_eigenvectors) 
    reconstructed = np.dot(projection, top_eigenvectors.T)  
    return reconstructed + mean  

M_test = [2, 10, 100, 1000, 4000]

plt.figure(figsize=(10, 5))
for index, M in enumerate(M_test):
    top_M_eigenvectors = eigenvectors[:, :M]
    reconstructed = (test_image_centered @ top_M_eigenvectors @ top_M_eigenvectors.T) + mean
    reconstructed = reconstructed.reshape(60,80)
    plt.subplot(1, 5, index+1)
    plt.imshow(reconstructed, cmap="magma")
    plt.title(f"M={M}")
    plt.axis("off")

plt.show()
plt.close()

plt.figure(figsize=(10, 5))
# Iterate over the top 10 eigenvectors
for i in range(10):
    eigenface = eigenvectors[:, i]
    eigenface = eigenface.reshape(60, 80)  # Reshape to 60x80 image dimensions
    eigenface_normalized = (eigenface - eigenface.min()) / (eigenface.max() - eigenface.min()) * 255  # Normalize to [0, 255]
    plt.subplot(2, 5, i + 1)  # Create a 2x5 grid for the first 10 eigenfaces
    plt.imshow(eigenface_normalized, cmap="magma")
    plt.title(f"Eigenface #{i+1}")
    plt.axis("off")

plt.show()