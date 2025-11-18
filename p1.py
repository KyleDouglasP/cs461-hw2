import numpy as np

# Define matrix Φ (4x4)
Phi = np.array([
    [1, 4, 1, 1],
    [1, 7, 0, 2],
    [1, 10, 1, 3],
    [1, 13, 0, 4]
])

# Define target vector y
y = np.array([16, 23, 36, 43])

# Compute normal equation solution if possible
Phi_T_Phi = Phi.T @ Phi

# Checking for full-rank
print("The Rank is:", np.linalg.matrix_rank(Phi_T_Phi))

# Compute psuedo-inverse solution
psuedo_inverse = np.linalg.pinv(Phi_T_Phi)
print("The psuedo-inverse is:\n", psuedo_inverse)

w_pseudo = psuedo_inverse @ Phi.T @ y
print("w using pseudo-inverse:", w_pseudo)

# Perform SVD on Phi_T_Phi
U, S, Vt = np.linalg.svd(Phi_T_Phi)

# Compute the inverse of eigenvalues (handling near-zero ones)
S_inv = np.zeros_like(S)
eps = 1e-10  # Small threshold for near-zero eigenvalues

for i in range(len(S)):
    if S[i] > eps:
        S_inv[i] = 1 / S[i]  # Invert nonzero eigenvalues
    else:
        S_inv[i] = 100  # Arbitrarily large value for values near-zero

# Construct the inverse diagonal matrix
Lambda_inv = np.diag(S_inv)

# Compute w using spectral decomposition
w_svd = U @ Lambda_inv @ Vt @ Phi.T @ y
print("w using spectral decomposition:", w_svd)

# Check if the solutions match
print("Solutions match:", np.allclose(w_pseudo, w_svd))


#Redefine Phi and y with the 4 new data points

Phi = np.array([
    [1, 4, 1, 1],
    [1, 7, 0, 2],
    [1, 10, 1, 3],
    [1, 13, 0, 4],
    [1, 16, 1, 5],
    [1, 19, 0 ,6],
    [1, 22, 1, 7],
    [1, 25, 0 ,8]
])

y = np.array([16.0, 23.0, 36.0, 43.0, 56.04, 62.77, 76.04, 82.96])

Phi_T_Phi = Phi.T @ Phi

# Compute psuedo-inverse solution
psuedo_inverse = np.linalg.pinv(Phi_T_Phi)

w_pseudo = psuedo_inverse @ Phi.T @ y
print("w including the new 4 datapoints:", w_pseudo)