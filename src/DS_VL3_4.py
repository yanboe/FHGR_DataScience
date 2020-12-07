import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------------------------
# Vorlesung 3, Folie 23
# ----------------------------------------------------

# Load data
data = np.loadtxt("../data/decomp_test.txt", delimiter=",")

# Plot
plt.plot(data[:, 0], data[:, 1], 'ro')
plt.show()

# Hauptkomponentenzerlegung
u, sigma, vt = np.linalg.svd(data)
plt.quiver([0, 0], [0, 0], vt[:, 0], vt[:, 1], scale=np.sort(sigma))
plt.scatter(data[:, 0], data[:, 1])
plt.show()