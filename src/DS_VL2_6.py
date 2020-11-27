import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# ----------------------------------------------------
# Vorlesung 2, Slide 27, k-Mitten (Ellenbogen-Methode)
# ----------------------------------------------------

# Load data
data = np.loadtxt("../data/clicks.txt", delimiter=",")

# Calculate inertia
inert = []
for k in range(1, 11):
    model = KMeans(n_clusters=k)
    model.fit(data)
    inert.append(model.inertia_)

# Plot data
x = np.linspace(1, 10, 10)  # create dots 1-10
plt.axis([1, 10, 0, 160000])
plt.plot(x, inert, 'b-')
plt.plot(x, inert, 'ro')
plt.show()
