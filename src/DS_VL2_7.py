import matplotlib.pyplot as plt
import matplotlib
from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons
import numpy as np
from sklearn.cluster import KMeans


# ----------------------------------------------------
# Vorlesung 2, Slide 30, Blobs and Moons
# ----------------------------------------------------
fig, a = plt.subplots(2, 2)

# Create and plot blobs
x, y_true = make_blobs(n_samples=300, centers=4)
colors = ['red', 'blue', 'green', 'cyan']
cmap = matplotlib.colors.ListedColormap(colors)
a[0][0].scatter(x[:, 0], x[:, 1], c=y_true, cmap=cmap)

# Calculate and plot inertia
inert = []
for k in range(1, 11):
    model = KMeans(n_clusters=k)
    model.fit(x)
    inert.append(model.inertia_)

x_space = np.linspace(1, 10, 10)
a[0][1].plot(x_space, inert, 'b-')
a[0][1].plot(x_space, inert, 'bo')
a[0][1].axis([1, 10, 0, 30000])

# Create and plot moons
x, y_true = make_moons(n_samples=200, noise=0.05)
colors = ['red', 'blue']
cmap = matplotlib.colors.ListedColormap(colors)
a[1][0].scatter(x[:, 0], x[:, 1], c=y_true, cmap=cmap)

# Calculate and plot inertia
inert = []
for k in range(1, 11):
    model = KMeans(n_clusters=k)
    model.fit(x)
    inert.append(model.inertia_)

x_space = np.linspace(1, 10, 10)
a[1][1].plot(x_space, inert, 'b-')
a[1][1].plot(x_space, inert, 'bo')
a[1][1].axis([1, 10, 0, 300])

plt.tight_layout()
plt.show()
