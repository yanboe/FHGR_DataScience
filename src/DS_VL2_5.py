import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# -------------------------------
# Vorlesung 2, Slide 25, k-Mitten
# -------------------------------

# Load data
data = np.loadtxt("../data/clicks.txt", delimiter=",")

# Plot data
# plt.plot(data[:, 0], data[:, 1], 'ro')
# plt.show()

# Create model and classify data (create clusters)
model = KMeans(n_clusters=3, max_iter=10)
y_pred = model.fit_predict(data)

# Plot the diagram
plt.axis([0, 70, 0, 10])
colors = ['red', 'blue', 'green']
cmap = matplotlib.colors.ListedColormap(colors)

# with c=y_pred, we color the dots based on the clusters we created earlier
plt.scatter(data[:, 0], data[:, 1], c=y_pred, cmap=cmap)
plt.show()
