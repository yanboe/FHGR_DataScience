from sklearn import datasets
import matplotlib.pyplot as plt

# ----------------------------------------------------
# Vorlesung 3, Folie 16
# ----------------------------------------------------

# Get data
data, shape = datasets.make_swiss_roll(n_samples=1000, noise=0.0)

# Create plot
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=shape)
plt.show()
