import matplotlib
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------
# Vorlesung 2, Slides 6+7, k-Nächste-Nachbarn
# -------------------------------------------

# Read credit.txt
data = np.loadtxt("../data/credit.txt", delimiter=",")

# Define x and y axis
plt.axis([10, 80, 0, 120])

# Create scatter diagram (Slide 6)
colors = ["red", "blue"]
cmap = matplotlib.colors.ListedColormap(colors)
scatter = plt.scatter(data[:, 0], data[:, 1], c=data[:, 2], cmap=cmap)

# Create legend (Slide 7)
handles, lables = scatter.legend_elements()
labels = ["niedrig", "hoch"]
plt.legend(handles, labels, frameon=False, loc="upper left")

# Show diagram
plt.show()
