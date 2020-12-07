import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

# ----------------------------------------------------
# Vorlesung 3, Folie 25 + 26
# ----------------------------------------------------

# Load data
data = np.loadtxt("../data/decomp_test.txt", delimiter=",")

# 2 components
model = PCA(n_components=2)

# Train and transform model
data_proj = model.fit_transform(data)
x = model.components_
y = model.explained_variance_ratio_

print(x)
print(y)


# 1 component
model = PCA(n_components=1)

# Train and transform model
data_proj = model.fit_transform(data)
y = np.zeros(100)
plt.plot(data_proj, y, 'mo')
plt.show()

# Inverse Projection
data_recovered = model.inverse_transform(data_proj)
plt.plot(data_recovered[:, 0], data_recovered[:, 1], 'mo')
plt.plot(data[:, 0], data[:, 1], 'bo')
plt.show()
