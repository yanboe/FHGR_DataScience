import numpy as np

# ----------------------------------------------------
# Vorlesung 3,
# ----------------------------------------------------

# Load data
a = np.loadtxt("../data/matA.txt", delimiter=",")
b = np.loadtxt("../data/matB.txt", delimiter=",")

# A
print(a)
print(np.shape(a))
a_transposed = np.transpose(a)
print(a_transposed)
print(np.shape(a_transposed))

# B
print(b)
print(np.shape(b))
b_transposed = np.transpose(b)
print(b_transposed)
print(np.shape(b_transposed))

# Matrixmultiplikation
c = np.dot(a, b_transposed)
print(c)