import numpy as np
import tensorly as tl

# ----------------------------------------------------
# Vorlesung 4, Folie 17
# ----------------------------------------------------

# Erzeugen eines Tensors X (3 x 4 x 2)
X = np.arange(24).reshape((3, 4, 2))
X = X + 1  # 1 addieren, um 1-24 zu erhalten (statt 0-23)

# Darstellen der frontalen Schichten X::0 und X::1
print(X[:, :, 0])
print(X[:, :, 1])

# Mode-n-Entfaltung X(0), X(1) und X(2)
print("X(0): \n", tl.unfold(X, 0))
print("X(1): \n", tl.unfold(X, 1))
print("X(2): \n", tl.unfold(X, 2))
