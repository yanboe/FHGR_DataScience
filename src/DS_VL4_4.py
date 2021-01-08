import numpy as np
from tensorly.decomposition import parafac
import tensorly as tl

# ----------------------------------------------------
# Vorlesung 4, Folie 21
# ----------------------------------------------------

# Erzeugen eines Tensors
X = np.arange(24.).reshape((3, 4, 2))
X = X + 1.

# CP-Faktorisierung
factors = parafac(X, 3)  # Zweiter Parameter ist der Rang
# Rang = 1 -> Ergebnis ungenau
# Rang = 2 -> Ergebnis ziemlich genau
# Rang = 3 -> Ergebnis sehr genau
print("factors:\n", factors)

# In w_ und fac zerlegen
w_, fac = factors
print("w_:\n", w_)
print("fac:\n", fac)

# In u, v und w zerlegen
u, v, w = fac
print("u:\n", u)  # Vektor in erste Richtung
print("v:\n", v)  # Vektor in zweite Richtung
print("w:\n", w)  # Vektor in dritte Richtung

# Tensor aus Faktormatrizen rekonstruieren
X_rec = tl.kruskal_to_tensor(factors)
print("Tensor (mit Tensorly):\n", tl.norm(X - X_rec))  # Originaler Tensor minus rekonstruierter Tensor
print("Tensor (mit Numpy):\n", np.linalg.norm(X - X_rec))  # Originaler Tensor minus rekonstruierter Tensor

# Das hier ist das wichtigste dieser Übung: wie gut können wir den originalen Tensor rekonstruieren?
# Dazu vergleichen wir die Werte des originalen Tensors mit dem rekonstruierten Tensors.
#   Die Genauigkeit hat mit dem Rang weiter oben zu tun:
#   Rang 1 ungenau, Rang 2 recht genau, Rang 3 sehr genau
print("Original Tensor:\n", X)
print("Rekonstruierter Tensor:\n", X_rec)
