import numpy as np

# ------------------------------
# Vorlesung 3, Slide 6, Vektoren
# ------------------------------

# 0-Vektor
null_vec = np.zeros(10)
print(null_vec)

# 1-Vektor
eins_vec = np.ones(10)
print(eins_vec)

# Vektorlänge berechnen
x = np.linalg.norm(eins_vec)
print(x)

# Skalarprodukt
vec1 = [3, 1, 2]
vec2 = [4, 0, 3]
print(np.dot(vec1, vec2))

# Skalarprodukt senkrechte Vektoren
vec1 = [1, 0]
vec2 = [0, 1]
print(np.dot(vec1, vec2))
