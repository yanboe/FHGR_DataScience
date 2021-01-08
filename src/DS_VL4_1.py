import numpy as np

# ----------------------------------------------------
# Vorlesung 4, Folie 6
# ----------------------------------------------------

u = [3, 1, 2]
v = [4, 0, 3]
w = [2, 3, 4]

# Berechnung des dyadischen Produkts (zwei Vektoren)
print("Dyadisches Produkt (2 Vektoren): \n", np.outer(u, v))

# Berechnung des äusseren Produkts (zwei Vektoren)
print("Äusseres Produkt (2 Vektoren): \n", np.einsum('i,j -> ij', u, v))

# Berechnung des äusseren Produkts (drei Vektoren)
x = np.einsum('i,j,k->ijk', u, v, w)
print("Äusseres Produkt (3 Vektoren): \n", x)

# x
a = np.outer(u, v)
x = np.einsum('ij,k->ijk', a, w)
print("Irgendwas mit Einstein?: \n", x)
