import numpy as np
import tensorly as tl

# ----------------------------------------------------
# Vorlesung 4, Folie 10
# ----------------------------------------------------

a = [[2, 1], [1, 3]]
b = [[3, 3], [-2, 4]]

# Kronecker-Produkt
print("Kronecker-Produkt: \n", tl.tenalg.kronecker((a, b)))

# Khatri-Rao-Produkt
a = [2, 1, 1, 3]
a = np.reshape(a, (2, 2))
b = [3, 3, -2, 4]
b = np.reshape(b, (2, 2))
print("Khatri-Rao-Produkt: \n", tl.tenalg.khatri_rao((a, b)))

# Hadamard-Produkt (kann Python sowieso)
print("Hadamard-Produkt: \n", a*b)
