from tensorly.decomposition import parafac
import tensorly as tl
import matplotlib.pyplot as plt

# ----------------------------------------------------
# Vorlesung 4, Folie 22
# ----------------------------------------------------

# Bild laden
lena = plt.imread("../data/lena.png")

# Tensorfaktorisierung
w_, fac = parafac(lena, 256)  # Je kleiner, desto schlechter das Bild (dafür kleinere Bildgrösse)
print(w_)
lena_rec = tl.kruskal_to_tensor((w_, fac))
plt.imshow(lena_rec, cmap="gray")
plt.show()
