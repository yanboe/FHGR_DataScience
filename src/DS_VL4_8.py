from sklearn.datasets import fetch_olivetti_faces
from mytensor import plot_faces
from tensorly.decomposition import tucker
import numpy as np
import tensorly as tl

# ----------------------------------------------------
# Vorlesung 4, Folie 32-35
# ----------------------------------------------------

# Olivetti-Datensatz (40 Personen mit jeweils 10 Gesichtsausdrücken)
data = fetch_olivetti_faces()

# Bilder anzeigen
plot_faces(data.images, 5, 10, rnd=True)

# Tucker-Zerlegung der Bilder mit Kerntensor 16 x 16 x 16
G, fac = tucker(data.images, (16, 16, 16))

# Was steckt eigentlich im Kerntensor...?
plot_faces(G, 4, 4, rnd=False)

# Und was steckt in den Faktormatrizen...?
G_ = np.ones(16 * 16 * 16).reshape(16, 16, 16)
plot_faces(tl.tucker_to_tensor((G_, fac)), 4, 4, rnd=False)
