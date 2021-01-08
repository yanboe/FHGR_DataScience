import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# ----------------------------------------------------
# Vorlesung 3, Folie 31-35
# ----------------------------------------------------

# Bild laden
img = plt.imread("../data/lena.png")
plt.imshow(img, cmap="gray")
plt.show()


# Experiment 1: Verteilung der Varianz
model = PCA().fit(img)
print(model.n_components_)  # 512 Hauptkomponenten (PCA = 0, d.h. alle)
x = np.linspace(1, 512, 512)
plt.plot(x, model.explained_variance_ratio_, "r.")
plt.plot(np.cumsum(model.explained_variance_ratio_), "b-")
plt.show()


# Experiment 2: Hauptkomponentenzerlegung
model = PCA(0.9)  # Varianz in Prozent (0.3 = 30%)
img_rec = model.inverse_transform(model.fit_transform(img))
plt.imshow(img_rec, cmap="gray")
plt.show()

# Differenz zwischen Originalbild und reduziertem Bild anzeigen (was ist verloren gegangen)
diff = img - img_rec
plt.imshow(diff, cmap="gray")
plt.show()


# Experiment 3: Entrauschung
img_noise = plt.imread("../data/lena_noise.png")
plt.imshow(img_noise, cmap="gray")
plt.show()
model = PCA(0.9)  # 90%
de_noise = model.inverse_transform(model.fit_transform(img_noise))
plt.imshow(de_noise, cmap="gray")
plt.show()

# Vergleich von Originalbild (schwarz/weiss) und Bild mit Noise
model = PCA()
model.fit(img)
model2 = PCA()
model2.fit(img_noise)
plt.plot(np.cumsum(model.explained_variance_ratio_), "b-")
plt.plot(np.cumsum(model2.explained_variance_ratio_), "r-")
plt.show()


# Experiment 4: Farbraumreduzierung
img_orig = plt.imread("../data/lena_color.png")
plt.imshow(img_orig)
plt.show()

lena_lin = np.reshape(img_orig, (512*512, 3))  # Linearisieren
plt.scatter(lena_lin[:, 0], lena_lin[:, 1], c=lena_lin[:, 2])
# Viel PC Power benötigt für die nächste Zeile
plt.show()

# 16 Cluster
model = KMeans(16)

# Viel PC Power benötigt für die nächste Zeile
model.fit(lena_lin)  # Model trainieren
data_reduced = model.cluster_centers_[model.predict(lena_lin)]
img16 = np.reshape(data_reduced, (512, 512, 3))
plt.imshow(img16)
plt.show()
