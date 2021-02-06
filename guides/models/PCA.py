from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# Funktion, um 4x10 zu zeichnen (von Mundani bereitgestellt)
def plot_digits(data):
    figr, axes = plt.subplots(4, 10, figsize=(10, 4),
                              subplot_kw={'xticks': [], 'yticks': []},
                              gridspec_kw=dict(hspace=0.1, wspace=0.1))
    for i, ax in enumerate(axes.flat):
        ax.imshow(data[i].reshape(8, 8),
                  cmap='binary', interpolation='nearest',
                  clim=(0, 16))


# Daten laden und Array-Grösse ausgeben
digits = load_digits()
print(digits.data.shape)  # 1797 Bilder mit Auflösung 8x8

# Bilder in 4x10 anzeigen
plot_digits(digits.data)
plt.show()

# Dimensionsreduktion auf 2 Hauptkomponenten
model = PCA(n_components=2)  # PCA erzeugen mit 2 Hauptkomponenten
d_proj = model.fit_transform(digits.data)  # Daten projizieren
print(model.explained_variance_ratio_)  # Varianz ausgeben

# Scatterplot erstellen und anzeigen
plt.scatter(d_proj[:, 0], d_proj[:, 1], c=digits.target, alpha=0.5, cmap=plt.cm.get_cmap('nipy_spectral', 10))
plt.colorbar()
plt.show()

# Projektion umkehren
d_recov = model.inverse_transform(d_proj)
plot_digits(d_recov)
plt.show()
