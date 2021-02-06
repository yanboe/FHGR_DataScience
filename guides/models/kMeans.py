import matplotlib.pyplot as plt
import matplotlib
from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons
import numpy as np
from sklearn.cluster import KMeans

# 2x2 Plots definieren (kein Lernstoff)
fig, a = plt.subplots(2, 2)

# Blobs erstellen und plotten
x, y_true = make_blobs(n_samples=300, centers=4)  # 300 Samples, 4 Cluster
colors = ['red', 'blue', 'green', 'cyan']  # Farben definieren
cmap = matplotlib.colors.ListedColormap(colors)  # Colormap erzeugen
a[0][0].scatter(x[:, 0], x[:, 1], c=y_true, cmap=cmap)  # Diagramm erstellen (links oben)

# Trägheit berechnen
inert = []
for k in range(1, 11):
    model = KMeans(n_clusters=k)  # kMeans aufsetzen
    model.fit(x)  # Trainieren
    inert.append(model.inertia_)  # Trägheit jedes k-Werts einem Array (inert) hinzufügen

# Ellenbogen-Methode plotten (rechts oben)
x_space = np.linspace(1, 10, 10)
a[0][1].plot(x_space, inert, 'b-')
a[0][1].plot(x_space, inert, 'bo')
a[0][1].axis([1, 10, 0, 30000])

# Moons erstellen und plotten
x, y_true = make_moons(n_samples=200, noise=0.05)  # 200 Samples, 0.05 Noise
colors = ['red', 'blue']  # Farben definieren
cmap = matplotlib.colors.ListedColormap(colors)  # Colormap erzeugen
a[1][0].scatter(x[:, 0], x[:, 1], c=y_true, cmap=cmap)  # Diagramm erstellen (link sunten)

# Trägheit berechnen
inert = []
for k in range(1, 11):
    model = KMeans(n_clusters=k)  # kMeans aufsetzen
    model.fit(x)  # Trainieren
    inert.append(model.inertia_)  # Trägheit jedes k-Werts einem Array (inert) hinzufügen

# Ellenbogen-Methode plotten (rechts unten)
x_space = np.linspace(1, 10, 10)
a[1][1].plot(x_space, inert, 'b-')
a[1][1].plot(x_space, inert, 'bo')
a[1][1].axis([1, 10, 0, 300])

# Plot anzeigen
plt.tight_layout()  # Kein Lernstoff
plt.show()
