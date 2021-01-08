import numpy as np  # Package numpy importieren
import matplotlib.pyplot as plt  # Modul pyplot aus Package matplotlib importieren
from sklearn.linear_model import LinearRegression as lr  # Klasse LinearRegression aus sklearn importieren

# ----------------------------------------------------
# Guide zum Erstellen von Plots
# ----------------------------------------------------

# -----------------------------------------------------------------------------------
# Teil 1: Textdatei einlesen, X- und Y-Achsen definieren, Plot erstellen und anzeigen
# -----------------------------------------------------------------------------------

# File einlesen (mehr Details in "read_files.py")
data = np.loadtxt("../data/smp_data.txt", delimiter=",")

# X- und Y-Achse definieren (data_x ist die erste Spalte, data_y die zweite Spalte des Inputfiles, mehr Details in
# "format_files.py")
data_x = data[:, 0]
data_y = data[:, 1]

# Die X-Achse soll die Anzahl Freunde anzeigen
plt.xlabel("Freunde")

# Die Y-Achse soll die Zeit in Sekunden anzeigen
plt.ylabel("Zeit [Sekunden]")

# Hier definieren wir noch, wie gross die Achsen sein sollen
# Die X-Achse geht von 0-12, die Y-Achse von 0-800
plt.axis([0, 12, 0, 800])

# Mit plt.grid(True) wird ein Grid eingezeichnet, damit wir die einzelnen Punkte besser sehen im Diagramm
plt.grid(True)

# Nun legen wir die Daten der X- und Y-Achse auf den Plot.
# Mit "r." geben wir an, dass die einzelnen Punkte als rote Punkte eingezeichnet werden sollen.
# Wir könnten auch "ro" verwenden, dann sind es rote Kreise.
plt.plot(data_x, data_y, "r.")

# Damit wir das Diagramm anschauen können, müssen wir es noch anzeigen lassen
plt.show()

# -----------------------------------------------------------------------------------
# Teil 2: Regressionsgerade einzeichnen
# -----------------------------------------------------------------------------------
# Wichtig: Bei model.fit gibt es einen Bug. Entweder man fügt ".astype(np.float32)" hinzu,
# oder das "plt.show()" in Teil 1 wird auskommentiert.

# X-Achse reshapen und Y-Achse aufbereiten
data_x = data[:, 0].reshape((-1, 1))
data_y = data[:, 1]

# Die Datenpunkte einzeichnen
plt.plot(data_x, data_y, "r.")

# Regressionsgerade vorbereiten
model = lr()  # lr = Lineare Regression

# Modell trainieren
# Das ".astype(np.float32)" kann ignoriert werden, es gibt einen Bug in sklearn, darum ist das nötig
model.fit(data_x.astype(np.float32), data_y)  # data_x = Trainingsdaten, data_y = Zieldaten

# Daten vorhersagen (für die Regressionsgerade)
y_pred = model.predict(data_x)

# Regressionsgerade einzeichnen
# Mit "b-" erhalten wir eine blaue Linie
plt.plot(data_x, y_pred, 'b-')

# Plot anzeigen
plt.show()
