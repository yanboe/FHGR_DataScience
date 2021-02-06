# Unterscheidung Classification / Regression:
# Regression and classification are both related to prediction, # where regression predicts a value from a
# continuous set, whereas classification predicts the 'belonging' to the class.
#
# For example, the price of a house depending on the 'size' (in some unit) and say 'location' of the house,
# can be some 'numerical value' (which can be continuous): this relates to regression.
#
# Similarly, the prediction of price can be in words, e.g.
# 'very expensive', 'expensive', 'affordable', 'cheap', and 'very cheap': this relates to classification.

from sklearn.neighbors import KNeighborsRegressor as Knr
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as split
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score

# Datensatz einlesen und formatieren
data = np.loadtxt("../../data/smp_data.txt", delimiter=",")
x_train, x_test, y_train, y_test = split(data[:, 0].reshape((-1, 1)), data[:, 1], test_size=0.2)  # 20% Testdaten

# Modell trainieren
model = Knr(5)  # kNN aufsetzen (Regression)
model.fit(x_train, y_train)  # Trainieren

# y vorhersagen (mit Testdaten)
y_pred = model.predict(x_test)

# y vorhersagen (mit Trainingsdaten)
y_pred_train = model.predict(x_train)

# Fehler bestimmen
print("MSE (Test)", mse(y_test, y_pred))  # Mean Squared Error (Testdaten)
print("MSE (Training): ", mse(y_train, y_pred_train))  # Mean Squared Error (Trainingsdaten)
print("R2 (Test)", r2_score(y_test, y_pred))  # Bestimmtheitsmass R2 (Testdaten)
print("R2 (Training): ", r2_score(y_train, y_pred_train))  # Bestimmtheitsmass R2 (Trainingsdaten)

# Optimalen k-Wert suchen
a = 0
b = 0
for k in range(1, 16):  # Wir probieren nur von 1-16
    model = Knr(k)  # kNN aufsetzen (Regression)
    model.fit(x_train, y_train)  # Trainieren
    y_pred = model.predict(x_test)  # y vorhersagen (mit Testdaten)
    print("k = " + str(k) + ":\t" + str(r2_score(y_test, y_pred)))  # Bestimmtheitsmass R2 ausgeben
    if r2_score(y_test, y_pred) > a:  # Wir merken uns den höchsten R2-Score
        a = r2_score(y_test, y_pred)
        b = k

# Mit dem optimalen k-Wert trainieren wir nun unser Modell nochmal
model = Knr(b)
model.fit(x_train, y_train)
x = np.linspace(0, 12, 200)  # Hier erstellen wir 200 x-Werte zwischen 0 und 12, um unser Modell zu testen
y_pred = model.predict(x.reshape((-1, 1)))  # y vorhersagen (mit den 200 x-Werten)

# Plot erstellen
plt.plot(data[:, 0], data[:, 1], 'ro')  # Ganzen Datensatz plotten (Test- und Trainingsdaten)
plt.plot(x, y_pred, 'b-')  # Die 200 x-Werte als blaue Linie einzeichnen
plt.axis([0, 12, 0, 800])  # x- und y-Achse festlegen
plt.grid(True)
plt.show()
