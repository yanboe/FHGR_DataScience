import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as lr
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score

# Daten einlesen und aufbereiten
data = np.loadtxt("../data/smp_data.txt", delimiter=",")
x_data = data[:, 0].reshape((-1, 1))
y_data = data[:, 1]

# Trainingsdaten definieren
# Damit trainieren wir unser Modell
x_train = x_data[: -10]
y_train = y_data[: -10]

# Testdaten definieren
# Die brauchen wir als Referenz, um zu schauen, wie genau
# unser trainiertes Modell ist
x_test = x_data[-10:]
y_test = y_data[-10:]

# Plot vorbereiten
plt.xlabel("Freunde")
plt.ylabel("Zeit [Sekunden]")
plt.axis([0, 12, 0, 800])
plt.grid(True)

# Modell trainieren
model = lr()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

# Testdaten einzeichnen (Kreise, 80%)
plt.plot(x_test, y_test, ls="none", marker="o")

# Trainingsdaten einzeichnen (Quadrate, 20%)
plt.plot(x_train, y_train, ls="none", marker="s")

# Regressionsgerade einzeichnen
plt.plot(x_test, y_pred, 'b-')

print("MSE: ", mse(y_test, y_pred))
print("R2 Score: ", r2_score(y_test, y_pred))

# Plot anzeigen
plt.show()
