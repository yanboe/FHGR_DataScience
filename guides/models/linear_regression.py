import numpy as np
from sklearn import datasets as ds
from sklearn.linear_model import LinearRegression as Lr
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score

# Datensatz einlesen und formatieren
x_diabetes, y_diabetes = ds.load_diabetes(return_X_y=True)
x_diabetes = x_diabetes[:, np.newaxis, 2]

# Trainingsdaten (80%)
x_train = x_diabetes[: -88]
y_train = y_diabetes[: -88]

# Testdaten (20%)
x_test = x_diabetes[-88:]
y_test = y_diabetes[-88:]

# Modell trainieren
model = Lr()  # Lineare Regression aufsetzen
model.fit(x_train, y_train)  # Trainieren

# y vorhersagen (mit Testdaten)
y_pred_test = model.predict(x_test)

# y vorhersagen (mit Trainingsdaten)
y_pred_train = model.predict(x_train)

# Plot erstellen
plt.plot(x_test, y_test, ls="none", marker="o")  # Testdaten (Kreise, 20%)
plt.plot(x_train, y_train, ls="none", marker="s")  # Trainingsdaten (Quadrate, 80%)
plt.plot(x_test, y_pred_test, 'b-')  # Regressionsgerade

# Fehler bestimmen
print("MSE (Test): ", mse(y_test, y_pred_test))  # Mean Squared Error (Testdaten)
print("MSE (Training): ", mse(y_train, y_pred_train))  # Mean Squared Error (Trainingsdaten)
print("R2 (Test): ", r2_score(y_test, y_pred_test))  # Bestimmtheitsmass R2 (Testdaten)
print("R2 (Training): ", r2_score(y_train, y_pred_train))  # Bestimmtheitsmass R2 (Trainingsdaten)

# Plot anzeigen
plt.grid(True)
plt.show()
