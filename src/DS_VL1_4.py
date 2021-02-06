import numpy as np
from sklearn import datasets
from sklearn.linear_model import LinearRegression as lr
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score

# Datensatz einlesen
diabetes_x, diabetes_y = datasets.load_diabetes(return_X_y=True)
diabetes_x = diabetes_x[:, np.newaxis, 2]

# Trainingsdaten definieren
# Damit trainieren wir unser Modell
x_train = diabetes_x[: -88]
y_train = diabetes_y[: -88]

# Testdaten definieren
# Die brauchen wir als Referenz, um zu schauen, wie genau
# unser trainiertes Modell ist
x_test = diabetes_x[-88:]
y_test = diabetes_y[-88:]

# Modell mit Testdaten trainieren (für R2 Test)
model = lr()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

# Modell mit Trainingsdaten trainieren (für R2 Train)
# So können wir schauen, ob der R2 Score bei Trainings-
# und Testdaten ähnlich sind
y_pred_train = model.predict(x_train)

# Testdaten einzeichnen (Kreise, 20%)
plt.plot(x_test, y_test, ls="none", marker="o")

# Trainingsdaten einzeichnen (Quadrate, 80%)
plt.plot(x_train, y_train, ls="none", marker="s")

# Regressionsgerade einzeichnen
plt.plot(x_test, y_pred, 'b-')

# MSE + R2 Score ausgeben
print("MSE: ", mse(y_test, y_pred))
print("R2 Score (Test): ", r2_score(y_test, y_pred))
print("R2 Score (Training): ", r2_score(y_train, y_pred_train))

# Plot anzeigen
plt.grid(True)
plt.show()
