import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as lr
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score

# Daten einlesen und aufbereiten
data = np.loadtxt("../data/smp_data.txt", delimiter=",")
data_x = data[:, 0].reshape((-1, 1))
data_y = data[:, 1]
train_x = data_x[: -10]
test_x = data_x[-10:]
train_y = data_y[: -10]
test_y = data_y[-10:]

# Plot vorbereiten
plt.xlabel("Freunde")
plt.ylabel("Zeit [Sekunden]")
plt.axis([0, 12, 0, 800])
plt.grid(True)

# Punkte einzeichnen
plt.plot(test_x, test_y, ls="none", marker=".")
plt.plot(train_x, train_y, ls="none", marker="s")

# Regressionsgerade vorbereiten
model = lr()
model.fit(train_x, train_y)
R2_score = model.score(test_x, test_y)
pred_y = model.predict(test_x)

# Regressionsgerade einzeichnen
plt.plot(test_x, pred_y, 'b-')

mse(test_y, pred_y)
r2_score(test_y, pred_y)

# Plot anzeigen
plt.show()
