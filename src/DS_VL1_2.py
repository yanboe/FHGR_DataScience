import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as lr

# Daten einlesen und aufbereiten
data = np.loadtxt("../data/smp_data.txt", delimiter=",")
data_x = data[:, 0].reshape((-1, 1))
data_y = data[:, 1]

# Plot vorbereiten
plt.xlabel("Freunde")
plt.ylabel("Zeit [Sekunden]")
plt.axis([0, 12, 0, 800])
plt.grid(True)

# Punkte einzeichnen
plt.plot(data_x, data_y, ls="none", marker=".")

# Regressionsgerade vorbereiten
model = lr()
model.fit(data_x, data_y)
R2_score = model.score(data_x, data_y)
beta0 = model.intercept_
beta1 = model.coef_
pred_y = model.predict(data_x)

# Regressionsgerade einzeichnen
plt.plot(data_x, pred_y, 'b-')

# Plot anzeigen
plt.show()
