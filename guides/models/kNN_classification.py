# Unterscheidung Classification / Regression:
# Regression and classification are both related to prediction, # where regression predicts a value from a
# continuous set, whereas classification predicts the 'belonging' to the class.
#
# For example, the price of a house depending on the 'size' (in some unit) and say 'location' of the house,
# can be some 'numerical value' (which can be continuous): this relates to regression.
#
# Similarly, the prediction of price can be in words, e.g.
# 'very expensive', 'expensive', 'affordable', 'cheap', and 'very cheap': this relates to classification.

from sklearn.neighbors import KNeighborsClassifier as Knn
import numpy as np
from sklearn.model_selection import train_test_split as split
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import accuracy_score

# Datensatz einlesen und formatieren
data = np.loadtxt("../../data/credit.txt", delimiter=",")
x_train, x_test, y_train, y_test = split(data[:, :2], data[:, 2], test_size=0.2)  # 20% Testdaten

# Modell trainieren
model = Knn(5)  # kNN aufsetzen (Classification)
model.fit(x_train, y_train)  # Trainieren

# y vorhersagen (mit Testdaten)
y_pred = model.predict(x_test)

# y vorhersagen (mit Trainingsdaten)
y_pred_train = model.predict(x_train)

# Fehler bestimmen
print("MSE (Test)", mse(y_test, y_pred))  # Mean Squared Error (Testdaten)
print("MSE (Training): ", mse(y_train, y_pred_train))  # Mean Squared Error (Trainingsdaten)
print("ACC Score (Test)", accuracy_score(y_test, y_pred))  # Accuracy (Testdaten)
print("ACC Score (Training): ", accuracy_score(y_train, y_pred_train))  # Accuracy (Trainingsdaten)

# Optimalen k-Wert suchen
a = 0
b = 0
for k in range(1, 16):  # Wir probieren nur von 1-16
    model = Knn(k)  # kNN aufsetzen (Classification)
    model.fit(x_train, y_train)  # Trainieren
    y_pred = model.predict(x_test)  # y vorhersagen (mit Testdaten)
    print("k = " + str(k) + ":\t" + str(accuracy_score(y_test, y_pred)))  # Accuracy ausgeben
    if accuracy_score(y_test, y_pred) > a:  # Wir merken uns den höchsten Accuracy-Score
        a = accuracy_score(y_test, y_pred)
        b = k

# Mit dem optimalen k-Wert trainieren wir nun unser Modell nochmal
print(b)
model = Knn(b)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print("MSE (Test)", mse(y_test, y_pred))  # Mean Squared Error (Testdaten)
print("ACC Score (Test)", accuracy_score(y_test, y_pred))  # Accuracy (Testdaten)
