from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.model_selection import train_test_split as split
from sklearn.metrics import accuracy_score
import numpy as np

# -----------------------------------------
# Vorlesung 2, k-Nächste-Nachbarn
# Kombinierter Code aus DS_VL2_2 + DS_VL2_3
# -----------------------------------------

# Read credit.txt
data = np.loadtxt("../data/credit.txt", delimiter=",")

# Create training and test data (80% training, 20% test)
x_train, x_test, y_train, y_test = split(data[:, :2], data[:, 2], test_size=0.2)

# Find highest k value and use it for model training
x = 0
y = 0
for k in range(1, 16):
    model = knn(n_neighbors=k)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print(k, "->", accuracy_score(y_test, y_pred))
    if accuracy_score(y_test, y_pred) > x:
        x = accuracy_score(y_test, y_pred)
        y = k

print("Grösstes k: ", y, "mit Accuracy Score ", x)

# Train model
model = knn(n_neighbors=y)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

# Print accuracy score
print("Accuracy score: ", accuracy_score(y_test, y_pred))
