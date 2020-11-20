from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.model_selection import train_test_split as split
from sklearn.metrics import accuracy_score
import numpy as np

# -----------------------------------------
# Vorlesung 2, Slide 15, k-Nächste-Nachbarn
# -----------------------------------------

# Read credit.txt
data = np.loadtxt("../data/credit.txt", delimiter=",")

# Create training and test data (test_size=0.2 = 20% test data)
# Also set random_state to control the shuffling applied to the data to get reproducible output
x_train, x_test, y_train, y_test = split(data[:, :2], data[:, 2], test_size=0.2, random_state=0)

# Find optimal k-value
for k in range(1, 16):
    model = knn(n_neighbors=k)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print(k, "->", accuracy_score(y_test, y_pred))
