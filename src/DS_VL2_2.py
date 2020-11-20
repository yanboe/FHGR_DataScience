from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.model_selection import train_test_split as split
from sklearn.metrics import accuracy_score
import numpy as np

# -----------------------------------------
# Vorlesung 2, Slide 14, k-Nächste-Nachbarn
# -----------------------------------------

# Read credit.txt
data = np.loadtxt("../data/credit.txt", delimiter=",")

# Create training and test data (80% training, 20% test)
x_train, x_test, y_train, y_test = split(data[:, :2], data[:, 2], test_size=0.2)

# Train model
model = knn(n_neighbors=5)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

# Print accuracy score
print("Accuracy score: ", accuracy_score(y_test, y_pred))
