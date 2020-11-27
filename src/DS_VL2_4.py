from sklearn.neighbors import KNeighborsRegressor as knr
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as split
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score

# -----------------------------------------
# Vorlesung 2, Slide 17, k-Nächste-Nachbarn
# -----------------------------------------

# Load data
data = np.loadtxt("../data/smp_data.txt", delimiter=",")

# Split data and reshape x axis
x_train, x_test, y_train, y_test = split(data[:, 0].reshape((-1, 1)), data[:, 1], test_size=0.2)

# Train model with knr=5
model = knr(5)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

# Print Mean Squared Error
print("MSE with knr=5", mse(y_test, y_pred))

# Print R2 Score
print("R2 with knr=5", r2_score(y_test, y_pred))

# Find optimal k-value
a = 0
b = 0
for k in range(1, 16):
    model = knr(n_neighbors=k)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print(k, "->", r2_score(y_test, y_pred))
    if r2_score(y_test, y_pred) > a:
        a = r2_score(y_test, y_pred)
        b = k

# Train model with optimal k-value (for the selected data... which changes each time
# you run this program as the split function shuffles the data before splitting)
model = knr(n_neighbors=b)
model.fit(x_train, y_train)
x = np.linspace(0, 12, 200)  # We create 200 x-values between 0 and 12 to test our model
y_pred = model.predict(x.reshape((-1, 1)))

# Now we plot it...
plt.plot(data[:, 0], data[:, 1], 'ro')  # whole dataset (test + train)
plt.plot(x, y_pred, 'b-')  # make the blue line with the 200 x-values between 0 and 12
plt.axis([0, 12, 0, 800])
plt.show()
