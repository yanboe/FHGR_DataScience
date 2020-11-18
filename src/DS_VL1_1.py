import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("../data/smp_data.txt", delimiter=",")

print(data)

print(data[:, 0])
print(data[:, 1])
# print(data[0:3, :])

data_x = data[:, 0]
data_y = data[:, 1]


plt.xlabel("Freunde")
plt.ylabel("Zeit [Sekunden]")
plt.axis([0, 12, 0, 800])
plt.grid(True)
plt.plot(data_x, data_y, ls="none", marker=".")
plt.show()
