import numpy as np  # Package numpy importieren

# ----------------------------------------------------
# Guide zum Formatieren von Dateiinhalten
# ----------------------------------------------------

# ------------------------------------------------------------------------------
# Teil 1: Textdatei mit zwei Spalten einlesen, als ganzes und je Spalte ausgeben
# ------------------------------------------------------------------------------

# File einlesen (mehr Details in "read_files.py")
data = np.loadtxt("../data/smp_data.txt", delimiter=",")

# Inhalt unformatiert ausgeben
print("Rohdaten:\n", data)

# Wir erhalten folgenden Output:
"""
[
 [  3.  80.]
 [  1.  35.]
 [  6. 392.]
 [  9. 515.]
 ...
]
"""

# Man erkennt, dass das File aus zwei Spalten besteht. In der ersten Spalte haben wir 3, 1, 6, 9, etc.
# In der zweiten Spalte haben wir 80, 35, 392, 515, etc.
# Die Punkte (z.Bsp. "3.", "80.") können wir ignorieren - die sind von Pyhton eingefügt worden, um uns zu zeigen,
# dass es sich um Zahlen handelt.

# Wir können auch nur die erste Spalte ausgeben:
print("Erste Spalte:\n", data[:, 0])

# Das sieht dann so aus:
"""
[ 3.  1.  6.  9. 11.  7.  5.  8.  2.  5.  7. 10.  6.  4.  7.  8.  2. 10. ...]
"""

# Oder die zweite Spalte:
print("Zweite Spalte:\n", data[:, 1])

# Das sieht dann so aus:
"""
[ 80.  35. 392. 515. 742. 401. 266. 390.  69. 157. 339. 486. 182. 234. ...]
"""

# ------------------------------------------------------------------------------
# Teil 2: Reshape
# ------------------------------------------------------------------------------

# In Teil 1 haben wir die erste und zweite Spalte separat ausgegeben.
# Der Output sah so aus: [ 3.  1.  6.  9. 11. ...]
# Die einzelnen Werte sind alle nacheinander aufgeführt und nur durch ein Space getrennt.
# Nun gibt es aber bestimmte Funktionen, die mit diesem Format nichts anfangen können. Wir müssen
# den Output also umformatieren. Das lässt sich mit der Funktion "reshape" machen:
print("Erste Spalte mit Reshape:\n", data[:, 0].reshape((-1, 1)))

# Das sieht dann so aus:
"""
Mit Reshape:
[
 [ 3.]
 [ 1.]
 [ 6.]
 [ 9.]
 [11.]
 ...
]

Ohne Reshape:
[ 3.  1.  6.  9. 11. ...]
"""
