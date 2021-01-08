import numpy as np  # Package numpy (mit "as np" erstellen wir eine Abkürzung, um nicht immer numpy schreiben zu müssen)

# ----------------------------------------------------
# Guide zum Einlesen von Files
# ----------------------------------------------------

# Textdatei (.txt) einlesen, deren Inhalt kommasepariert ist (delimiter=",")
# Wir verwenden dazu loadtxt aus dem Package "numpy" (dessen Namen wir mit np abkürzen)
data = np.loadtxt("../data/smp_data.txt", delimiter=",")

# Nach dem Einlesen ist der Inhalt des Files in der Variable "data" gepseichert. Wir können den Inhalt anschauen:
print(data)
