from tensorly.decomposition import parafac
from mytensor import gen_tensor_one_feature as gen1f
from mytensor import plot_uvw_one_feature as plot1f
from mytensor import gen_tensor_three_feature as gen3f
from mytensor import plot_uvw_three_feature as plot3f

# ----------------------------------------------------
# Vorlesung 4, Folie 27 + 29
# ----------------------------------------------------

# Beispiel zur Analyse mehrdimensionaler Daten (1)
X = gen1f()
w_, fac = parafac(X, 1)
plot1f(fac)

# Beispiel zur Analyse mehrdimensionaler Daten (2)
X = gen3f()
w_, fac = parafac(X, 3)
plot3f(fac)
