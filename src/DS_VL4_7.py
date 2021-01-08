from tensorly.decomposition import tucker
import tensorly as tl

# ----------------------------------------------------
# Vorlesung 4, Folie 31
# ----------------------------------------------------

# Tucker-Zerlegung
G, factors = tucker(X, (rank1, rank2, ..., rankN))

# Tensor rekonstruieren
X_rec = tl.tucker_to_tensor((G, factors))
tl.norm(X - X_rec)
