import numpy as np
import tensorflow as tf


# Dense feature matrix
X = np.array([
    [1, 2],   # vertex 0
    [3, 4],   # vertex 1
    [5, 6],   # vertex 2
], dtype=np.float32)


# CSR representation of adjacency matrix
# indptr = row pointers
# indices = column indices
# values  = edge weights 

indptr = np.array([0, 1, 2, 3], dtype=np.int32)   # size V+1
indices = np.array([1, 0, 1], dtype=np.int32)     # column indices
values  = np.ones(len(indices), dtype=np.float32)


# TensorFlow gather
src = np.array([1, 0, 1], dtype=np.int32)
tf_out = np.take(X, src, axis=0)
print("TF-style output (gather equivalent):\n", tf_out)

Y = np.zeros((len(indptr) - 1, X.shape[1]), dtype=np.float32)

for row in range(len(indptr) - 1):
    for j in range(indptr[row], indptr[row + 1]):
        Y[row] += values[j] * X[indices[j]]

print("\nCSR SpMM output:\n", Y)

print("\nMatch:", np.allclose(tf_out, Y))

#we can see they output the same thing