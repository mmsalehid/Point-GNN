import numpy as np
import tensorflow as tf


# Dense feature matrix
X = np.array([
    [1, 2],   # vertex 0
    [3, 4],   # vertex 1
    [5, 6],   # vertex 2
], dtype=np.float32)


# COO edge list
# edges[e] = (src, dst)
edges = np.array([
    [0, 1],
    [2, 1],
    [1, 0],
], dtype=np.int32)

src = edges[:, 0]


# TensorFlow gather
tf_out = tf.gather(X, src).numpy()
print(f"tf.gather output: {tf_out}")


# COO format SpMM 
# A_sel * X
# A_sel is a sparse matrix where the rows are edges and colums are vertices
# A_sel[e, v] = 1 if edge connects to vertex, else 0
# ---------------------------------------------------
row_idx = np.arange(len(src))   # edge rows
col_idx = src                   # selected vertices
values  = np.ones(len(src))

Y = np.zeros((len(src), X.shape[1]), dtype=np.float32)

for i in range(len(row_idx)):
    Y[row_idx[i]] += values[i] * X[col_idx[i]]

print(f"\nCOO SpMM output: {Y}")
