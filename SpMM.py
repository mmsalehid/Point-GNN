# ── SpMM COO / CSR / tf.gather comparison ─────────────────────────────────
    '''
    This first block is able to convert the graph into exact indexing structuures used in SpMM
    For example: src_idx uses tf.gather, dst_idx is used in scatter_max 
    COO SpMM then uses (src, dst)
    CSR SpMM uses the compressed edge structures 
    '''  
    graph_edge_list   = profile_edges #edge list over a frame (src, dst)
    num_of_vertices   = profile_vcoords.shape[0] #number of vertices in the graph
    num_of_edges   = graph_edge_list.shape[0] #number of edges in the graph
    src_idx = graph_edge_list[:, 0].astype(np.int32) 
    '''
    take all the rows of the edge list and take the first column (src) and convert to int32
    This src_idx is used for tf.gather, hence SpMM must produce the same output as tf.gather(X, src_idx)
    Y[e] = X[src_idx[e]] for all edges
    '''
    dst_idx = graph_edge_list[:, 1].astype(np.int32)
    '''
    take all the rows of the edge list and take the second column (dst) and convert to int32
    This dst_idx is not used in this SpMM test, but are used in aggregation steps in GNN's like scatter_max
    Y[v] = max over edges e where dst_idx[e] == v of Y[e]
    This is another SpMM-style graph aggregation
    '''

# Creating a dummy feature matrix matching the GNN dimensions to contruct tf.gather matrix
    np.random.seed(42) #random numbers for testing
    feat_dim = 300 #feature dimension of the vertex features
    X = np.random.randn(num_of_vertices, feat_dim).astype(np.float32) #generate random vertex features
    '''
    From the above we have now created a sprase matrix (E x V) where each row corresponds to an edge and has a single 1 in the column corresponding to the source vertex of that edge
    Y = Sparse_Matrix * X 
    '''

    #Build sparse matrices for SpMM
    data  = np.ones(num_of_edges, dtype=np.float32) #creates [1,1,1,1,1,..], they are all ones because each row of the sparse matrix should select exactly one source vertex
    rows  = np.arange(num_of_edges, dtype=np.int32) #creates [0,1,2,3,...], each row corresponds to an edge, which is one output row: Y[e]

    # COO
    sparse_coo = coo_matrix((data, (rows, src_idx)),shape=(num_of_edges, num_of_vertices)) #creates a COO sparse matrix 
    '''
    COO stores sparse matrices as a list of (row, col, value) 
    '''

    # CSR
    sparse_csr = sparse_coo.tocsr() #convert COO to CSR format
    '''
    CSR uses three arrays: indptr, indices, data
    '''

    # tf.gather
    t_X     = tf.constant(X)
    t_src   = tf.constant(src_idx)

    # Warm up TF gather
    with tf.Session() as sess_spmm:
        for _ in range(3):
            sess_spmm.run(tf.gather(t_X, t_src))

        N_RUNS = 50
        gather_times = []
        for _ in range(N_RUNS):
            t0 = time.perf_counter()
            Y_gather = sess_spmm.run(tf.gather(t_X, t_src)) #TensorFlow GatherV2 kernel
            t1 = time.perf_counter()
            gather_times.append((t1-t0)*1000)
        t_gather_mean = np.mean(gather_times)
        t_gather_std  = np.std(gather_times)

  #SpMM COO 
    def spmm_coo(coo_matrix, X):
        """
        SpMM using COO format.
        Y[e] = sum over nonzeros in row e of A[e,j] * X[j]
        For A_sel with one nonzero per row:
        Y[e] = A[e, src[e]] * X[src[e]] = X[src[e]]
        """
        Y = np.zeros((coo_matrix.shape[0], X.shape[1]),dtype=np.float32)
        # COO: iterate over all nonzeros
        # This is O(n) = O(E) — same as gather
        for i in range(len(coo_matrix.data)):
            row = coo_matrix.row[i]
            col = coo_matrix.col[i]
            Y[row] += coo_matrix.data[i] * X[col]
        return Y

#Warm Up
    for _ in range(3):
        _ = spmm_coo(sparse_coo, X)

    coo_times = []

    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        Y_coo = spmm_coo(sparse_coo, X)
        t1 = time.perf_counter()
        coo_times.append((t1 - t0) * 1000)

#SpMM CSR
    def spmm_csr(csr_matrix, X):
        """
        SpMM using CSR format.
        For row i: Y[i] = sum_{j in indptr[i]:indptr[i+1]} data[j] * X[indices[j]]
        CSR key advantage: indptr gives O(1) access to each row's nonzeros
        For A_sel: each row has exactly 1 nonzero so this equals X[src_idx]
        """
        Y = np.zeros((csr_matrix.shape[0], X.shape[1]),dtype=np.float32)

        for i in range(csr_matrix.shape[0]): # loop over rows 
            row_start = csr_matrix.indptr[i] 
            row_end   = csr_matrix.indptr[i+1]
            for j in range(row_start, row_end):
                col = csr_matrix.indices[j]
                Y[i] += csr_matrix.data[j] * X[col]
        return Y


# Warm up
    _ = spmm_csr(sparse_csr, X)
    csr_times = []

    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        Y_csr = spmm_csr(sparse_csr, X)
        t1 = time.perf_counter()
        csr_times.append((t1 - t0) * 1000)

    t_csr_mean = np.mean(csr_times)
    t_csr_std  = np.std(csr_times)

#Verifying Operations Are the Same
    print(f"\n  Numerical verification (all three must match):")
    coo_match = np.allclose(Y_gather, Y_coo,  atol=1e-4)
    csr_match = np.allclose(Y_gather, Y_csr,  atol=1e-4)
    print(f"    tf.gather == SpMM COO:  "
          f"{'✅ MATCH' if coo_match else '❌ MISMATCH'}")
    print(f"    tf.gather == SpMM CSR:  "
          f"{'✅ MATCH' if csr_match else '❌ MISMATCH'}")