import numpy as np
import math
from scipy.spatial.distance import cdist

def Row(matrix):
    M = matrix.astype(float).copy()
    row_sums = np.sum(M, axis=1)
    non_zero_rows = row_sums != 0
    M[non_zero_rows, :] /= row_sums[non_zero_rows, np.newaxis]

    return M

def Col(matrix):
    W = matrix.astype(float).copy()
    col_sums = np.sum(W, axis=0)
    non_zero_cols = col_sums != 0
    W[:, non_zero_cols] /= col_sums[non_zero_cols]

    return W

def generate_exp_matrices(n, seed=42):

    def generate_exponential_weight_matrix(n):
        denominator = np.abs(np.log2(n)) + 1
        i_indices, j_indices = np.indices((n, n))
        mod_vals = (j_indices - i_indices) % n
        is_power_of_two = (mod_vals != 0) & ((mod_vals & (mod_vals - 1)) == 0)
        mask = (i_indices == j_indices) | is_power_of_two
        weight_matrix = np.where(mask, 1.0 / denominator, 0.0)
        return weight_matrix
    
    original_matrix = generate_exponential_weight_matrix(n)
    np.random.seed(seed)
    random_matrix = np.where(original_matrix != 0, np.random.randint(1, 3, size=original_matrix.shape), 0)
    random_matrix = np.array(random_matrix)
    M = random_matrix.copy().astype(float)
    A = Row(M)
    B = Col(M)
    return A, B

def generate_grid_matrices(n, seed=42):
    grid_size_f = math.sqrt(n)
    grid_size = int(grid_size_f)
    if grid_size * grid_size != n:
        print(f"Error: n ({n}) must be a perfect square.")
        return None
    N = np.zeros((n, n), dtype=int)

    for r in range(grid_size):
        for c in range(grid_size):
            idx = r * grid_size + c
            N[idx, idx] = 1
            if c + 1 < grid_size:
                neighbor_idx = r * grid_size + (c + 1)
                N[idx, neighbor_idx] = 1
                N[neighbor_idx, idx] = 1

            if r + 1 < grid_size:
                neighbor_idx = (r + 1) * grid_size + c
                N[idx, neighbor_idx] = 1
                N[neighbor_idx, idx] = 1

    np.random.seed(seed)
    W = np.zeros((n, n), dtype=float)
    rows, cols = np.where(N == 1)
    random_weights = np.random.randint(1, 3, size=len(rows))
    W[rows, cols] = random_weights
    A = Row(W)
    B = Col(W)

    return A, B

def generate_ring_matrices(n, seed=42):
    if n < 2:
        print(f"Error: n ({n}) must be >= 2.")
        return None

    N = np.zeros((n, n), dtype=int)
    for i in range(n):
        N[i, i] = 1 
        N[i, (i + 1) % n] = 1 
    idx_n_4 = int(n / 4)
    idx_n_2 = int(n / 2)
    idx_3n_4 = int(3 * n / 4)
    if 0 <= idx_n_4 < n:
        N[0, idx_n_4] = 1
    if 0 <= idx_n_2 < n and 0 <= idx_3n_4 < n:
        N[idx_n_2, idx_3n_4] = 1
    np.random.seed(seed)
    W = np.zeros((n, n), dtype=float)
    rows, cols = np.where(N == 1)
    random_weights = np.random.randint(1, 3, size=len(rows))
    W[rows, cols] = random_weights
    A = Row(W)
    B = Col(W)

    return A, B

def generate_random_graph_matrices(n: int, seed: int = 42):
    np.random.seed(seed)
    N = np.zeros((n, n), dtype=int)
    prob_edge_exists = 1/3

    for i in range(n):
        N[i, i] = 1

    for i in range(n):
        for j in range(i + 1, n): # j > i
            if np.random.rand() < prob_edge_exists:
                N[i, j] = 1
                N[j, i] = 1

    W = np.zeros((n, n), dtype=float)

    rows, cols = np.where(N == 1)
    
    random_weights = np.random.randint(1, 3 + 1, size=len(rows))
    W[rows, cols] = random_weights

    A = Row(W)
    B = Col(W)

    return A, B

def generate_stochastic_geometric_matrices(n, seed, threshold=5):

    if not isinstance(n, int) or n < 6:
        raise ValueError("n must be an integer greater than or equal to 6.")
    if not isinstance(seed, int):
        raise ValueError("seed must be an integer.")

    np.random.seed(seed)

    positions = np.random.rand(n, 2) * 10 
    W = np.zeros((n, n))
    np.fill_diagonal(W, 1)
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist <= threshold:
                W[i, j] = 1
                W[j, i] = 1 
    one_indices = np.where(W == 1)
    random_values = np.random.choice([1, 2, 3], size=len(one_indices[0]))
    W[one_indices] = random_values

    A = Row(W)
    B = Col(W)

    return A, B

def generate_nearest_neighbor_matrices(n: int, k: int = 3, seed: int = 42):
    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be an integer greater than or equal to 1.")
    if not isinstance(k, int) or k < 0:
        raise ValueError("k must be a non-negative integer.")
    if n == 1 and k != 0:
        raise ValueError("if n == 1, k must be 0 (no neighbors).")
    if n > 1 and k >= n:
        raise ValueError("k must be less than n (cannot connect to all nodes including self).")
    if not isinstance(seed, int):
        raise ValueError("seed must be an integer.")

    np.random.seed(seed)
    positions = np.random.rand(n, 2) * 10

    W = np.zeros((n, n), dtype=float)
    np.fill_diagonal(W, 1)

    if n > 1 and k > 0:
        all_distances = cdist(positions, positions)

        for i in range(n):
            distances_from_i = all_distances[i, :]
            sorted_neighbor_indices = np.argsort(distances_from_i)
            for neighbor_idx in sorted_neighbor_indices[1 : k + 1]:
                W[i, neighbor_idx] = 1
                W[neighbor_idx, i] = 1
    rows, cols = np.where(W == 1)
    num_edges_to_weight = len(rows)
    
    if num_edges_to_weight > 0:
        random_weights = np.random.randint(1, 3 + 1, size=num_edges_to_weight)
        W[rows, cols] = random_weights
    A = Row(W)
    B = Col(W)

    return A, B