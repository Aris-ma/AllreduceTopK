import cupy as cp
import numpy as np
import pandas as pd
import wandb

def grad_with_batch_batched_gpu(
    x_batched,      # Shape: (num_runs, n, d)
    y_nodes_gpu,    # Shape: (n, L) - global y_tilde on GPU
    h_nodes_gpu,    # Shape: (n, L, d) - global h_tilde on GPU
    rho,
    batch_size,
    num_runs        # Number of parallel simulation runs
):
    """
    Calculates the gradient for a batch of data on the GPU, for multiple parallel runs.

    Args:
        x_batched (cp.ndarray): Batched parameters for each node and run. Shape: (num_runs, n, d).
        y_nodes_gpu (cp.ndarray): Labels for each node, on GPU. Shape: (n, L).
        h_nodes_gpu (cp.ndarray): Input data for each node, on GPU. Shape: (n, L, d).
        rho (float): Regularization parameter.
        batch_size (int or None): The size of the mini-batch for gradient calculation. If None, use the full dataset.
        num_runs (int): The number of parallel runs.

    Returns:
        cp.ndarray: The calculated gradients. Shape: (num_runs, n, d).
    """
    n_nodes, L_samples, d_dims = h_nodes_gpu.shape  # n, L, d from h_tilde

    if batch_size is None or batch_size >= L_samples:
        # Use the full batch
        batch_size_eff = L_samples
        # Expand h_nodes_gpu and y_nodes_gpu for num_runs using broadcasting
        # h_batch_gpu shape: (num_runs, n, L, d)
        # y_batch_gpu shape: (num_runs, n, L)
        h_batch_gpu = cp.broadcast_to(h_nodes_gpu[cp.newaxis, ...], (num_runs, n_nodes, L_samples, d_dims))
        y_batch_gpu = cp.broadcast_to(y_nodes_gpu[cp.newaxis, ...], (num_runs, n_nodes, L_samples))
    else:
        # Use a mini-batch
        batch_size_eff = batch_size
        # Sample indices for each run and each node independently
        # A practical way to get indices for shape (num_runs, n, batch_size_eff):
        all_indices = cp.random.rand(num_runs, n_nodes, L_samples).argsort(axis=-1)[:, :, :batch_size_eff]
        batch_indices = all_indices.astype(cp.int32)  # Ensure integer indices

        # Gather h_batch and y_batch using these indices
        # h_batch_gpu shape: (num_runs, n, batch_size_eff, d)
        # y_batch_gpu shape: (num_runs, n, batch_size_eff)
        
        # Create indices for gathering
        run_idx = cp.arange(num_runs)[:, cp.newaxis, cp.newaxis]  # Shape: (num_runs, 1, 1)
        node_idx = cp.arange(n_nodes)[cp.newaxis, :, cp.newaxis]    # Shape: (1, n, 1)
        
        # Advanced indexing to select mini-batches for all runs and nodes
        h_batch_gpu = h_nodes_gpu[node_idx, batch_indices, :]
        y_batch_gpu = y_nodes_gpu[node_idx, batch_indices]

    # x_batched has shape (num_runs, n, d)
    # h_batch_gpu has shape (num_runs, n, batch_size_eff, d)
    # y_batch_gpu has shape (num_runs, n, batch_size_eff)

    # einsum for h_dot_x: result shape (num_runs, n, batch_size_eff)
    # This computes the inner product <h, x> for each sample in the batch, for each node and run.
    h_dot_x = cp.einsum('rnbd,rnd->rnb', h_batch_gpu, x_batched)

    
    # Calculate the exponential term for the logistic loss gradient
    exp_val = cp.exp(y_batch_gpu * h_dot_x)
    # Clip to avoid overflow
    cp.clip(exp_val, a_min=None, a_max=1e300, out=exp_val)

    # einsum for g1 (gradient of logistic loss): result shape (num_runs, n, d)
    g1 = -cp.einsum('rnbd,rnb->rnd', h_batch_gpu, y_batch_gpu / (1 + exp_val)) / batch_size_eff
    
    # g2 is the gradient of the regularization term: rho * sum(x^2 / (1 + x^2))
    x_squared = x_batched**2
    g2 = 2 * x_batched / (1 + x_squared)**2
    
    # Combine the two parts of the gradient
    grad_val = (g1 + rho * g2)  # Shape: (num_runs, n, d)
    return grad_val



def loss_full_batched_gpu(x, y_nodes_gpu, h_nodes_gpu, rho):
    """
    Compute full logistic loss + nonlinear regularizer for all runs.
    
    Args:
        x: (runs, n, d), but we only use x[:,0,:] since they are identical across nodes
        y_nodes_gpu: (n, L, )
        h_nodes_gpu: (n, L, d)
        rho: reg parameter
        
    Returns:
        loss_per_run: (runs,)
    """
    import cupy as cp

    runs, n, d = x.shape
    L = y_nodes_gpu.shape[1]

    # x_mean: (runs, 1, d)
    x_mean = x[:, 0:1, :]  

    # logistic loss ----------------------------------------
    # compute y * (h x)
    # z shape: (runs, n, L)
    z = cp.einsum("rnd,nld->rnl", x_mean, h_nodes_gpu)  # (runs,n,L)
    y_z = y_nodes_gpu[None, :, :] * z  # broadcast to (runs,n,L)
    
    # log(1 + exp(- y*h*x))
    logistic_loss = cp.log1p(cp.exp(-y_z))

    # average over L_i per node, then average across nodes
    logistic_loss = cp.mean(logistic_loss, axis=(1,2))   # (runs,)

    # regularizer -----------------------------------------
    x_sq = x_mean[:,0,:] ** 2
    reg = rho * cp.sum(x_sq / (1 + x_sq), axis=1)         # (runs,)

    return logistic_loss + reg

def topk_gpu(g, k):
    """
    在 GPU 上按最后一维做 Top-K（按绝对值）。
    g: (num_runs, n, d)
    k: int
    返回 g_topk (同 shape)
    """
    if k <= 0:
        return cp.zeros_like(g)
    num_runs, n, d = g.shape
    k = int(min(max(1, k), d))  # 至少 1，最多 d

    abs_g = cp.abs(g)
    # argpartition 找到前 k 的位置（无序），取出最后 k 列
    topk_idx = cp.argpartition(abs_g, -k, axis=2)[:, :, -k:]  # shape (num_runs, n, k)得到的 topk_idx 仍然是原始的维度编号

    g_topk = cp.zeros_like(g)

    # 构造索引
    batch_idx = cp.arange(num_runs)[:, None, None]  # (num_runs,1,1)
    node_idx = cp.arange(n)[None, :, None]          # (1,n,1)
    # topk_idx shape: (num_runs, n, k)
    # 直接赋值
    g_topk[batch_idx, node_idx, topk_idx] = g[batch_idx, node_idx, topk_idx]

    return g_topk


def centralized_MSGD_batched_gpu(
    init_x_gpu_batched, # init_x shape: (num_runs, n, d)
    h_data_nodes_gpu, y_data_nodes_gpu, # Original h_tilde, y_tilde on GPU
    grad_func_batched_gpu, # The new batched gradient function for GPU
    rho, lr, sigma_n, eta,
    max_it, batch_size, num_runs,
    topk_ratio=None,
    use_ef=None
):
    """
    Implements centralized Momentum Stochastic Gradient Descent (MSGD), 
    batched for multiple runs on a GPU.

    Args:
        init_x_gpu_batched (cp.ndarray): Initial parameters. All nodes in a run start with the same parameters. Shape: (num_runs, n, d).
        h_data_nodes_gpu (cp.ndarray): Full input data on GPU. Shape: (n, L, d).
        y_data_nodes_gpu (cp.ndarray): Full labels on GPU. Shape: (n, L).
        grad_func_batched_gpu (function): The GPU-accelerated gradient function.
        rho (float): Regularization parameter.
        lr (float): Learning rate.
        sigma_n (float): Standard deviation of the noise added to the gradient.
        max_it (int): Maximum number of iterations.
        batch_size (int): Mini-batch size for gradient calculation.
        num_runs (int): Number of parallel simulation runs.
        beta (float): Momentum parameter.

    Returns:
        pd.DataFrame: A DataFrame containing the history of the average gradient norm.
    """
    wandb.init(
            project=f"numerial_d40_iter5000_hetero3", 
            name=f"topk"
        )
    
    loss_history = []

    x = cp.copy(init_x_gpu_batched)  # Parameters. Shape: (num_runs, n, d)
    num_n, num_d = x.shape[1], x.shape[2]  # Number of nodes (n) and dimensions (d)
    
    # Initialize momentum (velocity)
    # velocity = cp.zeros((num_runs, 1, num_d), dtype=x.dtype)

    # EF memory init
    if use_ef:
        e = cp.zeros_like(x)  # (num_runs, n, d)
    else:
        e = None

    h = cp.zeros_like(x)

    # Store average gradient norm over runs
    avg_gradient_norm_history = []

    least_loss = 1
    for iter_num in range(max_it):
        # Calculate gradients for all nodes based on their current parameters
        g = grad_func_batched_gpu(x, y_data_nodes_gpu, h_data_nodes_gpu, rho, batch_size, num_runs)
        if sigma_n > 0:
            # Add Gaussian noise to the gradient if specified
            g += sigma_n * cp.random.normal(size=(num_runs, num_n, num_d))

        # 如果启用 Top-K / EF21，则在这里对 g 做 EF21，然后用压缩结果 c 来聚合
        if topk_ratio is not None:
            k = int(max(1, num_d * topk_ratio))
            if use_ef:
                # # EF14：
                # v = g + e
                # c = topk_gpu(v, k)
                # e = v - c
                # g_for_agg = c
                # EF21：
                if iter_num ==0:
                    e = g
                    g_for_agg = e
                else:
                    h = (1-eta) * h + eta * g
                    v = h - e
                    c = topk_gpu(v, k)
                    e = e + c
                    g_for_agg = e

            else:
                # 仅 Top-K（无误差补偿）
                h = (1-eta) * h + eta * g
                g_for_agg = topk_gpu(h, k)
        else:
            # 无压缩
            h = (1-eta) * h + eta * g
            g_for_agg = h
    
        # Aggregate gradients by averaging across nodes for each run
        g_avg = cp.mean(g_for_agg, axis=1, keepdims=True)  # Shape: (num_runs, 1, d)

        # # Update momentum (velocity)
        # # m_{t+1} = beta * m_t + (1 - beta) * g_avg
        # velocity = beta * velocity + (1 - beta) * g_avg

        # # Update the parameters using momentum. The update is the same for all nodes in a run.
        # x_updated_slice = x[:, 0:1, :] - lr * velocity
        
        x_updated_slice = x[:, 0:1, :] - lr * g_avg

        # Broadcast the updated parameters to all nodes within each run
        x = cp.broadcast_to(x_updated_slice, (num_runs, num_n, num_d))

        # --- Record history (averaged over runs) ---
        # The parameters `x` are already the mean parameters (x_mean_per_run)
        # 1. Calculate full batch gradient for each run's x_mean: _grad_on_full_per_run shape (num_runs, n, d)
        #    Use batch_size=None for full dataset, no sigma_n noise for this evaluation
        _grad_on_full_per_run = grad_func_batched_gpu(
            x, y_data_nodes_gpu, h_data_nodes_gpu,
            rho=rho, batch_size=None, num_runs=num_runs
        )
        
        # 2. Calculate mean gradient over nodes for each run: mean_grad_per_run shape (num_runs, 1, d)
        mean_grad_per_run = cp.mean(_grad_on_full_per_run, axis=1, keepdims=True)
        
        # 3. Calculate norm of mean_grad for each run: norm_per_run shape (num_runs,)
        norm_per_run = cp.linalg.norm(mean_grad_per_run, axis=2).squeeze()
        
        # 4. Average these norms over all runs: avg_norm_over_runs (scalar)
        avg_norm_over_runs_scalar = cp.mean(norm_per_run)
        avg_gradient_norm_history.append(cp.asnumpy(avg_norm_over_runs_scalar))  # Store as numpy float

        # Compute full loss (for monitoring)
        loss_value = loss_full_batched_gpu(
           x, y_data_nodes_gpu, h_data_nodes_gpu, rho
        )
        l=cp.asnumpy(cp.mean(loss_value)).item()
        loss_history.append(l)
        if least_loss > l:
            least_loss = l
        
        if (iter_num + 1) % 10 == 0:  # Print progress
            wandb.log({
                "train_loss": l,
                "least_loss":least_loss,
            },step=iter_num+1)

            print(f"Iteration {iter_num+1}/{max_it}, Avg Grad Norm: {avg_norm_over_runs_scalar:.6f}, Least_Loss:{least_loss}")

    print("\n")
    return pd.DataFrame({
        "gradient_norm_on_full_trainset_avg": avg_gradient_norm_history,
        "loss_full_avg": loss_history,
    })

def randk_gpu(g, k):
    """
    RandK compressor for 3D tensor (num_runs, n, d).
    For each (run, node) independently select k coordinates at random (without replacement).
    Returns compressed tensor same shape as g.
    """
    num_runs, n, d = g.shape

    # ensure ints
    d = int(d)
    k = int(k)

    # if k >= d just return copy (no compression)
    if k >= d:
        return g.copy()

    # generate random numbers per (run, node, dim) and take top-k by random order:
    # idx shape: (num_runs, n, k)
    rnd = cp.random.rand(num_runs, n, d)
    idx = rnd.argsort(axis=-1)[:, :, :k]  # first k indices of the random permutation

    out = cp.zeros_like(g)

    # broadcast indices for advanced indexing
    run_idx = cp.arange(num_runs)[:, None, None]   # (num_runs,1,1)
    node_idx = cp.arange(n)[None, :, None]         # (1,n,1)

    out[run_idx, node_idx, idx] = g[run_idx, node_idx, idx]

    return out



def centralized_MSGD_batched_gpu_randk(
    init_x_gpu_batched, # init_x shape: (num_runs, n, d)
    h_data_nodes_gpu, y_data_nodes_gpu, # Original h_tilde, y_tilde on GPU
    grad_func_batched_gpu, # The new batched gradient function for GPU
    rho, lr, sigma_n,eta,
    max_it, batch_size, num_runs,
    topk_ratio=None,
    use_ef=None
):
    """
    Implements centralized Momentum Stochastic Gradient Descent (MSGD), 
    batched for multiple runs on a GPU.

    Args:
        init_x_gpu_batched (cp.ndarray): Initial parameters. All nodes in a run start with the same parameters. Shape: (num_runs, n, d).
        h_data_nodes_gpu (cp.ndarray): Full input data on GPU. Shape: (n, L, d).
        y_data_nodes_gpu (cp.ndarray): Full labels on GPU. Shape: (n, L).
        grad_func_batched_gpu (function): The GPU-accelerated gradient function.
        rho (float): Regularization parameter.
        lr (float): Learning rate.
        sigma_n (float): Standard deviation of the noise added to the gradient.
        max_it (int): Maximum number of iterations.
        batch_size (int): Mini-batch size for gradient calculation.
        num_runs (int): Number of parallel simulation runs.
        beta (float): Momentum parameter.

    Returns:
        pd.DataFrame: A DataFrame containing the history of the average gradient norm.
    """
    wandb.init(
            project=f"numerial_d40_iter5000_hetero3", 
            name=f"randk"
        )
    loss_history = []

    x = cp.copy(init_x_gpu_batched)  # Parameters. Shape: (num_runs, n, d)
    num_n, num_d = x.shape[1], x.shape[2]  # Number of nodes (n) and dimensions (d)
    
    # Initialize momentum (velocity)
    velocity = cp.zeros((num_runs, 1, num_d), dtype=x.dtype)

    # EF memory init
    if use_ef:
        e = cp.zeros_like(x)  # (num_runs, n, d)
        h = cp.zeros_like(x)
    else:
        e = None

    # Store average gradient norm over runs
    avg_gradient_norm_history = []
    least_loss = 1
    for iter_num in range(max_it):
        # Calculate gradients for all nodes based on their current parameters
        g = grad_func_batched_gpu(x, y_data_nodes_gpu, h_data_nodes_gpu, rho, batch_size, num_runs)
        if sigma_n > 0:
            # Add Gaussian noise to the gradient if specified
            g += sigma_n * cp.random.normal(size=(num_runs, num_n, num_d))

        # 如果启用 Top-K / EF21，则在这里对 g 做 EF21，然后用压缩结果 c 来聚合
        if topk_ratio is not None:
            k = int(max(1, num_d * topk_ratio))
            # print(k)
            if use_ef:
                # # EF14：v = g + e; c = TopK(v); e = v - c; 使用 c 聚合
                # v = g + e
                # c = topk_gpu(v, k)
                # e = v - c
                # g_for_agg = c
                # EF21：v = g - e; c = TopK(v); e = e + c
                if iter_num ==0:
                    e = g
                    g_for_agg = e
                else:
                    h = (1-eta) * h + eta * g
                    v = h - e
                    c = randk_gpu(v, k)
                    e = e + c
                    g_for_agg = e

            else:
                # 仅 Top-K（无误差补偿）
                g_for_agg = randk_gpu(v, k)
        else:
            # 无压缩
            g_for_agg = g
    
        # Aggregate gradients by averaging across nodes for each run
        g_avg = cp.mean(g_for_agg, axis=1, keepdims=True)  # Shape: (num_runs, 1, d)
        
        # # Update momentum (velocity)
        # # m_{t+1} = beta * m_t + (1 - beta) * g_avg
        # velocity = beta * velocity + (1 - beta) * g_avg

        # # Update the parameters using momentum. The update is the same for all nodes in a run.
        # x_updated_slice = x[:, 0:1, :] - lr * velocity
        
        x_updated_slice = x[:, 0:1, :] - lr * g_avg
        
        # Broadcast the updated parameters to all nodes within each run
        x = cp.broadcast_to(x_updated_slice, (num_runs, num_n, num_d))

        # --- Record history (averaged over runs) ---
        # The parameters `x` are already the mean parameters (x_mean_per_run)
        # 1. Calculate full batch gradient for each run's x_mean: _grad_on_full_per_run shape (num_runs, n, d)
        #    Use batch_size=None for full dataset, no sigma_n noise for this evaluation
        _grad_on_full_per_run = grad_func_batched_gpu(
            x, y_data_nodes_gpu, h_data_nodes_gpu,
            rho=rho, batch_size=None, num_runs=num_runs
        )
        
        # 2. Calculate mean gradient over nodes for each run: mean_grad_per_run shape (num_runs, 1, d)
        mean_grad_per_run = cp.mean(_grad_on_full_per_run, axis=1, keepdims=True)
        
        # 3. Calculate norm of mean_grad for each run: norm_per_run shape (num_runs,)
        norm_per_run = cp.linalg.norm(mean_grad_per_run, axis=2).squeeze()
        
        # 4. Average these norms over all runs: avg_norm_over_runs (scalar)
        avg_norm_over_runs_scalar = cp.mean(norm_per_run)
        avg_gradient_norm_history.append(cp.asnumpy(avg_norm_over_runs_scalar))  # Store as numpy float

        # === Compute full loss (for monitoring) ===
        loss_value = loss_full_batched_gpu(
           x, y_data_nodes_gpu, h_data_nodes_gpu, rho
        )
        l=cp.asnumpy(cp.mean(loss_value)).item()
        loss_history.append(l)
        if least_loss > l:
            least_loss = l
        if (iter_num + 1) % 10 == 0:  # Print progress
            wandb.log({
                "train_loss": l,
                "least_loss":least_loss,
            },step=iter_num+1)

            print(f"Iteration {iter_num+1}/{max_it}, Avg Grad Norm: {avg_norm_over_runs_scalar:.6f}, Least_Loss:{least_loss}")
    
    print("\n")
    return pd.DataFrame({
        "gradient_norm_on_full_trainset_avg": avg_gradient_norm_history,
        "loss_full_avg": loss_history,
    })

def arctopk_gpu(g, m, r, mu, seed=None):
    """
    ARC-Top-K compressor (GPU-batched).
    Inputs:
        g: cp.ndarray, shape (num_runs, num_nodes, d)
        m: int, number of rows to reshape into (must divide d)
        r: int, projection dimension
        mu: float in (0,1], compressor ratio; K = ceil(mu * m)
        seed: optional int, random seed to generate V (shared across nodes/runs)
    Returns:
        dict with keys:
          - local_comp: (num_runs, num_nodes, K * ncols)  # flattened selected rows per node
          - local_recon: (num_runs, num_nodes, d)         # reconstructed dense vector with zeros elsewhere
          - global_comp: (num_runs, K * ncols)            # average of local_comp over nodes
          - global_recon: (num_runs, d)                   # reconstructed dense average (same shape as g reduced)
          - topk_idx: (num_runs, K)                       # selected row indices per run
    Notes:
        Requires d % m == 0.
    """
    num_runs, num_nodes, d = g.shape
    if d % m != 0:
        raise ValueError(f"d must be divisible by m (got d={d}, m={m})")
    ncols = d // m
    K = int(cp.ceil(mu * m))
    K = max(1, min(K, m))

    # reshape g -> (num_runs, num_nodes, m, ncols)
    g_mat = g.reshape((num_runs, num_nodes, m, ncols))

    # # generate projection matrix V of shape (ncols, r)
    # if seed is None:
    #     rng = cp.random
    # else:
    #     rng = cp.random.RandomState(seed)
    # V = rng.normal(size=(ncols, r), dtype=g.dtype)  # (ncols, r)

    # # compute P_i = G_i @ V  -> shape (num_runs, num_nodes, m, r)
    # # einsum: 'rnmc,cr->r n m r'
    # P = cp.einsum('bnmc,cr->bnmr', g_mat, V)
    P = g_mat

    # All-Reduce (average across nodes) -> P_avg shape (num_runs, m, r)
    P_avg = cp.mean(P, axis=1)  # axis=1: average over nodes

    # Sigma_t: row-wise squared-norm of P_avg -> shape (num_runs, m)
    Sigma = cp.sum(P_avg * P_avg, axis=2)  # sum over r

    # select top-K rows per run
    # use argpartition for speed, then sort selected indices for stability
    # topk_idx shape: (num_runs, K)
    idx_part = cp.argpartition(Sigma, -K, axis=1)[:, -K:]  # unsorted K indices per run
    # optionally sort by descending Sigma
    # convert to gather sorted order
    # get values
    runs_idx = cp.arange(num_runs)[:, None]
    idx_vals = Sigma[runs_idx, idx_part]  # (num_runs, K)
    order = cp.argsort(-idx_vals, axis=1)
    topk_idx = idx_part[runs_idx, order]  # (num_runs, K)

    # Now gather selected rows from g_mat:
    # we want selected shape (num_runs, num_nodes, K, ncols)
    run_idx = cp.arange(num_runs)[:, None, None]    # (num_runs,1,1)
    node_idx = cp.arange(num_nodes)[None, :, None]  # (1,num_nodes,1)
    row_idx = topk_idx[:, None, :]                   # (num_runs,1,K)
    selected = g_mat[run_idx, node_idx, row_idx, :]  # (num_runs,num_nodes,K,ncols)

    # flatten selected rows -> local_comp (num_runs,num_nodes, K*ncols)
    local_comp = selected.reshape(num_runs, num_nodes, K * ncols)

    # reconstruct to full dense per-node vector: create zeros and scatter back
    local_recon = cp.zeros_like(g)  # (num_runs, num_nodes, d)
    local_recon_mat = local_recon.reshape(num_runs, num_nodes, m, ncols)
    # assign selected rows back
    local_recon_mat[run_idx, node_idx, row_idx, :] = selected

    # global compressed (average of local_comp across nodes)
    global_comp = cp.mean(local_comp, axis=1)  # (num_runs, K*ncols)

    # reconstruct global dense vector (average of local_recon)
    global_recon = cp.mean(local_recon, axis=1)  # (num_runs, d)

    return {
        'local_comp': local_comp,
        'local_recon': local_recon,
        'global_comp': global_comp,
        'global_recon': global_recon,
        'topk_idx': topk_idx
    }

def centralized_MSGD_batched_gpu_arc(
    init_x_gpu_batched, # init_x shape: (num_runs, n, d)
    h_data_nodes_gpu, y_data_nodes_gpu, # Original h_tilde, y_tilde on GPU
    grad_func_batched_gpu, # The new batched gradient function for GPU
    rho, lr, sigma_n,eta,
    max_it, batch_size, num_runs, m, r, seed,
    topk_ratio=None,
    use_ef=None
):
    """
    Implements centralized Momentum Stochastic Gradient Descent (MSGD), 
    batched for multiple runs on a GPU.

    Args:
        init_x_gpu_batched (cp.ndarray): Initial parameters. All nodes in a run start with the same parameters. Shape: (num_runs, n, d).
        h_data_nodes_gpu (cp.ndarray): Full input data on GPU. Shape: (n, L, d).
        y_data_nodes_gpu (cp.ndarray): Full labels on GPU. Shape: (n, L).
        grad_func_batched_gpu (function): The GPU-accelerated gradient function.
        rho (float): Regularization parameter.
        lr (float): Learning rate.
        sigma_n (float): Standard deviation of the noise added to the gradient.
        max_it (int): Maximum number of iterations.
        batch_size (int): Mini-batch size for gradient calculation.
        num_runs (int): Number of parallel simulation runs.
        beta (float): Momentum parameter.

    Returns:
        pd.DataFrame: A DataFrame containing the history of the average gradient norm.
    """
    wandb.init(
            project=f"numerial_d40_iter5000_hetero3", 
            name=f"arctopk_2"
        )
    loss_history = []

    x = cp.copy(init_x_gpu_batched)  # Parameters. Shape: (num_runs, n, d)
    num_n, num_d = x.shape[1], x.shape[2]  # Number of nodes (n) and dimensions (d)
    
    # Initialize momentum (velocity)
    velocity = cp.zeros((num_runs, 1, num_d), dtype=x.dtype)

    # # EF memory init
    # if use_ef:
    #     e = cp.zeros_like(x)  # (num_runs, n, d)
    #     h = cp.zeros_like(x)
    # else:
    #     e = None

    # # Store average gradient norm over runs
    # avg_gradient_norm_history = []

    # least_loss = 1
    # for iter_num in range(max_it):
    #     # Calculate gradients for all nodes based on their current parameters
    #     g = grad_func_batched_gpu(x, y_data_nodes_gpu, h_data_nodes_gpu, rho, batch_size, num_runs)
    #     if sigma_n > 0:
    #         # Add Gaussian noise to the gradient if specified
    #         g += sigma_n * cp.random.normal(size=(num_runs, num_n, num_d))

    #     # 如果启用 Top-K / EF21，则在这里对 g 做 EF21，然后用压缩结果 c 来聚合
    #     if topk_ratio is not None:
    #         k = int(max(1, num_d * topk_ratio))
    #         if use_ef:
    #             # # EF14：v = g + e; c = TopK(v); e = v - c; 使用 c 聚合
    #             # v = g + e
    #             # c = topk_gpu(v, k)
    #             # e = v - c
    #             # g_for_agg = c
    #             # EF21：v = g - e; c = TopK(v); e = e + c
    #             if iter_num ==0:
    #                 e = g
    #                 g_for_agg = e
    #             else:
    #                 h = (1-eta) * h + eta * g
    #                 v = h - e
    #                 arc = arctopk_gpu(v, m=m, r=r, mu=topk_ratio, seed=seed)
    #                 c_local_recon = arc['local_recon']  # (runs,nodes,d)
    #                 e = e + c_local_recon
    #                 g_for_agg = e

    #         else:
    #             # 仅 ARC-Top-K（无误差补偿）
    #             arc = arctopk_gpu(g, m=m, r=r, mu=topk_ratio, seed=seed)
    #             g_for_agg = arc['local_recon']
    #     else:
    #         # 无压缩
    #         g_for_agg = g
    # EF memory init
    if use_ef:
        e = cp.zeros((num_runs, 1, num_d), dtype=x.dtype)  # (num_runs, n, d)
        h = cp.zeros((num_runs, 1, num_d), dtype=x.dtype)
    else:
        e = None

    # Store average gradient norm over runs
    avg_gradient_norm_history = []

    least_loss = 1
    for iter_num in range(max_it):
        # Calculate gradients for all nodes based on their current parameters
        g = grad_func_batched_gpu(x, y_data_nodes_gpu, h_data_nodes_gpu, rho, batch_size, num_runs)
        g = cp.mean(g, axis=1, keepdims=True)
        if sigma_n > 0:
            # Add Gaussian noise to the gradient if specified
            g += sigma_n * cp.random.normal(size=(num_runs, 1, num_d))

        # 如果启用 Top-K / EF21，则在这里对 g 做 EF21，然后用压缩结果 c 来聚合
        if topk_ratio is not None:
            k = int(max(1, num_d * topk_ratio))
            if use_ef:
                # # EF14：v = g + e; c = TopK(v); e = v - c; 使用 c 聚合
                # v = g + e
                # c = topk_gpu(v, k)
                # e = v - c
                # g_for_agg = c
                # EF21：v = g - e; c = TopK(v); e = e + c
                if iter_num ==0:
                    e = g
                    g_for_agg = e
                else:
                    h = (1-eta) * h + eta * g
                    v = h - e
                    arc = arctopk_gpu(v, m=m, r=r, mu=topk_ratio, seed=seed)
                    c_local_recon = arc['local_recon']  # (runs,nodes,d)
                    e = e + c_local_recon
                    g_for_agg = e

            else:
                # 仅 ARC-Top-K（无误差补偿）
                arc = arctopk_gpu(g, m=m, r=r, mu=topk_ratio, seed=seed)
                g_for_agg = arc['local_recon']
        else:
            # 无压缩
            g_for_agg = g
        
        #尝试2
        # Aggregate gradients by averaging across nodes for each run 
        #g_avg = cp.mean(g_for_agg, axis=1, keepdims=True)  # Shape: (num_runs, 1, d)
        g_avg = g_for_agg  # Shape: (num_runs, 1, d)

        # # Update momentum (velocity)
        # # m_{t+1} = beta * m_t + (1 - beta) * g_avg
        # velocity = beta * velocity + (1 - beta) * g_avg

        # # Update the parameters using momentum. The update is the same for all nodes in a run.
        # x_updated_slice = x[:, 0:1, :] - lr * velocity
        
        x_updated_slice = x[:, 0:1, :] - lr * g_avg
        
        # Broadcast the updated parameters to all nodes within each run
        x = cp.broadcast_to(x_updated_slice, (num_runs, num_n, num_d))

        # --- Record history (averaged over runs) ---
        # The parameters `x` are already the mean parameters (x_mean_per_run)
        # 1. Calculate full batch gradient for each run's x_mean: _grad_on_full_per_run shape (num_runs, n, d)
        #    Use batch_size=None for full dataset, no sigma_n noise for this evaluation
        _grad_on_full_per_run = grad_func_batched_gpu(
            x, y_data_nodes_gpu, h_data_nodes_gpu,
            rho=rho, batch_size=None, num_runs=num_runs
        )
        
        # 2. Calculate mean gradient over nodes for each run: mean_grad_per_run shape (num_runs, 1, d)
        mean_grad_per_run = cp.mean(_grad_on_full_per_run, axis=1, keepdims=True)
        
        # 3. Calculate norm of mean_grad for each run: norm_per_run shape (num_runs,)
        norm_per_run = cp.linalg.norm(mean_grad_per_run, axis=2).squeeze()
        
        # 4. Average these norms over all runs: avg_norm_over_runs (scalar)
        avg_norm_over_runs_scalar = cp.mean(norm_per_run)
        avg_gradient_norm_history.append(cp.asnumpy(avg_norm_over_runs_scalar))  # Store as numpy float

        # === Compute full loss (for monitoring) ===
        loss_value = loss_full_batched_gpu(
           x, y_data_nodes_gpu, h_data_nodes_gpu, rho
        )
        l=cp.asnumpy(cp.mean(loss_value)).item()
        loss_history.append(l)
        if least_loss > l:
            least_loss = l
        if (iter_num + 1) % 10 == 0:  # Print progress
            wandb.log({
                    "train_loss": l,
                    "least_loss":least_loss,
                },step=iter_num+1)
            print(f"Iteration {iter_num+1}/{max_it}, Avg Grad Norm: {avg_norm_over_runs_scalar:.6f}, Least_Loss:{least_loss}")
    
    print("\n")
    return pd.DataFrame({
        "gradient_norm_on_full_trainset_avg": avg_gradient_norm_history,
        "loss_full_avg": loss_history,
    })

def centralized_MSGD_batched_gpu_arc_1(
    init_x_gpu_batched, # init_x shape: (num_runs, n, d)
    h_data_nodes_gpu, y_data_nodes_gpu, # Original h_tilde, y_tilde on GPU
    grad_func_batched_gpu, # The new batched gradient function for GPU
    rho, lr, sigma_n,eta,
    max_it, batch_size, num_runs, m, r, seed,
    topk_ratio=None,
    use_ef=None
):
    """
    Implements centralized Momentum Stochastic Gradient Descent (MSGD), 
    batched for multiple runs on a GPU.

    Args:
        init_x_gpu_batched (cp.ndarray): Initial parameters. All nodes in a run start with the same parameters. Shape: (num_runs, n, d).
        h_data_nodes_gpu (cp.ndarray): Full input data on GPU. Shape: (n, L, d).
        y_data_nodes_gpu (cp.ndarray): Full labels on GPU. Shape: (n, L).
        grad_func_batched_gpu (function): The GPU-accelerated gradient function.
        rho (float): Regularization parameter.
        lr (float): Learning rate.
        sigma_n (float): Standard deviation of the noise added to the gradient.
        max_it (int): Maximum number of iterations.
        batch_size (int): Mini-batch size for gradient calculation.
        num_runs (int): Number of parallel simulation runs.
        beta (float): Momentum parameter.

    Returns:
        pd.DataFrame: A DataFrame containing the history of the average gradient norm.
    """
    wandb.init(
            project=f"numerial_d40_iter5000_hetero3", 
            name=f"arctopk_1"
        )
    loss_history = []

    x = cp.copy(init_x_gpu_batched)  # Parameters. Shape: (num_runs, n, d)
    num_n, num_d = x.shape[1], x.shape[2]  # Number of nodes (n) and dimensions (d)
    
    # Initialize momentum (velocity)
    velocity = cp.zeros((num_runs, 1, num_d), dtype=x.dtype)

    # EF memory init
    if use_ef:
        e = cp.zeros_like(x)  # (num_runs, n, d)
        h = cp.zeros_like(x)
    else:
        e = None

    # Store average gradient norm over runs
    avg_gradient_norm_history = []

    least_loss = 1
    for iter_num in range(max_it):
        # Calculate gradients for all nodes based on their current parameters
        g = grad_func_batched_gpu(x, y_data_nodes_gpu, h_data_nodes_gpu, rho, batch_size, num_runs)
        if sigma_n > 0:
            # Add Gaussian noise to the gradient if specified
            g += sigma_n * cp.random.normal(size=(num_runs, num_n, num_d))

        # 如果启用 Top-K / EF21，则在这里对 g 做 EF21，然后用压缩结果 c 来聚合
        if topk_ratio is not None:
            k = int(max(1, num_d * topk_ratio))
            if use_ef:
                # # EF14：v = g + e; c = TopK(v); e = v - c; 使用 c 聚合
                # v = g + e
                # c = topk_gpu(v, k)
                # e = v - c
                # g_for_agg = c
                # EF21：v = g - e; c = TopK(v); e = e + c
                if iter_num ==0:
                    e = g
                    g_for_agg = e
                else:
                    h = (1-eta) * h + eta * g
                    v = h - e
                    arc = arctopk_gpu(v, m=m, r=r, mu=topk_ratio, seed=seed)
                    c_local_recon = arc['local_recon']  # (runs,nodes,d)
                    e = e + c_local_recon
                    g_for_agg = e

            else:
                # 仅 ARC-Top-K（无误差补偿）
                arc = arctopk_gpu(g, m=m, r=r, mu=topk_ratio, seed=seed)
                g_for_agg = arc['local_recon']
        else:
            # 无压缩
            g_for_agg = g
    
    
        

        # Aggregate gradients by averaging across nodes for each run 
        g_avg = cp.mean(g_for_agg, axis=1, keepdims=True)  # Shape: (num_runs, 1, d)
        

        # # Update momentum (velocity)
        # # m_{t+1} = beta * m_t + (1 - beta) * g_avg
        # velocity = beta * velocity + (1 - beta) * g_avg

        # # Update the parameters using momentum. The update is the same for all nodes in a run.
        # x_updated_slice = x[:, 0:1, :] - lr * velocity
        
        x_updated_slice = x[:, 0:1, :] - lr * g_avg
        
        # Broadcast the updated parameters to all nodes within each run
        x = cp.broadcast_to(x_updated_slice, (num_runs, num_n, num_d))

        # --- Record history (averaged over runs) ---
        # The parameters `x` are already the mean parameters (x_mean_per_run)
        # 1. Calculate full batch gradient for each run's x_mean: _grad_on_full_per_run shape (num_runs, n, d)
        #    Use batch_size=None for full dataset, no sigma_n noise for this evaluation
        _grad_on_full_per_run = grad_func_batched_gpu(
            x, y_data_nodes_gpu, h_data_nodes_gpu,
            rho=rho, batch_size=None, num_runs=num_runs
        )
        
        # 2. Calculate mean gradient over nodes for each run: mean_grad_per_run shape (num_runs, 1, d)
        mean_grad_per_run = cp.mean(_grad_on_full_per_run, axis=1, keepdims=True)
        
        # 3. Calculate norm of mean_grad for each run: norm_per_run shape (num_runs,)
        norm_per_run = cp.linalg.norm(mean_grad_per_run, axis=2).squeeze()
        
        # 4. Average these norms over all runs: avg_norm_over_runs (scalar)
        avg_norm_over_runs_scalar = cp.mean(norm_per_run)
        avg_gradient_norm_history.append(cp.asnumpy(avg_norm_over_runs_scalar))  # Store as numpy float

        # === Compute full loss (for monitoring) ===
        loss_value = loss_full_batched_gpu(
           x, y_data_nodes_gpu, h_data_nodes_gpu, rho
        )
        l=cp.asnumpy(cp.mean(loss_value)).item()
        loss_history.append(l)
        if least_loss > l:
            least_loss = l
        if (iter_num + 1) % 10 == 0:  # Print progress
            wandb.log({
                "train_loss": l,
                "least_loss":least_loss,
            },step=iter_num+1)

            print(f"Iteration {iter_num+1}/{max_it}, Avg Grad Norm: {avg_norm_over_runs_scalar:.6f}, Least_Loss:{least_loss}")
    
    print("\n")
    return pd.DataFrame({
        "gradient_norm_on_full_trainset_avg": avg_gradient_norm_history,
        "loss_full_avg": loss_history,
    })












def centralized_SGD2M_batched_gpu(
    init_x_gpu_batched, # init_x shape: (num_runs, n, d)
    h_data_nodes_gpu, y_data_nodes_gpu, # Original h_tilde, y_tilde on GPU
    grad_func_batched_gpu, # The new batched gradient function for GPU
    rho, lr, sigma_n, eta,
    max_it, batch_size, num_runs,
    topk_ratio=None,
    use_ef=None
):
    """
    Implements centralized Momentum Stochastic Gradient Descent (MSGD), 
    batched for multiple runs on a GPU.

    Args:
        init_x_gpu_batched (cp.ndarray): Initial parameters. All nodes in a run start with the same parameters. Shape: (num_runs, n, d).
        h_data_nodes_gpu (cp.ndarray): Full input data on GPU. Shape: (n, L, d).
        y_data_nodes_gpu (cp.ndarray): Full labels on GPU. Shape: (n, L).
        grad_func_batched_gpu (function): The GPU-accelerated gradient function.
        rho (float): Regularization parameter.
        lr (float): Learning rate.
        sigma_n (float): Standard deviation of the noise added to the gradient.
        max_it (int): Maximum number of iterations.
        batch_size (int): Mini-batch size for gradient calculation.
        num_runs (int): Number of parallel simulation runs.
        beta (float): Momentum parameter.

    Returns:
        pd.DataFrame: A DataFrame containing the history of the average gradient norm.
    """
    loss_history = []

    x = cp.copy(init_x_gpu_batched)  # Parameters. Shape: (num_runs, n, d)
    num_n, num_d = x.shape[1], x.shape[2]  # Number of nodes (n) and dimensions (d)
    
    # Initialize momentum (velocity)
    velocity = cp.zeros((num_runs, 1, num_d), dtype=x.dtype)

    # EF memory init
    if use_ef:
        e = cp.zeros_like(x)  # (num_runs, n, d)
        h = cp.zeros_like(x)
        u = cp.zeros_like(x)
    else:
        e = None

    # Store average gradient norm over runs
    avg_gradient_norm_history = []

    least_loss = 1
    for iter_num in range(max_it):
        # Calculate gradients for all nodes based on their current parameters
        g = grad_func_batched_gpu(x, y_data_nodes_gpu, h_data_nodes_gpu, rho, batch_size, num_runs)
        if sigma_n > 0:
            # Add Gaussian noise to the gradient if specified
            g += sigma_n * cp.random.normal(size=(num_runs, num_n, num_d))

        # 如果启用 Top-K / EF21，则在这里对 g 做 EF21，然后用压缩结果 c 来聚合
        if topk_ratio is not None:
            k = int(max(1, num_d * topk_ratio))
            if use_ef:
                # # EF14：v = g + e; c = TopK(v); e = v - c; 使用 c 聚合
                # v = g + e
                # c = topk_gpu(v, k)
                # e = v - c
                # g_for_agg = c
                # EF21：v = g - e; c = TopK(v); e = e + c
                if iter_num ==0:
                    e = g
                    g_for_agg = e
                else:
                    h = (1-eta) * h + eta * g
                    u = (1-eta) * u +eta * h
                    v = u - e
                    c = topk_gpu(v, k)
                    e = e + c
                    g_for_agg = e

            else:
                # 仅 Top-K（无误差补偿）
                g_for_agg = topk_gpu(g, k)
        else:
            # 无压缩
            g_for_agg = g
    
        # Aggregate gradients by averaging across nodes for each run
        g_avg = cp.mean(g_for_agg, axis=1, keepdims=True)  # Shape: (num_runs, 1, d)

        # # Update momentum (velocity)
        # # m_{t+1} = beta * m_t + (1 - beta) * g_avg
        # velocity = beta * velocity + (1 - beta) * g_avg

        # # Update the parameters using momentum. The update is the same for all nodes in a run.
        # x_updated_slice = x[:, 0:1, :] - lr * velocity
        
        x_updated_slice = x[:, 0:1, :] - lr * g_avg

        # Broadcast the updated parameters to all nodes within each run
        x = cp.broadcast_to(x_updated_slice, (num_runs, num_n, num_d))

        # --- Record history (averaged over runs) ---
        # The parameters `x` are already the mean parameters (x_mean_per_run)
        # 1. Calculate full batch gradient for each run's x_mean: _grad_on_full_per_run shape (num_runs, n, d)
        #    Use batch_size=None for full dataset, no sigma_n noise for this evaluation
        _grad_on_full_per_run = grad_func_batched_gpu(
            x, y_data_nodes_gpu, h_data_nodes_gpu,
            rho=rho, batch_size=None, num_runs=num_runs
        )
        
        # 2. Calculate mean gradient over nodes for each run: mean_grad_per_run shape (num_runs, 1, d)
        mean_grad_per_run = cp.mean(_grad_on_full_per_run, axis=1, keepdims=True)
        
        # 3. Calculate norm of mean_grad for each run: norm_per_run shape (num_runs,)
        norm_per_run = cp.linalg.norm(mean_grad_per_run, axis=2).squeeze()
        
        # 4. Average these norms over all runs: avg_norm_over_runs (scalar)
        avg_norm_over_runs_scalar = cp.mean(norm_per_run)
        avg_gradient_norm_history.append(cp.asnumpy(avg_norm_over_runs_scalar))  # Store as numpy float

        # === Compute full loss (for monitoring) ===
        loss_value = loss_full_batched_gpu(
           x, y_data_nodes_gpu, h_data_nodes_gpu, rho
        )
        l=cp.asnumpy(cp.mean(loss_value)).item()
        loss_history.append(l)
        if least_loss > l:
            least_loss = l
        if (iter_num + 1) % 10 == 0:  # Print progress
            print(f"Iteration {iter_num+1}/{max_it}, Avg Grad Norm: {avg_norm_over_runs_scalar:.6f}, Least_Loss:{least_loss}")
    print("\n")
    return pd.DataFrame({
        "gradient_norm_on_full_trainset_avg": avg_gradient_norm_history,
        "loss_full_avg": loss_history,
    })


def centralized_SGD2M_batched_gpu_randk(
    init_x_gpu_batched, # init_x shape: (num_runs, n, d)
    h_data_nodes_gpu, y_data_nodes_gpu, # Original h_tilde, y_tilde on GPU
    grad_func_batched_gpu, # The new batched gradient function for GPU
    rho, lr, sigma_n,eta,
    max_it, batch_size, num_runs,
    topk_ratio=None,
    use_ef=None
):
    """
    Implements centralized Momentum Stochastic Gradient Descent (MSGD), 
    batched for multiple runs on a GPU.

    Args:
        init_x_gpu_batched (cp.ndarray): Initial parameters. All nodes in a run start with the same parameters. Shape: (num_runs, n, d).
        h_data_nodes_gpu (cp.ndarray): Full input data on GPU. Shape: (n, L, d).
        y_data_nodes_gpu (cp.ndarray): Full labels on GPU. Shape: (n, L).
        grad_func_batched_gpu (function): The GPU-accelerated gradient function.
        rho (float): Regularization parameter.
        lr (float): Learning rate.
        sigma_n (float): Standard deviation of the noise added to the gradient.
        max_it (int): Maximum number of iterations.
        batch_size (int): Mini-batch size for gradient calculation.
        num_runs (int): Number of parallel simulation runs.
        beta (float): Momentum parameter.

    Returns:
        pd.DataFrame: A DataFrame containing the history of the average gradient norm.
    """
    loss_history = []

    x = cp.copy(init_x_gpu_batched)  # Parameters. Shape: (num_runs, n, d)
    num_n, num_d = x.shape[1], x.shape[2]  # Number of nodes (n) and dimensions (d)
    
    # Initialize momentum (velocity)
    velocity = cp.zeros((num_runs, 1, num_d), dtype=x.dtype)

    # EF memory init
    if use_ef:
        e = cp.zeros_like(x)  # (num_runs, n, d)
        h = cp.zeros_like(x)
        u = cp.zeros_like(x)
    else:
        e = None

    # Store average gradient norm over runs
    avg_gradient_norm_history = []
    
    least_loss = 1
    for iter_num in range(max_it):
        # Calculate gradients for all nodes based on their current parameters
        g = grad_func_batched_gpu(x, y_data_nodes_gpu, h_data_nodes_gpu, rho, batch_size, num_runs)
        if sigma_n > 0:
            # Add Gaussian noise to the gradient if specified
            g += sigma_n * cp.random.normal(size=(num_runs, num_n, num_d))

        # 如果启用 Top-K / EF21，则在这里对 g 做 EF21，然后用压缩结果 c 来聚合
        if topk_ratio is not None:
            k = int(max(1, num_d * topk_ratio))
            # print(k)
            if use_ef:
                # # EF14：v = g + e; c = TopK(v); e = v - c; 使用 c 聚合
                # v = g + e
                # c = topk_gpu(v, k)
                # e = v - c
                # g_for_agg = c
                # EF21：v = g - e; c = TopK(v); e = e + c
                if iter_num ==0:
                    e = g
                    g_for_agg = e
                else:
                    h = (1-eta) * h + eta * g
                    u = (1-eta) * u +eta * h
                    v = u - e
                    c = randk_gpu(v, k)
                    e = e + c
                    g_for_agg = e


            else:
                # 仅 Top-K（无误差补偿）
                g_for_agg = randk_gpu(v, k)
        else:
            # 无压缩
            g_for_agg = g
    
        # Aggregate gradients by averaging across nodes for each run
        g_avg = cp.mean(g_for_agg, axis=1, keepdims=True)  # Shape: (num_runs, 1, d)
        
        # # Update momentum (velocity)
        # # m_{t+1} = beta * m_t + (1 - beta) * g_avg
        # velocity = beta * velocity + (1 - beta) * g_avg

        # # Update the parameters using momentum. The update is the same for all nodes in a run.
        # x_updated_slice = x[:, 0:1, :] - lr * velocity
        
        x_updated_slice = x[:, 0:1, :] - lr * g_avg
        
        # Broadcast the updated parameters to all nodes within each run
        x = cp.broadcast_to(x_updated_slice, (num_runs, num_n, num_d))

        # --- Record history (averaged over runs) ---
        # The parameters `x` are already the mean parameters (x_mean_per_run)
        # 1. Calculate full batch gradient for each run's x_mean: _grad_on_full_per_run shape (num_runs, n, d)
        #    Use batch_size=None for full dataset, no sigma_n noise for this evaluation
        _grad_on_full_per_run = grad_func_batched_gpu(
            x, y_data_nodes_gpu, h_data_nodes_gpu,
            rho=rho, batch_size=None, num_runs=num_runs
        )
        
        # 2. Calculate mean gradient over nodes for each run: mean_grad_per_run shape (num_runs, 1, d)
        mean_grad_per_run = cp.mean(_grad_on_full_per_run, axis=1, keepdims=True)
        
        # 3. Calculate norm of mean_grad for each run: norm_per_run shape (num_runs,)
        norm_per_run = cp.linalg.norm(mean_grad_per_run, axis=2).squeeze()
        
        # 4. Average these norms over all runs: avg_norm_over_runs (scalar)
        avg_norm_over_runs_scalar = cp.mean(norm_per_run)
        avg_gradient_norm_history.append(cp.asnumpy(avg_norm_over_runs_scalar))  # Store as numpy float

        # === Compute full loss (for monitoring) ===
        loss_value = loss_full_batched_gpu(
           x, y_data_nodes_gpu, h_data_nodes_gpu, rho
        )
        l=cp.asnumpy(cp.mean(loss_value)).item()
        loss_history.append(l)
        if least_loss > l:
            least_loss = l
        if (iter_num + 1) % 10 == 0:  # Print progress
            print(f"Iteration {iter_num+1}/{max_it}, Avg Grad Norm: {avg_norm_over_runs_scalar:.6f}, Least_Loss:{least_loss}")
    print("\n")
    return pd.DataFrame({
        "gradient_norm_on_full_trainset_avg": avg_gradient_norm_history,
        "loss_full_avg": loss_history,
    })


def centralized_SGD2M_batched_gpu_arc(
    init_x_gpu_batched, # init_x shape: (num_runs, n, d)
    h_data_nodes_gpu, y_data_nodes_gpu, # Original h_tilde, y_tilde on GPU
    grad_func_batched_gpu, # The new batched gradient function for GPU
    rho, lr, sigma_n,eta,
    max_it, batch_size, num_runs, m, r, seed,
    topk_ratio=None,
    use_ef=None
):
    """
    Implements centralized Momentum Stochastic Gradient Descent (MSGD), 
    batched for multiple runs on a GPU.

    Args:
        init_x_gpu_batched (cp.ndarray): Initial parameters. All nodes in a run start with the same parameters. Shape: (num_runs, n, d).
        h_data_nodes_gpu (cp.ndarray): Full input data on GPU. Shape: (n, L, d).
        y_data_nodes_gpu (cp.ndarray): Full labels on GPU. Shape: (n, L).
        grad_func_batched_gpu (function): The GPU-accelerated gradient function.
        rho (float): Regularization parameter.
        lr (float): Learning rate.
        sigma_n (float): Standard deviation of the noise added to the gradient.
        max_it (int): Maximum number of iterations.
        batch_size (int): Mini-batch size for gradient calculation.
        num_runs (int): Number of parallel simulation runs.
        beta (float): Momentum parameter.

    Returns:
        pd.DataFrame: A DataFrame containing the history of the average gradient norm.
    """
    loss_history = []

    x = cp.copy(init_x_gpu_batched)  # Parameters. Shape: (num_runs, n, d)
    num_n, num_d = x.shape[1], x.shape[2]  # Number of nodes (n) and dimensions (d)
    
    # Initialize momentum (velocity)
    velocity = cp.zeros((num_runs, 1, num_d), dtype=x.dtype)

    # EF memory init
    if use_ef:
        e = cp.zeros_like(x)  # (num_runs, n, d)
        h = cp.zeros_like(x)
        u = cp.zeros_like(x)
    else:
        e = None

    # Store average gradient norm over runs
    avg_gradient_norm_history = []

    least_loss = 1
    for iter_num in range(max_it):
        # Calculate gradients for all nodes based on their current parameters
        g = grad_func_batched_gpu(x, y_data_nodes_gpu, h_data_nodes_gpu, rho, batch_size, num_runs)
        if sigma_n > 0:
            # Add Gaussian noise to the gradient if specified
            g += sigma_n * cp.random.normal(size=(num_runs, num_n, num_d))

        # 如果启用 Top-K / EF21，则在这里对 g 做 EF21，然后用压缩结果 c 来聚合
        if topk_ratio is not None:
            k = int(max(1, num_d * topk_ratio))
            if use_ef:
                # # EF14：v = g + e; c = TopK(v); e = v - c; 使用 c 聚合
                # v = g + e
                # c = topk_gpu(v, k)
                # e = v - c
                # g_for_agg = c
                # EF21：v = g - e; c = TopK(v); e = e + c
                if iter_num ==0:
                    e = g
                    g_for_agg = e
                else:
                    h = (1-eta) * h + eta * g
                    u = (1-eta) * u +eta * h
                    v = u - e
                    arc = arctopk_gpu(v, m=m, r=r, mu=topk_ratio, seed=seed)
                    c_local_recon = arc['local_recon']  # (runs,nodes,d)
                    e = e + c_local_recon
                    g_for_agg = e


            else:
                # 仅 ARC-Top-K（无误差补偿）
                arc = arctopk_gpu(g, m=m, r=r, mu=topk_ratio, seed=seed)
                g_for_agg = arc['local_recon']
        else:
            # 无压缩
            g_for_agg = g
    
        # Aggregate gradients by averaging across nodes for each run
        g_avg = cp.mean(g_for_agg, axis=1, keepdims=True)  # Shape: (num_runs, 1, d)

        # # Update momentum (velocity)
        # # m_{t+1} = beta * m_t + (1 - beta) * g_avg
        # velocity = beta * velocity + (1 - beta) * g_avg

        # # Update the parameters using momentum. The update is the same for all nodes in a run.
        # x_updated_slice = x[:, 0:1, :] - lr * velocity
        
        x_updated_slice = x[:, 0:1, :] - lr * g_avg
        
        # Broadcast the updated parameters to all nodes within each run
        x = cp.broadcast_to(x_updated_slice, (num_runs, num_n, num_d))

        # --- Record history (averaged over runs) ---
        # The parameters `x` are already the mean parameters (x_mean_per_run)
        # 1. Calculate full batch gradient for each run's x_mean: _grad_on_full_per_run shape (num_runs, n, d)
        #    Use batch_size=None for full dataset, no sigma_n noise for this evaluation
        _grad_on_full_per_run = grad_func_batched_gpu(
            x, y_data_nodes_gpu, h_data_nodes_gpu,
            rho=rho, batch_size=None, num_runs=num_runs
        )
        
        # 2. Calculate mean gradient over nodes for each run: mean_grad_per_run shape (num_runs, 1, d)
        mean_grad_per_run = cp.mean(_grad_on_full_per_run, axis=1, keepdims=True)
        
        # 3. Calculate norm of mean_grad for each run: norm_per_run shape (num_runs,)
        norm_per_run = cp.linalg.norm(mean_grad_per_run, axis=2).squeeze()
        
        # 4. Average these norms over all runs: avg_norm_over_runs (scalar)
        avg_norm_over_runs_scalar = cp.mean(norm_per_run)
        avg_gradient_norm_history.append(cp.asnumpy(avg_norm_over_runs_scalar))  # Store as numpy float

        # === Compute full loss (for monitoring) ===
        loss_value = loss_full_batched_gpu(
           x, y_data_nodes_gpu, h_data_nodes_gpu, rho
        )
        l=cp.asnumpy(cp.mean(loss_value)).item()
        loss_history.append(l)
        if least_loss > l:
            least_loss = l
        if (iter_num + 1) % 10 == 0:  # Print progress
            print(f"Iteration {iter_num+1}/{max_it}, Avg Grad Norm: {avg_norm_over_runs_scalar:.6f}, Least_Loss:{least_loss}")
    
    print("\n")
    return pd.DataFrame({
        "gradient_norm_on_full_trainset_avg": avg_gradient_norm_history,
        "loss_full_avg": loss_history,
    })
