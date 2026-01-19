import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime

# --- 0. 基础设置 ---
def setup_seed(seed):
    np.random.seed(seed)

def get_timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# --- 1. 目标函数 (严谨、防溢出、平移耦合) ---

class RobustShiftedObjective:
    def __init__(self, num_nodes, dim, blocks, block_size, 
                 noise_block_count, signal_block_indices, 
                 noise_scale, signal_scale, shift_gamma):
        
        self.num_nodes = num_nodes
        self.dim = dim
        self.block_size = block_size
        self.noise_block_count = noise_block_count
        self.signal_block_indices = signal_block_indices
        self.noise_scale = noise_scale    
        self.signal_scale = signal_scale 
        self.gamma = shift_gamma          
        self.scale_factor = 1.0 / max(1, self.noise_block_count)

        # 1. 初始化 shift 和 gamma 矩阵 (保持原样)
        self.static_shifts = np.zeros((num_nodes, dim))
        self.gamma_vecs = np.zeros((num_nodes, dim))
        half_n = num_nodes // 2
        
        noise_indices = list(range(self.noise_block_count))
        for i, b_idx in enumerate(noise_indices):
            s = self._get_slice(b_idx)
            # Group A
            self.static_shifts[:half_n, s] = self.noise_scale
            self.gamma_vecs[:half_n, s] = self.gamma
            # Group B
            self.static_shifts[half_n:, s] = -self.noise_scale
            self.gamma_vecs[half_n:, s] = -self.gamma

        # --- 2. 修正：使用数值方法求解精确的 w_star ---
        # 理由：手算公式依赖完美对称假设，容易在参数变动时失效。
        # 既然是凸问题，我们直接计算梯度为0的解。
        
        # 提取全局平均的参数 (Global Average)
        # 因为我们是在解 Global Consensus Problem (所有节点 w 相同)
        # Loss = 0.5(w-1)^2 + Mean_blocks( 0.5 * Mean_nodes( ||w_n - (xi + gam*w_s)||^2 ) )
        
        # 计算所有节点在噪声块上的 Gamma 和 Shift 的平均值
        # 注意：这里我们只取第一个噪声块的统计量即可，因为所有噪声块同分布
        sample_slice = self._get_slice(0)
        mean_xi = np.mean(self.static_shifts[:, sample_slice]) # 理论应为 0
        mean_gam = np.mean(self.gamma_vecs[:, sample_slice])   # 理论应为 0
        
        # 交互项的平均值 E[gamma * xi] 和 E[gamma^2]
        # 注意：这里是对 "所有节点" 求平均
        vals_xi = self.static_shifts[:, sample_slice]
        vals_gam = self.gamma_vecs[:, sample_slice]
        
        mean_gam_xi = np.mean(vals_gam * vals_xi) # 理论应为 G*L
        mean_gam_sq = np.mean(vals_gam ** 2)      # 理论应为 G^2
        
        # 使用更通用的公式计算 Signal Block 的最优解
        # w_s * (1 + E[gam^2]) = signal_scale - E[gam * xi] + E[gam]*E[xi]... 
        # 简化后: w_s = (S - E[gam*xi] - mean_xi*mean_gam) / (1 + mean_gam_sq)
        # (假设 mean_xi 和 mean_gam 接近0，主要由交互项主导)
        
        numerator = self.signal_scale - mean_gam_xi
        denominator = 1.0 + mean_gam_sq
        
        real_w_s = numerator / denominator
        print(f"[System] Computed Robust w_s*: {real_w_s:.6f}")

        # 计算 Noise Block 的最优解
        # w_n* = E[xi] + E[gam] * w_s*
        real_w_n = mean_xi + mean_gam * real_w_s
        print(f"[System] Computed Robust w_n*: {real_w_n:.6f} (Should be close to 0)")

        # 赋值
        self.w_star = np.zeros(dim)
        for b_idx in self.signal_block_indices:
            s = self._get_slice(b_idx)
            self.w_star[s] = real_w_s
            
        for b_idx in noise_indices:
            s = self._get_slice(b_idx)
            self.w_star[s] = real_w_n
    
    def _get_slice(self, block_idx):
        return slice(block_idx * self.block_size, (block_idx + 1) * self.block_size)

    def get_grads(self, w_current, noise_std=0.0):
        """
        计算梯度，并支持添加高斯噪声。
        noise_std: 噪声的标准差 (Standard Deviation)
        """
        grads = np.zeros((1, self.num_nodes, self.dim))
        
        # 1. 提取 Signal Block
        sig_idx = self.signal_block_indices[0]
        sig_slice = self._get_slice(sig_idx)
        w_s_vals = w_current[sig_slice] 
        w_s_broad = np.tile(w_s_vals, (self.num_nodes, 1))
        
        grad_s_accum = np.zeros((self.num_nodes, self.block_size))
        
        # 2. 遍历噪声块
        for i in range(self.noise_block_count):
            n_slice = self._get_slice(i)
            
            w_n_local = np.tile(w_current[n_slice], (self.num_nodes, 1))
            xi = self.static_shifts[:, n_slice]
            gam = self.gamma_vecs[:, n_slice]
            
            shift_dynamic = xi + gam * w_s_broad
            residual = w_n_local - shift_dynamic
            
            # dL/dw_n
            grads[0, :, n_slice] = residual * self.scale_factor
            
            # dL/dw_s (Cross Term)
            grad_s_accum += (residual * (-gam)) * self.scale_factor
            
        # 3. Signal Block Gradient
        grads[0, :, sig_slice] = (w_s_broad - self.signal_scale) + grad_s_accum
        
        # 4. 溢出保护
        if np.isnan(grads).any():
            grads = np.nan_to_num(grads, nan=0.0, posinf=1e5, neginf=-1e5)

        # 5. 添加高斯噪声 (只在 noise_std > 0 时添加)
        if noise_std > 0:
            noise = np.random.normal(loc=0.0, scale=noise_std, size=grads.shape)
            grads += noise
            
        return grads

    def get_loss(self, w_current):
        """
        计算全局 Loss (Average over nodes):
        L(w) = 0.5 ||w_s - S||^2 + (0.5/Count) * Mean_over_Nodes( Sum_j ||w_nj - (xi_j + gamma_j * w_s)||^2 )
        """
        # 1. Signal Loss
        sig_idx = self.signal_block_indices[0]
        sig_slice = self._get_slice(sig_idx)
        w_s = w_current[sig_slice]
        loss_s = 0.5 * np.sum((w_s - self.signal_scale)**2)
        
        # 2. Noise Loss
        noise_end = self.noise_block_count * self.block_size
        noise_slice = slice(0, noise_end)
        
        w_n_all = w_current[noise_slice] # (TotalNoiseDim,)
        
        # Reconstruct w_s for broadcasting (Tile w_s for each noise block)
        w_s_tiled = np.tile(w_s, self.noise_block_count) # (TotalNoiseDim,)
        w_s_broad_all = np.tile(w_s_tiled, (self.num_nodes, 1)) # (Nodes, TotalNoiseDim)
        
        xi_all = self.static_shifts[:, noise_slice]
        gam_all = self.gamma_vecs[:, noise_slice]
        
        target_all = xi_all + gam_all * w_s_broad_all
        
        # Broadcasting w_n_all to (Nodes, TotalNoiseDim)
        diff = w_n_all - target_all 
        
        sq_err = np.sum(diff**2, axis=1) # Sum over dimensions -> (Nodes,)
        avg_sq_err = np.mean(sq_err)     # Average over nodes
        
        loss_n = self.scale_factor * 0.5 * avg_sq_err
        
        return loss_s + loss_n

    def get_dist(self, w_current):
        return np.linalg.norm(w_current - self.w_star)

# --- 2. 压缩算子 ---

def full_numpy(g, m, mu, **kwargs):
    return g

def topk_block_numpy(g, m, mu, **kwargs):
    runs, n, d = g.shape
    ncols = d // m
    k = max(1, min(int(np.ceil(mu * m)), m))
    g_view = g.reshape(runs, n, m, ncols)
    norms = np.sum(g_view ** 2, axis=-1)
    idx = np.argpartition(norms, -k, axis=-1)[..., -k:]
    out = np.zeros_like(g_view)
    for r in range(runs):
        for i in range(n):
            selected = idx[r, i]
            out[r, i, selected, :] = g_view[r, i, selected, :]
    return out.reshape(runs, n, d)

def random_block_numpy(g, m, mu, **kwargs):
    """
    Random Block: 随机选择 k 个 Block。
    关键约束：所有 Node 必须选择相同的 Block (Shared Randomness)。
    """
    runs, n, d = g.shape
    ncols = d // m
    k = max(1, min(int(np.ceil(mu * m)), m))
    g_view = g.reshape(runs, n, m, ncols)
    out = np.zeros_like(g_view)
    
    for r in range(runs):
        # 1. 随机生成 k 个不重复的 block index
        # 注意：这里只生成一次，用于该 run 下的所有 nodes
        selected_blocks = np.random.choice(m, k, replace=False)
        
        # 2. 将选中的 blocks 应用到所有 nodes
        # [r, :, selected_blocks, :] 选中了当前 run 的所有 node 的指定 blocks
        out[r, :, selected_blocks, :] = g_view[r, :, selected_blocks, :]
            
    return out.reshape(runs, n, d)

def arctopk_numpy(g, m, mu, **kwargs):
    num_runs, num_nodes, d = g.shape
    ncols = d // m
    K = max(1, min(int(np.ceil(mu * m)), m))
    g_mat = g.reshape((num_runs, num_nodes, m, ncols))
    P_avg = np.mean(g_mat, axis=1) 
    Sigma = np.sum(P_avg * P_avg, axis=2) 
    idx_part = np.argpartition(Sigma, -K, axis=1)[:, -K:]
    out = np.zeros_like(g_mat)
    for r in range(num_runs):
        selected = idx_part[r]
        out[r, :, selected, :] = g_mat[r, :, selected, :]
    return {'local_recon': out.reshape(num_runs, num_nodes, d)}

def arctopk_sketch_numpy(g, m, mu, sketch_dim=2, **kwargs):
    """
    Sketch ArcTopK: 先投影，再算能量，再选块
    """
    num_runs, num_nodes, d = g.shape
    ncols = d // m 
    K = max(1, min(int(np.ceil(mu * m)), m))
    g_mat = g.reshape((num_runs, num_nodes, m, ncols))
    
    # 1. Global Mean
    P_avg = np.mean(g_mat, axis=1) # (runs, m, ncols)
    
    # 2. Sketching
    # R: (runs, ncols, sketch_dim)
    # 保证所有节点使用同一个随机矩阵 R (Seed控制)
    R = np.random.randn(num_runs, ncols, sketch_dim)
    P_sketch = np.matmul(P_avg, R) # (runs, m, sketch_dim)
    
    # 3. Energy Estimation
    Sigma_sketch = np.sum(P_sketch * P_sketch, axis=2) 
    
    # 4. Selection
    idx_part = np.argpartition(Sigma_sketch, -K, axis=1)[:, -K:]
    
    out = np.zeros_like(g_mat)
    for r in range(num_runs):
        selected = idx_part[r]
        out[r, :, selected, :] = g_mat[r, :, selected, :]
    return {'local_recon': out.reshape(num_runs, num_nodes, d)}

# --- 3. 优化器 (Strict Cold Start) ---
class UniversalOptimizer:
    def __init__(self, mode, compressor_func, shape, m, mu, eta, sketch_dim=2):
        self.mode = mode
        self.compressor_func = compressor_func
        self.m = m
        self.mu = mu
        self.eta = eta
        self.sketch_dim = sketch_dim
        
        self.v_state = np.zeros(shape) 
        self.u_state = np.zeros(shape) 
        self.e_state = np.zeros(shape) # 初始化为0
        self.iter_num = 0

    def step(self, g):
        if self.mode == "EF21-MSGD":
            # 动量更新
            # if self.iter_num == 0: self.v_state = g.copy()
            # else: self.v_state = self.eta * self.v_state + (1 - self.eta) * g
            self.v_state = self.eta * self.v_state + g
            
            target = self.v_state
            
            # [Strict Cold Start] 无论第几轮，都计算 diff 并压缩
            diff = target - self.e_state
            
            res = self.compressor_func(diff, self.m, self.mu, sketch_dim=self.sketch_dim)
            c = res['local_recon'] if isinstance(res, dict) else res
            
            self.e_state = self.e_state + c
            self.iter_num += 1
            return self.e_state
            
        elif self.mode == "EF21 Double Momentum":
            # if self.iter_num == 0:
            #     self.v_state = g.copy()
            #     self.u_state = g.copy()
            # else:
            #     self.v_state = self.eta * self.v_state + (1 - self.eta) * g
            #     self.u_state = self.eta * self.u_state + (1 - self.eta) * self.v_state

            self.v_state = self.eta * self.v_state + g
            self.u_state = self.eta * self.u_state + self.v_state
            
            target = self.u_state
            diff = target - self.e_state
            res = self.compressor_func(diff, self.m, self.mu, sketch_dim=self.sketch_dim)
            c = res['local_recon'] if isinstance(res, dict) else res
            
            self.e_state = self.e_state + c
            self.iter_num += 1
            return self.e_state
            
        else: raise ValueError(f"Unknown mode: {self.mode}")

# --- 4. 实验主逻辑 ---

def run_experiment(seed=42):
    setup_seed(seed)
    timestamp = get_timestamp()
    
    # === 参数 ===
    NUM_NODES = 10
    DIM = 2000
    BLOCKS = 200
    BLOCK_SIZE = 10
    
    # MU = 0.01 # Top-1 Communication
    MU = 0.05 # Top-1 Communication
    
    NOISE_BLOCK_COUNT = 150
    SIGNAL_BLOCK_INDICES = [NOISE_BLOCK_COUNT]
    
    NOISE_SCALE = 100.0
    SIGNAL_SCALE = 1.0
    
    # Gamma: 信号变动导致噪声平移的系数
    # 调大 Gamma 可以增加 "惩罚" 力度
    SHIFT_GAMMA = 5.0 
    
    # 为了稳定，Learning Rate 不宜过大
    # LR = 0.005
    LR = [0.001, 0.001, 0.001, 0.001] # 对应四种优化器的学习率
    STEPS = 1000
    MOMENTUM_BETA = 0.5
    SKETCH_DIM = 2
    SMOOTH_FACTOR = 0.0
    NOISE_STD = 0.001
    
    print(f"=== Experiment: Robust Shifted Objective ({timestamp}) ===")
    print(f"Goal: Prove LocalTopK fails under Bandwidth Starvation + Moving Target")
    print("-" * 60)
    
    objective = RobustShiftedObjective(
        NUM_NODES, DIM, BLOCKS, BLOCK_SIZE,
        NOISE_BLOCK_COUNT, SIGNAL_BLOCK_INDICES,
        NOISE_SCALE, SIGNAL_SCALE, SHIFT_GAMMA
    )
    
    compressors = {
        'No Compressor': full_numpy,
        'Random Block': random_block_numpy,  # <--- Added here
        'Local TopK': topk_block_numpy,
        'ArcTopK': arctopk_numpy,
        'ArcTopK-Sketch': arctopk_sketch_numpy
    }
    

    optimizers = ["EF21-MSGD", "EF21 Double Momentum"]
    
    results = {opt: {} for opt in optimizers}
    loss_results = {opt: {} for opt in optimizers}
    
    # === 运行 ===
    for idx, opt_mode in enumerate(optimizers):
        for comp_name, comp_func in compressors.items():
            setup_seed(seed) # Reset seed
            print(f"Running [{opt_mode}] + [{comp_name}] ...")
            
            w = np.zeros(DIM)
            opt = UniversalOptimizer(
                opt_mode, comp_func, (1, NUM_NODES, DIM), 
                BLOCKS, MU, MOMENTUM_BETA, SKETCH_DIM
            )
            
            dists = []
            losses = []
            for t in range(STEPS):
                # 计算梯度
                g = objective.get_grads(w, noise_std=NOISE_STD)
                # 优化步
                g_compressed = opt.step(g)
                update = np.mean(g_compressed, axis=1).flatten()
                w -= LR[idx] * update
                
                # 记录距离
                dist = objective.get_dist(w)
                dists.append(dist)
                
                # 记录 Loss
                loss = objective.get_loss(w)
                losses.append(loss)
                
                # 安全检查
                if dist > 1e5 or np.isnan(dist):
                    print("  -> Diverged! Stopping early.")
                    dists.extend([dist] * (STEPS - t - 1))
                    losses.extend([loss] * (STEPS - t - 1))
                    break
            
            results[opt_mode][comp_name] = dists
            loss_results[opt_mode][comp_name] = losses

    # === 保存 ===
    data_dict = {}
    loss_data_dict = {}
    for opt in optimizers:
        for comp in compressors:
            data_dict[f"{opt}_{comp}"] = results[opt][comp]
            loss_data_dict[f"{opt}_{comp}"] = loss_results[opt][comp]
            
    df = pd.DataFrame(data_dict)
    df.to_csv(f"robust_benchmark_{timestamp}.csv", index_label="Iteration")
    
    df_loss = pd.DataFrame(loss_data_dict)
    df_loss.to_csv(f"robust_benchmark_loss_{timestamp}.csv", index_label="Iteration")
    
# === 绘图 (优化版：确保曲线都在图像内部) ===

    # 平滑函数
    def smooth_curve(points, factor=0.9):
        smoothed_points = []
        for point in points:
            if smoothed_points:
                previous = smoothed_points[-1]
                smoothed_points.append(previous * factor + point * (1 - factor))
            else:
                smoothed_points.append(point)
        return smoothed_points
    
    # 1. 收集所有有效数据以确定全局 Y 轴范围 (Distance)
    all_values = []
    for opt in optimizers:
        for comp in compressors:
            series = results[opt][comp]
            valid_series = [v for v in series if np.isfinite(v)]
            all_values.extend(valid_series)
            
    # 收集 Loss 数据范围
    all_loss_values = []
    for opt in optimizers:
        for comp in compressors:
            series = loss_results[opt][comp]
            valid_series = [v for v in series if np.isfinite(v)]
            all_loss_values.extend(valid_series)
    
    # 2. 确定上下界 (Log Scale 适配)
    def get_log_limits(values):
        valid_pos = [v for v in values if v > 1e-20]
        if not valid_pos: return 0.01, 10.0
        g_min, g_max = min(valid_pos), max(valid_pos)
        return g_min * 0.5, g_max * 2.0

    y_min, y_max = get_log_limits(all_values)
    l_min, l_max = get_log_limits(all_loss_values)
    
    # 修改布局为 2 行 4 列
    fig, axes = plt.subplots(2, len(optimizers), figsize=(24, 12), sharex=True)
    
    styles = {
        'No Compressor': {'c': 'black', 'ls': '-', 'lw': 1.5, 'alpha': 0.6},
        'Random Block':  {'c': 'purple','ls': '-.', 'lw': 2.0, 'alpha': 0.8}, # <--- Added here
        'Local TopK':    {'c': 'red',   'ls': ':', 'lw': 2.0, 'alpha': 0.9},
        'ArcTopK':       {'c': 'blue',  'ls': '-', 'lw': 2.5, 'alpha': 0.8},
        'ArcTopK-Sketch':{'c': 'green', 'ls': '--','lw': 2.5, 'alpha': 0.8}
    }
    
    for i, opt_mode in enumerate(optimizers):
        # Row 0: Distance
        ax_dist = axes[0, i]
        # Row 1: Loss
        ax_loss = axes[1, i]
        
        for comp_name in compressors.keys():
            # Plot Distance
            data = results[opt_mode][comp_name]
            # Apply smoothing
            data_smooth = smooth_curve(data, factor=SMOOTH_FACTOR)
            ax_dist.plot(data_smooth, label=comp_name, **styles[comp_name])
            
            # Plot Loss
            loss_data = loss_results[opt_mode][comp_name]
            # Apply smoothing
            loss_smooth = smooth_curve(loss_data, factor=SMOOTH_FACTOR)
            ax_loss.plot(loss_smooth, label=comp_name, **styles[comp_name])
        
        ax_dist.set_title(f"{opt_mode} (Dist)", fontsize=14)
        ax_loss.set_title(f"{opt_mode} (Loss)", fontsize=14)
        
        ax_loss.set_xlabel("Iteration")
        
        # Log Scale & Limits
        ax_dist.set_yscale('log')
        ax_dist.set_ylim(y_min, y_max)
        
        ax_loss.set_yscale('log')
        ax_loss.set_ylim(l_min, l_max)
        
        if i == 0: 
            ax_dist.set_ylabel("Distance to $w^*$")
            ax_loss.set_ylabel("Loss Value")
        
        ax_dist.grid(True, which="both", alpha=0.3)
        ax_loss.grid(True, which="both", alpha=0.3)
        
        # Legend only on first row to save space
        ax_dist.legend(loc='best', fontsize=10)
        
    plt.suptitle(f"Robust Shift-Coupled Attack ({timestamp})", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"robust_benchmark_{timestamp}.png")
    print(f"\nDone. Plot saved.")


if __name__ == "__main__":
    run_experiment(seed=42)