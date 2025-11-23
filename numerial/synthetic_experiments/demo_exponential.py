from useful_functions_with_batch import *
from network_utils import *
import cupy as cp
from cupy_fuc import grad_with_batch_batched_gpu, PushPull_with_batch_batched_gpu

# --- Experiment Parameters ---
d = 10  # Dimension of the data features
L_total = 1440000  # Total number of data samples across all nodes
n = 16  # Number of nodes in the network
num_runs = 20  # Number of parallel simulation runs to average results over
device_id = "cuda:0"  # GPU device to use
rho = 1e-2  # Regularization parameter
lr = 5e-2  # Learning rate for the optimization algorithm
max_it = 3000  # Maximum number of iterations for the algorithm
bs = 200  # Batch size for stochastic gradient calculation


# --- GPU Setup ---
# Set the active GPU device
gpu_id_int = int(device_id.split(":")[1])
cp.cuda.Device(gpu_id_int).use()
print(f"Using GPU: {cp.cuda.Device(gpu_id_int).pci_bus_id}")

# --- Data Initialization (CPU) ---
# Generate global synthetic data on the CPU
# h_global_cpu: (L_total, d), y_global_cpu: (L_total,), x_opt_cpu: (1, d)
h_global_cpu, y_global_cpu, x_opt_cpu = init_global_data(d=d, L_total=L_total, seed=42)
print("h_global_cpu shape:", h_global_cpu.shape)
print("y_global_cpu shape:", y_global_cpu.shape)

# Distribute the global data among the n nodes
# h_tilde_cpu: (n, L_total/n, d), y_tilde_cpu: (n, L_total/n)
h_tilde_cpu, y_tilde_cpu = distribute_data(h=h_global_cpu, y=y_global_cpu, n=n)
print("h_tilde_cpu shape:", h_tilde_cpu.shape)
print("y_tilde_cpu shape:", y_tilde_cpu.shape)

# Initialize the starting parameters for each node for a single run
# init_x_cpu_single: (n, d)
init_x_cpu_single = init_x_func(n=n, d=d, seed=42)

# Generate the mixing matrices for the Push-Pull algorithm (exponential graph)
# A_cpu (row-stochastic), B_cpu (column-stochastic): (n, n)
A_cpu, B_cpu = generate_exp_matrices(n=n, seed=42)
print("CPU data is prepared.")


# --- Data Transfer to GPU ---
# Move the mixing matrices and distributed data to the selected GPU
A_gpu = cp.asarray(A_cpu)
B_gpu = cp.asarray(B_cpu)
h_tilde_gpu_nodes = cp.asarray(h_tilde_cpu)  # Shape: (n, L, d)
y_tilde_gpu_nodes = cp.asarray(y_tilde_cpu)  # Shape: (n, L)

# Replicate the initial parameters for each of the `num_runs` parallel simulations
# init_x_gpu_batched: (num_runs, n, d)
init_x_gpu_batched = cp.repeat(
    cp.asarray(init_x_cpu_single)[cp.newaxis, ...], num_runs, axis=0
)

print("Data moved to GPU.")
print("A_gpu shape:", A_gpu.shape)
print("h_tilde_gpu_nodes shape:", h_tilde_gpu_nodes.shape)
print("init_x_gpu_batched shape:", init_x_gpu_batched.shape)

# --- Run Batched Experiment on GPU ---
print(
    f"\nStarting batched experiment with n={n}, num_runs={num_runs} on GPU {device_id}"
)

# Execute the Push-Pull algorithm in a batched manner on the GPU
L1_avg_df = PushPull_with_batch_batched_gpu(
    A_gpu=A_gpu,
    B_gpu=B_gpu,
    init_x_gpu_batched=init_x_gpu_batched,
    h_data_nodes_gpu=h_tilde_gpu_nodes,
    y_data_nodes_gpu=y_tilde_gpu_nodes,
    grad_func_batched_gpu=grad_with_batch_batched_gpu,
    rho=rho,
    lr=lr,
    sigma_n=0,  # Manually set noise, here it is 0
    max_it=max_it,
    batch_size=bs,
    num_runs=num_runs,
)
print("\nL1_avg_df (from GPU batched execution):")
print(L1_avg_df.head())


# --- Save Results ---
# Define the output path for the results CSV file
output_path = (
    f"./EXP_out/EXP_avg_n={n}_gpu_batched.csv"
)
# Save the DataFrame containing the average gradient norm history
L1_avg_df.to_csv(output_path, index_label="iteration")
print(f"Average results saved to {output_path}")


# --- GPU Memory Cleanup ---
# Free up GPU memory pools
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()
