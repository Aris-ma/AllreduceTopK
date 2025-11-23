import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from training import train_track_grad_norm_with_hetero
from utils import show_row, generate_grid_matrices

lr = 3e-2
num_epochs = 100
bs = 128
alpha = 0.9
use_hetero=True
remark="new"
device = "cuda:0"
root = "/home/lg/PushPull/output"

def main():
    n=4
    A, B = generate_grid_matrices(n = n, seed=48)
    show_row(A)
    print(A.shape)
    for i in range(1):
        df = train_track_grad_norm_with_hetero(
            algorithm="PushPull",
            lr=lr,
            A=A,
            B=B,
            dataset_name="MNIST",
            batch_size=bs,
            num_epochs=20,
            remark=remark,
            alpha = alpha,
            root = root,
            use_hetero=use_hetero,
            device=device,
            seed = i+2
        )
        
        if i == 0:
            df_sum = df
            sum = 1
        else:
            df_sum = df_sum+df
            sum = sum + 1
        df_output = df_sum/sum
        df_output.to_csv(f"/home/lg/PushPull/output/repeated_grid_mnist_n={n}_lr={lr}.csv")

if __name__ == "__main__":
    main()
    print("Done")
    print("Output saved to /home/lg/PushPull/output/repeated_grid_mnist_n={n}_lr={lr}.csv")