import os
import sys
import pickle
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.advi import run_advi
from src.yoasovi import run_yoasovi
from src.qmcvi import run_qmcvi
from src.bbvi import run_bbvi
from src.iaf import run_iaf

from sklearn.model_selection import train_test_split
from src.evaluation import compute_rmse, compute_dic

methods = {
    "ADVI": run_advi,
    "IAF": run_iaf,
    "QMCVI": run_qmcvi,
    "YOASOVI": run_yoasovi,
    "BBVI_500": lambda *a, **kw: run_bbvi(*a, num_particles=500, **kw),
    "BBVI_1000": lambda *a, **kw: run_bbvi(*a, num_particles=1000, **kw)
}

class BayesianNeuralNetwork4Layer(PyroModule):
    def __init__(self, in_features, hidden_dim_1=50, hidden_dim_2=30):
        super().__init__()
        
        self.fc1 = PyroModule[nn.Linear](in_features, hidden_dim_1)
        self.fc2 = PyroModule[nn.Linear](hidden_dim_1, hidden_dim_1)
        self.fc3 = PyroModule[nn.Linear](hidden_dim_1, hidden_dim_2)
        self.fc4 = PyroModule[nn.Linear](hidden_dim_2, 1)

        self.fc1.weight = PyroSample(dist.Normal(0., 1.).expand([hidden_dim_1, in_features]).to_event(2))
        self.fc1.bias = PyroSample(dist.Normal(0., 1.).expand([hidden_dim_1]).to_event(1))
        
        self.fc2.weight = PyroSample(dist.Normal(0., 1.).expand([hidden_dim_1, hidden_dim_1]).to_event(2))
        self.fc2.bias = PyroSample(dist.Normal(0., 1.).expand([hidden_dim_1]).to_event(1))
        
        self.fc3.weight = PyroSample(dist.Normal(0., 1.).expand([hidden_dim_2, hidden_dim_1]).to_event(2))
        self.fc3.bias = PyroSample(dist.Normal(0., 1.).expand([hidden_dim_2]).to_event(1))
        
        self.fc4.weight = PyroSample(dist.Normal(0., 1.).expand([1, hidden_dim_2]).to_event(2))
        self.fc4.bias = PyroSample(dist.Normal(0., 1.).expand([1]).to_event(1))

    def forward(self, x, y=None):
        hidden1 = F.leaky_relu(self.fc1(x))
        hidden2 = F.leaky_relu(self.fc2(hidden1))
        hidden3 = F.leaky_relu(self.fc3(hidden2))

        mu = self.fc4(hidden3).squeeze(-1)
        sigma = pyro.sample("sigma", dist.Uniform(0., 10.))
        
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Normal(mu, sigma), obs=y)
            
        return mu

def main():
    parser = argparse.ArgumentParser(description="Run VI Benchmarks on 4-Layer BNN")
    parser.add_argument("--data", type=str, default=None, 
                        help="Path to CSV data file. Last column is assumed to be the target (y).")
    parser.add_argument("--has_header", action="store_true", 
                        help="Flag to indicate if the data file has a header row.")
    parser.add_argument("--normalize_data", action="store_true",
                        help="Flag to indicate if the data should be normalized.")
    parser.add_argument("--n_samples", type=int, default=500, 
                        help="Number of synthetic samples if no data file is provided.")
    parser.add_argument("--num_iterations", type=int, default=2500, 
                        help="Number of VI iterations.")
    parser.add_argument("--learning_rate", type=float, default=0.01, 
                        help="Optimizer learning rate.")
    parser.add_argument("--M_init", type=float, default=0.1, 
                        help="Initial M (inverse temperature) for YOASOVI.")
    parser.add_argument("--M_max", type=float, default=15.0, 
                        help="Maximum M (inverse temperature) for YOASOVI.")
    
    args = parser.parse_args()

    # Setup results directory
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'neural4')
    os.makedirs(output_dir, exist_ok=True)
    
    if args.data is not None:
        print(f"Loading data from {args.data}...")
        header_val = 0 if args.has_header else None
        df = pd.read_csv(args.data, header=header_val)
        # Assume last column is y, everything else is X
        X_df = df.iloc[:, :-1].copy()
        y_df = df.iloc[:, -1].copy()

        if args.normalize_data:
            # Min-max normalize each X column independently to [0, 1]
            X_df = (X_df - X_df.min()) / (X_df.max() - X_df.min())
            y_df = (y_df - y_df.min()) / (y_df.max() - y_df.min())

        X_numpy = X_df.values
        y_numpy = y_df.values

    else:
        print(f"Generating {args.n_samples} synthetic data samples...")
        torch.manual_seed(42)
        in_features = 10
        X_numpy = torch.randn(args.n_samples, in_features).numpy()
        y_numpy = (torch.sin(torch.tensor(X_numpy[:, 0])) + 
                   2 * torch.tensor(X_numpy[:, 1])**2 - 
                   1.5 * torch.tensor(X_numpy[:, 2]) + 
                   torch.randn(args.n_samples) * 0.5).numpy()
    
    # 80/20 Train-Test Split
    X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
        X_numpy, y_numpy, test_size=0.2, random_state=42
    )
    
    # Convert to PyTorch Tensors
    X_train = torch.tensor(X_train_np, dtype=torch.float32)
    y_train = torch.tensor(y_train_np, dtype=torch.float32)
    X_test = torch.tensor(X_test_np, dtype=torch.float32)
    y_test = torch.tensor(y_test_np, dtype=torch.float32)
    
    in_features = X_train.shape[1]

    for method_name, method_func in methods.items():
        print(f"--- Running {method_name} ---")
        pyro.set_rng_seed(42)
        
        model = BayesianNeuralNetwork4Layer(in_features=in_features, hidden_dim_1=50, hidden_dim_2=30)
        
        if method_name == "YOASOVI":
            guide, elbo_hist, time_hist, evals_hist = method_func(
                model, X_train, y=y_train, 
                num_iterations=args.num_iterations, 
                lr=args.learning_rate, 
                M_init=args.M_init, 
                M_max=args.M_max
            )
        else:
            guide, elbo_hist, time_hist, evals_hist = method_func(
                model, X_train, y=y_train, 
                num_iterations=args.num_iterations, 
                lr=args.learning_rate
            )
            
        # Compute post-training metrics on test set and print to console
        print(f"Computing evaluation metrics for {method_name} on test set...")
        final_rmse, y_pred_test = compute_rmse(model, guide, X_test, y_test, num_samples=100)
        final_dic = compute_dic(model, guide, X_test, y_test, num_samples=100)
        final_elbo = elbo_hist[-1]
        final_time = time_hist[-1]
        print(f"Final ELBO: {final_elbo:.2f} | Time: {final_time:.2f}s | Test RMSE: {final_rmse:.4f} | Test DIC: {final_dic:.2f}")
            
        results = {
            "elbo": elbo_hist,
            "time": time_hist,
            "evals": evals_hist,
            "rmse": final_rmse,
            "dic": final_dic,
            "y_true": y_test_np,
            "y_pred": y_pred_test
        }
        
        dataset_name = os.path.splitext(os.path.basename(args.data))[0] if args.data else "synthetic"
        filename = os.path.join(output_dir, f"{method_name.lower()}_{dataset_name}_results.pkl")
        
        with open(filename, 'wb') as f:
            pickle.dump(results, f)
            
        print(f"Saved {method_name} results to {filename}\n")

if __name__ == "__main__":
    main()