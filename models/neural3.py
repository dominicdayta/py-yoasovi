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

class BayesianNeuralNetwork3Layer(PyroModule):
    def __init__(self, in_features, hidden_dim=50):
        super().__init__()
        
        self.fc1 = PyroModule[nn.Linear](in_features, hidden_dim)
        self.fc2 = PyroModule[nn.Linear](hidden_dim, hidden_dim)
        self.fc3 = PyroModule[nn.Linear](hidden_dim, 1)

        self.fc1.weight = PyroSample(dist.Normal(0., 1.).expand([hidden_dim, in_features]).to_event(2))
        self.fc1.bias = PyroSample(dist.Normal(0., 1.).expand([hidden_dim]).to_event(1))
        
        self.fc2.weight = PyroSample(dist.Normal(0., 1.).expand([hidden_dim, hidden_dim]).to_event(2))
        self.fc2.bias = PyroSample(dist.Normal(0., 1.).expand([hidden_dim]).to_event(1))
        
        self.fc3.weight = PyroSample(dist.Normal(0., 1.).expand([1, hidden_dim]).to_event(2))
        self.fc3.bias = PyroSample(dist.Normal(0., 1.).expand([1]).to_event(1))

    def forward(self, x, y=None):
        hidden1 = F.leaky_relu(self.fc1(x))
        hidden2 = F.leaky_relu(self.fc2(hidden1))
        
        mu = self.fc3(hidden2).squeeze(-1)
        sigma = pyro.sample("sigma", dist.Uniform(0., 10.))
        
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Normal(mu, sigma), obs=y)
            
        return mu

def main():
    parser = argparse.ArgumentParser(description="Run VI Benchmarks on 3-Layer BNN")
    parser.add_argument("--data", type=str, default=None, 
                        help="Path to CSV data file. Last column is assumed to be the target (y).")
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
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'neural3')
    os.makedirs(output_dir, exist_ok=True)
    
    if args.data is not None:
        print(f"Loading data from {args.data}...")
        df = pd.read_csv(args.data)
        # Assume last column is y, everything else is X
        X_numpy = df.iloc[:, :-1].values
        y_numpy = df.iloc[:, -1].values
        
        X_train = torch.tensor(X_numpy, dtype=torch.float32)
        y_train = torch.tensor(y_numpy, dtype=torch.float32)
        in_features = X_train.shape[1]
    else:
        print(f"Generating {args.n_samples} synthetic data samples...")
        torch.manual_seed(42)
        in_features = 10
        X_train = torch.randn(args.n_samples, in_features)
        y_train = torch.sin(X_train[:, 0]) + 2 * X_train[:, 1]**2 - 1.5 * X_train[:, 2] + torch.randn(args.n_samples) * 0.5

    methods = {
        "ADVI": run_advi,
        "QMCVI": run_qmcvi,
        "YOASOVI": run_yoasovi
    }

    for method_name, method_func in methods.items():
        print(f"--- Running {method_name} ---")
        pyro.set_rng_seed(42)
        
        model = BayesianNeuralNetwork3Layer(in_features=in_features, hidden_dim=50)
        
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
            
        results = {
            "elbo": elbo_hist,
            "time": time_hist,
            "evals": evals_hist
        }
        
        dataset_name = os.path.splitext(os.path.basename(args.data))[0] if args.data else "synthetic"
        filename = os.path.join(output_dir, f"{method_name.lower()}_{dataset_name}_results.pkl")
        
        with open(filename, 'wb') as f:
            pickle.dump(results, f)
            
        print(f"Saved {method_name} results to {filename}\n")

if __name__ == "__main__":
    main()