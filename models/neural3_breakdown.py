import os
import sys
import pickle
import argparse
import pandas as pd
import torch
import pyro
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.yoasovi import run_yoasovi
from src.evaluation import compute_rmse, compute_dic

from models.neural3 import BayesianNeuralNetwork3Layer

def main():
    parser = argparse.ArgumentParser(description="YOASOVI Hyperparameter Sensitivity Analysis")
    parser.add_argument("--data", type=str, default=None, help="Path to CSV data file.")
    parser.add_argument("--data", type=str, default=None, 
                        help="Path to CSV data file. Last column is assumed to be the target (y).")
    parser.add_argument("--has_header", action="store_true", 
                        help="Flag to indicate if the data file has a header row.")
    parser.add_argument("--normalize_data", action="store_true",
                        help="Flag to indicate if the data should be normalized.")
    parser.add_argument("--n_samples", type=int, default=500, help="Number of synthetic samples.")
    parser.add_argument("--num_iterations", type=int, default=2500, help="Number of VI iterations.")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Optimizer learning rate.")
    
    args = parser.parse_args()

    output_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'neural3_breakdown')
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

    # Consider three M-schedule scenarios
    scenarios = {
        "Dynamic_Annealing": {"M_init": 0.1, "M_max": 15.0},  # The intended schedule
        "Static_Too_High": {"M_init": 15.0, "M_max": 15.0},   # Cold system: overly greedy
        "Static_Too_Low": {"M_init": 0.1, "M_max": 0.1}       # Hot system: random walk
    }

    for scenario_name, params in scenarios.items():
        print(f"\n--- Running YOASOVI Scenario: {scenario_name} ---")
        pyro.set_rng_seed(42) 
        
        model = BayesianNeuralNetwork3Layer(in_features=in_features, hidden_dim=50)
        
        guide, elbo_hist, time_hist, evals_hist = run_yoasovi(
            model, X_train, y=y_train, 
            num_iterations=args.num_iterations, 
            lr=args.learning_rate, 
            M_init=params["M_init"], 
            M_max=params["M_max"]
        )
            
        print(f"Computing evaluation metrics for {scenario_name} on test set...")
        final_rmse, y_pred_test = compute_rmse(model, guide, X_test, y_test, num_samples=100)
        final_dic = compute_dic(model, guide, X_test, y_test, num_samples=100)
        print(f"Test RMSE: {final_rmse:.4f} | Test DIC: {final_dic:.2f}")
            
        results = {
            "scenario": scenario_name,
            "elbo": elbo_hist,
            "M_init": params["M_init"],
            "M_max": params["M_max"],
            "rmse": final_rmse,
            "dic": final_dic
        }
        
        dataset_name = os.path.splitext(os.path.basename(args.data))[0] if args.data else "synthetic"
        filename = os.path.join(output_dir, f"{scenario_name}_{dataset_name}_results.pkl")
        
        with open(filename, 'wb') as f:
            pickle.dump(results, f)

if __name__ == "__main__":
    main()