import torch
import pyro.poutine as poutine
from pyro.infer import Predictive

def compute_rmse(model, guide, X, y, num_samples=100):
    """
    Computes predictive RMSE and returns the raw predictions.
    """
    predictive = Predictive(model, guide=guide, num_samples=num_samples, return_sites=["obs"])
    
    with torch.no_grad():
        samples = predictive(X, y=None)
        
    y_pred = samples["obs"].mean(dim=0)
    rmse = torch.sqrt(torch.mean((y_pred - y)**2))
    
    # Return both the scalar RMSE and the numpy array of predictions
    return rmse.item(), y_pred.cpu().numpy()

def compute_dic(model, guide, X, y, num_samples=100):
    """
    Computes the Deviance Information Criterion (DIC).
    """
    deviances = []
    
    with torch.no_grad():
        # 1. Compute Expected Deviance (D_bar) via Monte Carlo sampling
        for _ in range(num_samples):
            guide_trace = poutine.trace(guide).get_trace(X, y)
            model_trace = poutine.trace(poutine.replay(model, trace=guide_trace)).get_trace(X, y)
            
            log_like = model_trace.nodes["obs"]["fn"].log_prob(model_trace.nodes["obs"]["value"]).sum()
            deviances.append(-2 * log_like)
            
        D_bar = torch.stack(deviances).mean().item()
        
        # 2. Extract Posterior Mean Parameters (theta_bar)
        try:
            # Fast, exact method for AutoDiagonalNormal (ADVI, YOASOVI, QMCVI)
            mean_params = guide.median(X, y)
        except NotImplementedError:
            # Fallback for Normalizing Flows (IAF) which lack closed-form medians
            # We draw samples from the guide and calculate the empirical mean
            sampled_params = {}
            for _ in range(num_samples):
                g_trace = poutine.trace(guide).get_trace(X, y)
                for name, node in g_trace.nodes.items():
                    if node["type"] == "sample" and not node["is_observed"]:
                        if name not in sampled_params:
                            sampled_params[name] = []
                        sampled_params[name].append(node["value"])
            
            # Average across the samples for each parameter
            mean_params = {name: torch.stack(vals).mean(dim=0) for name, vals in sampled_params.items()}
            
        # 3. Compute Deviance at the Posterior Mean (D_hat)
        # Condition the model strictly on these mean parameters
        model_trace_mean = poutine.trace(poutine.condition(model, data=mean_params)).get_trace(X, y)
        log_like_mean = model_trace_mean.nodes["obs"]["fn"].log_prob(model_trace_mean.nodes["obs"]["value"]).sum()
        
        D_hat = -2 * log_like_mean.item()
        
    # 4. Calculate Effective Number of Parameters and DIC
    pD = D_bar - D_hat
    dic = D_bar + pD
    
    return dic