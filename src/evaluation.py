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
    
    return rmse.item(), y_pred.cpu().numpy()

def compute_dic(model, guide, X, y, num_samples=100):
    """
    Computes the Deviance Information Criterion (DIC).
    """
    deviances = []
    
    with torch.no_grad():
        # Compute Expected Deviance (D_bar) via Monte Carlo sampling
        for _ in range(num_samples):
            guide_trace = poutine.trace(guide).get_trace(X, y)

            # Replay the model using the parameters sampled by the guide
            model_trace = poutine.trace(poutine.replay(model, trace=guide_trace)).get_trace(X, y)
            log_like = model_trace.nodes["obs"]["fn"].log_prob(model_trace.nodes["obs"]["value"]).sum()
            deviances.append(-2 * log_like)
            
        D_bar = torch.stack(deviances).mean().item()
        
        # Compute Deviance at the Posterior Mean (D_hat)
        mean_params = guide.median(X, y)
        model_trace_mean = poutine.trace(poutine.condition(model, data=mean_params)).get_trace(X, y)
        log_like_mean = model_trace_mean.nodes["obs"]["fn"].log_prob(model_trace_mean.nodes["obs"]["value"]).sum()
        
        D_hat = -2 * log_like_mean.item()
        
    pD = D_bar - D_hat
    dic = D_bar + pD
    
    return dic