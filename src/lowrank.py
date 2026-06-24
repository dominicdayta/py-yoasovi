import time
import pyro
import pyro.optim as optim
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoLowRankMultivariateNormal

def run_lowrank(model, *args, num_iterations=2000, lr=0.01, rank=10, **kwargs):
    """
    Modern Baseline: Low-Rank Multivariate Normal.
    Captures parameter correlations without the O(N^2) memory explosion of flows.
    """
    pyro.clear_param_store()
    
    guide = AutoLowRankMultivariateNormal(model, rank=rank)
    
    # ClippedAdam prevents early gradient explosions common in correlated guides
    adam = optim.ClippedAdam({"lr": lr, "clip_norm": 10.0})
    svi = SVI(model, guide, adam, loss=Trace_ELBO(num_particles=1))
    
    elbo_history, time_history, evals_history = [], [], []
    start_time = time.perf_counter()
    
    for t in range(num_iterations):
        loss = svi.step(*args, **kwargs) 
        
        elbo_history.append(-loss) 
        time_history.append(time.perf_counter() - start_time)
        evals_history.append(t + 1) 
        
    return guide, elbo_history, time_history, evals_history