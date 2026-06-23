import time
import pyro
import pyro.optim as optim
from pyro.infer import SVI, TraceGraph_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal

def run_bbvi(model, *args, num_iterations=2000, lr=0.01, num_particles=500, **kwargs):
    """
    Standard Black Box Variational Inference using REINFORCE (score-function) gradients.
    Requires TraceGraph_ELBO to build the non-reparameterized computation graph.
    """
    pyro.clear_param_store()
    guide = AutoDiagonalNormal(model)
    adam = optim.Adam({"lr": lr})
    
    # TraceGraph_ELBO is critical here for score-function gradients
    svi = SVI(model, guide, adam, loss=TraceGraph_ELBO(num_particles=num_particles))
    
    elbo_history, time_history, evals_history = [], [], []
    start_time = time.perf_counter()
    grad_evals = 0
    
    for t in range(num_iterations):
        loss = svi.step(*args, **kwargs) 
        grad_evals += num_particles # Tracks the massive S compute cost
        
        elbo_history.append(-loss) 
        time_history.append(time.perf_counter() - start_time)
        evals_history.append(grad_evals) 
        
    return guide, elbo_history, time_history, evals_history