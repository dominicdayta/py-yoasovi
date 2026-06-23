import time
import pyro
import pyro.optim as optim
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoIAFNormal

def run_iaf(model, *args, num_iterations=2000, lr=0.01, **kwargs):
    """
    Modern Baseline: Inverse Autoregressive Flows (IAF).
    Uses Normalizing Flows to create a highly expressive, non-mean-field posterior.
    """
    pyro.clear_param_store()
    
    # AutoIAFNormal wraps the model in a normalizing flow
    guide = AutoIAFNormal(model)
    adam = optim.Adam({"lr": lr})
    
    # Flows still use standard reparameterized ELBO, so Trace_ELBO is fine
    svi = SVI(model, guide, adam, loss=Trace_ELBO(num_particles=1))
    
    elbo_history, time_history, evals_history = [], [], []
    start_time = time.perf_counter()
    
    for t in range(num_iterations):
        loss = svi.step(*args, **kwargs) 
        
        elbo_history.append(-loss) 
        time_history.append(time.perf_counter() - start_time)
        evals_history.append(t + 1) 
        
    return guide, elbo_history, time_history, evals_history