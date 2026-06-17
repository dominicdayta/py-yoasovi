import time
import pyro
import pyro.optim as optim
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal

def run_advi(model, *args, num_iterations=2000, lr=0.01, **kwargs):
    """
    Standard ADVI using Pyro's AutoGuide and SVI.
    *args and **kwargs are passed directly to the model and guide.
    """
    pyro.clear_param_store()
    guide = AutoDiagonalNormal(model)
    adam = optim.Adam({"lr": lr})
    svi = SVI(model, guide, adam, loss=Trace_ELBO(num_particles=1))
    
    elbo_history, time_history, evals_history = [], [], []
    start_time = time.perf_counter()
    
    for t in range(num_iterations):
        loss = svi.step(*args, **kwargs) 
        
        elbo_history.append(-loss) 
        time_history.append(time.perf_counter() - start_time)
        evals_history.append(t + 1)
        
    return guide, elbo_history, time_history, evals_history