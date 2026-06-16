import torch
import pyro
import pyro.optim as optim
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal

import time

def run_advi(model, data, num_iterations=2000, lr=0.01):
    pyro.clear_param_store()
    guide = AutoDiagonalNormal(model)
    adam = optim.Adam({"lr": lr})
    svi = SVI(model, guide, adam, loss=Trace_ELBO(num_particles=1))
    
    elbo_history, time_history, evals_history = [], [], []
    start_time = time.perf_counter()
    
    for t in range(num_iterations):
        loss = svi.step(data) 
        
        elbo_history.append(-loss) 
        time_history.append(time.perf_counter() - start_time)
        evals_history.append(t + 1) # 1 eval per iteration
        
    return guide, elbo_history, time_history, evals_history