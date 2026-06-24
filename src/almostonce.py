import time
import torch
import pyro
from pyro.infer import Trace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal

def run_almostonce(model, *args, num_iterations=2000, lr=0.01, M_init=0.1, M_max=10.0, num_particles=1, **kwargs):
    """
    EXPERIMENTAL: "You-Only-Accept-Almost-One" 
    
    A demonstration script testing the YOASOVI algorithm with S >= 1 samples.
    Increasing `num_particles` reduces the stochastic variance of the gradient.
    In the context of Tempered Stochastic Line Search, this reduction in variance
    often prematurely truncates the exploration phase, demonstrating why the S=1 
    estimator is mathematically optimal for escaping local minima.
    """
    pyro.clear_param_store()
    guide = AutoDiagonalNormal(model)
    
    # Trace_ELBO now scales its sampling based on the experimental parameter
    elbo_obj = Trace_ELBO(num_particles=num_particles)
    
    guide(*args, **kwargs) 
    
    optim_params = [unconstrained for name, unconstrained in pyro.get_param_store().named_parameters()]
    torch_opt = torch.optim.Adam(optim_params, lr=lr)
    
    M_schedule = torch.linspace(M_init, M_max, num_iterations)
    
    elbo_history, time_history, evals_history = [], [], []
    start_time = time.perf_counter()
    
    prev_elbo = -elbo_obj.loss(model, guide, *args, **kwargs)
    
    grad_evals = 0
    
    for t in range(num_iterations):
        state_dict_backup = {
            name: unconstrained.data.clone() 
            for name, unconstrained in pyro.get_param_store().named_parameters()
        }
        
        torch_opt.zero_grad()
        loss = elbo_obj.differentiable_loss(model, guide, *args, **kwargs)
        loss.backward()
        torch_opt.step()
        
        # Track the increased computational cost of drawing multiple samples
        grad_evals += num_particles 
        
        new_elbo = -elbo_obj.loss(model, guide, *args, **kwargs)
        
        T_t = abs(prev_elbo) / M_schedule[t] 
        diff = new_elbo - prev_elbo
        
        p_accept = min(1.0, torch.exp(diff / T_t).item())
        
        if torch.rand(1).item() < p_accept:
            prev_elbo = new_elbo
        else:
            current_unconstrained_params = dict(pyro.get_param_store().named_parameters())
            for name, saved_tensor in state_dict_backup.items():
                current_unconstrained_params[name].data.copy_(saved_tensor)
                
        elbo_history.append(prev_elbo)
        time_history.append(time.perf_counter() - start_time)
        evals_history.append(grad_evals)
        
    return guide, elbo_history, time_history, evals_history