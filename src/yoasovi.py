import copy

def run_yoasovi(model, data, num_iterations=2000, lr=0.01, M_init=0.1, M_max=10.0):
    pyro.clear_param_store()
    guide = AutoDiagonalNormal(model)
    elbo_obj = Trace_ELBO(num_particles=1)
    
    guide(data) 
    params = pyro.get_param_store().get_all_param_names()
    torch_opt = torch.optim.Adam(pyro.get_param_store().parameters(), lr=lr)
    
    M_schedule = torch.linspace(M_init, M_max, num_iterations)
    
    elbo_history, time_history, evals_history = [], [], []
    start_time = time.perf_counter()
    
    with torch.no_grad():
        prev_elbo = -elbo_obj.loss(model, guide, data)
    
    grad_evals = 0
    
    for t in range(num_iterations):
        state_dict_backup = {k: v.clone() for k, v in pyro.get_param_store().named_parameters()}
        
        torch_opt.zero_grad()
        loss = elbo_obj.loss(model, guide, data)
        loss.backward()
        torch_opt.step()
        grad_evals += 1
        
        with torch.no_grad():
            new_elbo = -elbo_obj.loss(model, guide, data)
        
        T_t = abs(prev_elbo) / M_schedule[t] 
        diff = new_elbo - prev_elbo
        p_accept = min(1.0, torch.exp(torch.tensor(diff / T_t)).item())
        
        if torch.rand(1).item() < p_accept:
            prev_elbo = new_elbo
        else:
            for k, v in pyro.get_param_store().named_parameters():
                v.data.copy_(state_dict_backup[k].data)
                
        elbo_history.append(prev_elbo)
        time_history.append(time.perf_counter() - start_time)
        evals_history.append(grad_evals)
        
    return guide, elbo_history, time_history, evals_history