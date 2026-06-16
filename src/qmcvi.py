from pyro.distributions import Normal
from torch.quasirandom import SobolEngine

class QMCAutoDiagonalNormal(AutoDiagonalNormal):
    """
    Custom guide that injects Quasi-Monte Carlo (Sobol) samples 
    into the reparameterization trick.
    """
    def __init__(self, model):
        super().__init__(model)
        self.sobol = None

    def _sample_latent(self, *args, **kwargs):
        loc, scale = self._get_loc_and_scale()
        
        if self.sobol is None:
            latent_dim = loc.shape[-1]
            self.sobol = SobolEngine(dimension=latent_dim, scramble=True)
            
        u_sobol = self.sobol.draw(10).to(loc.device)
        
        z_qmc = torch.distributions.Normal(0, 1).icdf(u_sobol)
        
        latent = loc + scale * z_qmc
        return latent

def run_qmcvi(model, data, num_iterations=2000, lr=0.01):
    pyro.clear_param_store()
    guide = QMCAutoDiagonalNormal(model)
    adam = optim.Adam({"lr": lr})
    svi = SVI(model, guide, adam, loss=Trace_ELBO(num_particles=10))
    
    elbo_history, time_history, evals_history = [], [], []
    start_time = time.perf_counter()
    grad_evals = 0
    
    for t in range(num_iterations):
        loss = svi.step(data)
        grad_evals += 10
        
        elbo_history.append(-loss)
        time_history.append(time.perf_counter() - start_time)
        evals_history.append(grad_evals)
        
    return guide, elbo_history, time_history, evals_history