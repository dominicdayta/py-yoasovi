import time
import pyro
import pyro.optim as optim
import pyro.poutine as poutine
from pyro.infer import SVI, TraceGraph_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal

class ForceScoreFunction(poutine.messenger.Messenger):
    def _pyro_sample(self, msg):
        if msg.get("type") == "sample" and not msg.get("is_observed", False):
            dist = msg["fn"]
            
            class NonReparamDist(dist.__class__):
                @property
                def has_rsample(self):
                    return False
            
            dist.__class__ = NonReparamDist

def run_bbvi(model, *args, num_iterations=2000, lr=0.01, num_particles=500, **kwargs):
    """
    STRICT Black Box Variational Inference using REINFORCE (score-function) gradients.
    Explicitly disables reparameterization to force true BBVI variance.
    """
    pyro.clear_param_store()
    guide = AutoDiagonalNormal(model)
    adam = optim.Adam({"lr": lr})
    
    def strict_bbvi_guide(*args, **kwargs):
        with ForceScoreFunction():
            return guide(*args, **kwargs)
    
    svi = SVI(model, strict_bbvi_guide, adam, loss=TraceGraph_ELBO(num_particles=num_particles))
    
    elbo_history, time_history, evals_history = [], [], []
    start_time = time.perf_counter()
    grad_evals = 0
    
    for t in range(num_iterations):
        loss = svi.step(*args, **kwargs) 
        grad_evals += num_particles 
        
        elbo_history.append(-loss) 
        time_history.append(time.perf_counter() - start_time)
        evals_history.append(grad_evals) 
        
    return guide, elbo_history, time_history, evals_history