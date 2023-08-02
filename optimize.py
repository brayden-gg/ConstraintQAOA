import numpy as np
import scipy as sp
import scipy.optimize

# find optimal gammas and betas to maximize the optimization function
def scipy_optimize(opt_fn, p, x0=None, trials=50, method=None):
    gammas = None
    betas = None
    F_max = np.inf
    bounds = [(0, np.pi * 4)] * (2 * p) # can be any angle between 0 and 2pi

    for i in range(trials):
        if i == 0 and x0 is not None:
            F_max = opt_fn(x0)
            gammas = np.array(x0[:p])
            betas = np.array(x0[p:])
        else:
            x0 = np.random.rand(2*p) * np.pi * 4
            
        optim_res = sp.optimize.minimize(opt_fn, x0=x0, bounds=bounds, method=method)
        if optim_res["fun"] < F_max:
            F_max = optim_res["fun"]
            
            gammas, betas = optim_res["x"][:p] % (np.pi * 4), optim_res["x"][p:] % (np.pi * 4)
            print(f"New local opt: {F_max}")
            print("gammas = " + str(gammas.tolist()))
            print("betas = " + str(betas.tolist()))

    return F_max, gammas, betas

def alternating_minimization(opt_fn, trials, p, optimizer, x0=None, **kwargs):
    F_max = np.inf
    bounds = [(0, np.pi * 4), (0, np.pi * 4)] # can be any angle between 0 and 2pi

    if x0 is None:
        x0 = np.random.rand(2*p) * np.pi * 4
        
    optim_res = sp.optimize.minimize(opt_fn, x0=x0, bounds=bounds * p)
    F_max = optim_res["fun"]
    gammas, betas = optim_res["x"][:p], optim_res["x"][p:]
        
    for _ in range(trials):
        for i in range(p):
            def inner_opt(gamma_beta):
                g = gammas.copy()
                b = betas.copy()
                g[i] = gamma_beta[0]
                b[i] = gamma_beta[1]
                return opt_fn(g + b)
            
            inner_F_max, inner_gammas, inner_betas = optimizer(inner_opt, 1, **kwargs)
            if inner_F_max < F_max:
                F_max = inner_F_max
                gammas = inner_gammas
                betas = inner_betas

    optim_res = sp.optimize.minimize(opt_fn, x0=[*gammas, *betas], bounds=bounds * p)
    F_max = optim_res["fun"]
    gammas, betas = optim_res["x"][:p], optim_res["x"][p:]

    return F_max, gammas, betas

def grid_search(opt_fn, p, res=20):
    domain = np.linspace(0, np.pi*4, res)
    inputs = np.meshgrid(*([domain] * (2*p)),  indexing='ij')
    def func(*args):
        return opt_fn(args)
    vec = np.vectorize(func)
    Fs = vec(*inputs)
    loc = np.where(Fs == Fs.min())
    
    return Fs[loc][0], inputs[0][loc].tolist(), inputs[1][loc].tolist()

