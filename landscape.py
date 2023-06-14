import qulacs as q
import numpy as np
import scipy as sp
import scipy.linalg
import scipy.optimize
import networkx as nx
from matplotlib import pyplot as plt
import json

import circuits
import evaluate
import helpers


def get_expectation_func(c, S, b, lam):
    n = len(c)
    nconstraints = S.shape[0]

    def C_i(i, z): 
        return z[i] * c[i]

    # subtract |S⋅z - b|^2 to minimize violating constraints
    def penalty(alpha, z):
        diff = -b[alpha, 0] + np.dot(z, S[alpha, :])
        return diff * diff
        
    # get hamiltonians for running QAOA
    C = circuits.get_cost_hamiltonian(C_i, range(n), n, penalty=penalty, lam=lam, nconstraints=nconstraints)
    B = circuits.get_mixer_hamiltonian(n)
    opt_fn = circuits.get_optimization_fn(n, p, C, B)

    def func(*args):
        return opt_fn(args)
    
    return np.vectorize(func), opt_fn
    

def get_success_rate(opt_fn, inputs, data, trials):

    gammas_ind, betas_ind = np.where(data - data.min() < tol*10)
    
    guess_gammas = inputs[0][gammas_ind, 0]
    guess_betas = inputs[1][0, betas_ind]
    global_opt = np.inf
    best_gammas = []
    best_betas = []
    for i in range(len(gammas_ind)):
        global_opt_res = sp.optimize.minimize(opt_fn, x0=[guess_gammas[i], guess_betas[i]], bounds=bounds)
        found_opt = global_opt_res["fun"]
        gamma, beta = global_opt_res["x"]

        if np.abs(global_opt - found_opt) < tol:
            best_gammas.append(gamma)
            best_betas.append(beta)

        if found_opt < global_opt:
            if np.abs(global_opt - found_opt) >= tol:
                best_gammas = [gamma]
                best_betas = [beta]
            global_opt = found_opt
        
    good_sols = 0
    for _ in range(trials):
        x0 = np.random.rand(2*p) * np.pi * 2
        local_opt = sp.optimize.minimize(opt_fn, x0=x0, bounds=bounds)["fun"]
        if  np.abs(local_opt - global_opt) < tol:
            good_sols += 1

    return good_sols/trials, best_gammas, best_betas, global_opt

p = 1 # layers of QAOA
bounds = [(0, np.pi * 2)] * (2 * p) # can be any angle between 0 and 2pi
trials = 100 # how many times to run optimizer




# constraints
#1:
# c = np.array([1, 2]) # vector to optimize with
# S = np.array([[1, 1]])
# b = np.array([[1]])
#2:
c = np.array([1, 1, 1])
S = np.array([[1, 1, 2]])
b = np.array([[2]])
#3:
# c = np.array([1, 1, 1])
# S = np.array([[1, 1, 2]])
# b = np.array([[1]])

tol = 0.001

resolution = 30
steps = 21
folder = "./results/BILP_penalty_3"

results = []

for lam in np.linspace(0, 2, steps):
    fig = plt.figure()
    f1_vec, f1 = get_expectation_func(c, S, b, lam)
    path = f"{folder}/heatmap/lam_{str(round(lam, 3)).zfill(5).replace('.', ',')}.png"
    domain = np.linspace(0, 2*np.pi, resolution)
    inputs = np.meshgrid(*([domain] * (2*p)),  indexing='ij')
    data = f1_vec(*inputs)

    prob, good_gammas, good_betas, global_opt = get_success_rate(f1, inputs, data, trials)

    plt.imshow(-data, vmin=-3, vmax=3, interpolation="bicubic")
    plt.colorbar()
    plt.title(f"<C> for lambda = {round(lam, 2)}")
    
    plt.xlabel("beta")
    plt.ylabel("gamma")
    

    results.append({
        "lambda": lam,
        "prob_opt": prob,
        "gammas": good_gammas,
        "betas": good_betas,
        "global_opt": -global_opt,
    })

    plt.scatter(np.array(good_betas) * resolution/(2*np.pi), np.array(good_gammas) * resolution/(2*np.pi), color="r")
    plt.savefig(path)
    plt.close()

with open(f"{folder}/landscape.json", "w") as outfile:
    json.dump(results, outfile)

# optimize over angles to produce a circuit
# F_max, gammas, betas = circuits.find_optimal_angles(opt_fn, trials, p)
# circ = circuits.get_circuit(n, p, C, B, gammas, betas)
# print(f"max <C> = {-F_max}")
