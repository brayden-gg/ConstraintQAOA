import qulacs as q
import numpy as np
import scipy as sp
import scipy.linalg
import scipy.optimize
import networkx as nx
from matplotlib import pyplot as plt

import circuits
import gates
import evaluate
import helpers

def get_S(n, k):
    S = np.zeros((1, n))
    Si = np.arange(n)
    for i in range(k):
        S[0, Si % k == i] = i + 1
    return S

def get_expectation_func(c, b, n, k, p):
    # Objective function: maximize c⋅z
    def cost(_, z): 
        return np.dot(z, c)


    def constraint(alpha, z):
        return np.dot(z, S[alpha, :]) == b[alpha, 0]


    initial_state = circuits.get_initial_state(constraint, n)
        
    # get hamiltonians for running QAOA
    C = circuits.get_cost_hamiltonian(cost, range(n), n)
    opt_fn = gates.get_optimization_fn(n, k, c, p, initial_state, C)
    nconstraints = S.shape[0]

    overall_cost_fn = circuits.get_cost_fn(range(n), cost, nconstraints, constraint)
    f_max, f_min, global_opt_state, global_opt_ind = evaluate.get_f_min_max(overall_cost_fn, n)
    
    def get_metrics(gamma, beta):
        circ = gates.get_circuit(n, k, c, [gamma], [beta])
        state = q.QuantumState(n)
        state.load(initial_state)
        circ.update_quantum_state(state)
        f_obs = C.get_expectation_value(state).real
        probabilities = np.abs(state.get_vector()) ** 2

        # evaluate performance of produced states
        prob_of_max = evaluate.prob_of_f_max(probabilities, overall_cost_fn, f_max, n)
        approx_ratio = evaluate.approx_ratio(f_obs, f_max, f_min)

        return approx_ratio, prob_of_max
    # def p_max(*args):
    #     circ = gates.get_circuit(n, k, c, gammas, betas)

    
    return np.vectorize(get_metrics)




pi = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3, 2, 3, 8, 4, 6, 2, 6, 4, 3, 3, 8, 3, 2, 7, 9, 5, 0, 2, 8, 8, 4, 1, 9, 7, 1, 0, 5]



p = 1
bounds = [(0, np.pi * 2)] * (2 * p) # can be any angle between 0 and 2pi
# tol = 0.001
resolution = 50
steps = 21
folder = "./results/Subspace"

offset = 6
n = 6
k = 3
b0 = 7
p = 1

for n in [6, 9, 12]:
    for offset in [0, 1, 2, 6, 12]:
        for b0 in range(1, 12):
            c = np.array(pi[offset:n+offset]) # vector to optimize with
            # constraints
            S = get_S(n, k)
            b = np.array([[b0]])


            metrics_fn = get_expectation_func(c, b, n, k, p)
            domain = np.linspace(0, 2*np.pi, resolution)
            inputs = np.meshgrid(*([domain] * (2*p)),  indexing='ij')

            approx_ratio, P_max = metrics_fn(*inputs)

            fig = plt.figure()
            plt.imshow(approx_ratio, interpolation="bicubic", extent=[0,np.pi*2,0,np.pi*2])
            plt.colorbar()
            plt.title(f"Approximation Ratio for n={n}, k={k}, p={p}, b={b[0, 0]}, offset={offset}")
            plt.xlabel("beta")
            plt.ylabel("gamma")
            plt.savefig(f"{folder}/n={n}_offset={offset}_G2/heatmap_k={k}_b={b[0, 0]}_o={offset}_approx_ratio.png")
            plt.close()

            fig = plt.figure()
            plt.imshow(P_max, interpolation="bicubic", extent=[0,np.pi*2,0,np.pi*2])
            plt.colorbar()
            plt.title(f"Probability of optimal solution for n={n}, k={k}, p={p}, b={b[0, 0]}, offset={offset}")
            plt.xlabel("beta")
            plt.ylabel("gamma")
            plt.savefig(f"{folder}/n={n}_offset={offset}_G2/heatmap_k={k}_b={b[0, 0]}_o={offset}_Pmax.png")
            plt.close()
