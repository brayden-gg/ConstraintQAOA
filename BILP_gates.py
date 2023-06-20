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

# Objective function: maximize c⋅z
def cost(_, z): 
    return np.dot(z, c)


def constraint(alpha, z):
    return np.dot(z, S[alpha, :]) == b[alpha, 0]

p = 3 # layers of QAOA
trials = 3 # how many times to run optimizer
k = 3
n = 6

# reward
c = np.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3, 2, 3, 8, 4, 6, 2, 6, 4][:n]) # vector to optimize with
n = c.size

# constraints
S = get_S(n, k)
b = np.array([[3]])

nconstraints = S.shape[0]

initial_state = circuits.get_initial_state(constraint, n)
    
# get hamiltonians for running QAOA
C = circuits.get_cost_hamiltonian(cost, range(n), n)
opt_fn = gates.get_optimization_fn(n, k, c, p, initial_state, C)

# optimize over angles to produce a circuit
F_max, gammas, betas = circuits.find_optimal_angles(opt_fn, trials, p)
# print(f"max <C> = {-F_max}")

circ = gates.get_circuit(n, k, c, gammas, betas)
print("Circuit depth: " + str(circ.calculate_depth()))
# sanity check (slow)
# assert helpers.preserves_subspace_gates(lambda c, b: gates.add_UM(c, b, n, k), subspace, n)

# run state through our circuit
state = q.QuantumState(n)
state.load(initial_state)
circ.update_quantum_state(state)
f_obs = C.get_expectation_value(state).real
probabilities = np.abs(state.get_vector()) ** 2

# evaluate performance of produced states
overall_cost_fn = circuits.get_cost_fn(range(n), cost, nconstraints, constraint)
f_max, f_min, global_opt_state, global_opt_ind = evaluate.get_f_min_max(overall_cost_fn, n)
prob_of_max = evaluate.prob_of_f_max(probabilities, overall_cost_fn, f_max, n)
prob_min = evaluate.prob_of_f_max(probabilities, overall_cost_fn, 0, n)
approx_ratio = evaluate.approx_ratio(f_obs, f_max, f_min)

top_cands = probabilities.argsort()[::-1]

print("Top 10 most likely states:")

for cand in top_cands[:10]:
    bits = helpers.int2bits(cand, n)
    s = q.QuantumState(n)
    v = np.zeros(2 ** n)
    v[cand] = 1
    s.load(v)
    EV = C.get_expectation_value(s).real
    print(f"bitstring {bits} ({cand}) has value {overall_cost_fn(bits)} with probability {probabilities[cand]}, <C>: {EV}")


s = q.QuantumState(n)
v = np.zeros(2 ** n)
v[global_opt_ind] = 1
s.load(v)
EV = C.get_expectation_value(s).real
print(f"found global optimum {global_opt_state} with value {f_max} with probability {probabilities[global_opt_ind]}, <C>: {EV}")
print(f"p: {p}, approximation ratio: {approx_ratio}, P(f_max) = {prob_of_max.round(5)}, P(f_min) = {prob_min.round(5)}")
# print(f"{n},{p},{k},{approx_ratio},{prob_of_max.round(5)},{prob_min.round(5)}")

