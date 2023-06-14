import qulacs as q
import numpy as np
import scipy as sp
import scipy.linalg
import scipy.optimize
import networkx as nx
from matplotlib import pyplot as plt

import circuits
import evaluate
import helpers

p = 1 # layers of QAOA
trials = 8 # how many times to run optimizer
lam = 0 # penalty weight

c = np.array([1, 2, 3, 4]) # vector to optimize with

# constraints
S = np.array([
    [1, 2, 1, 2],
    # [1, 2, 1, 0, 0, 0],
    # [0, 0, 0, 1, 2, 1],
])

b = np.array([[2]])

nconstraints = S.shape[0]
n = len(c)


# Objective function: maximize c⋅z
def cost(_, z): 
    return np.dot(z, c)

# subtract |S⋅z - b|^2 to minimize violating constraints
# def penalty(alpha, z):
#     diff = -b[alpha, 0] + np.dot(z, S[alpha, :])
#     return diff * diff

def constraint(alpha, z):
    return np.dot(z, S[alpha, :]) == b[alpha, 0]



coefs = [(0.25+0j), (0.25+0j), (0.25+0j), (0.25+0j), (-0.125+0j), (0.125+0j), (0.25+0j), (0.125+0j), (0.375+0j), (0.25+0j), (0.25+0j), (-0.25+0j), (0.25+0j), (0.25+0j), (-0.125+0j), (0.125+0j), (0.25+0j), (-0.25+0j), (0.125+0j), (0.375+0j), (0.125+0j), (-0.125+0j), (0.125+0j), (-0.125+0j), (-0.125+0j), (0.125+0j), (-0.125+0j), (0.125+0j)]
paulis = ['X 0 X 1 X 2 X 3', 'X 0 X 1 X 2', 'X 0 Y 1 X 2 Y 3', 'X 0 Y 1 Y 2', 'X 0 Z 1 X 2 Z 3', 'X 0 Z 1 X 2', 'X 0 X 2 X 3', 'X 0 X 2 Z 3', 'X 0 X 2', 'X 0 Y 2 Y 3', 'Y 0 X 1 Y 2 X 3', 'Y 0 X 1 Y 2', 'Y 0 Y 1 X 2', 'Y 0 Y 1 Y 2 Y 3', 'Y 0 Z 1 Y 2 Z 3', 'Y 0 Z 1 Y 2', 'Y 0 X 2 Y 3', 'Y 0 Y 2 X 3', 'Y 0 Y 2 Z 3', 'Y 0 Y 2', 'Z 0 X 1 Z 2 X 3', 'Z 0 X 1 X 3', 'Z 0 Y 1 Z 2 Y 3', 'Z 0 Y 1 Y 3', 'X 1 Z 2 X 3', 'X 1 X 3', 'Y 1 Z 2 Y 3', 'Y 1 Y 3']
perm = np.arange(2**n)

# coefs = [(0.25+0j), (0.25+0j), (-0.25+0j), (-0.25+0j), (-0.25+0j), (0.25+0j), (-0.25+0j), (0.25+0j), (0.25+0j), (0.25+0j), (0.25+0j), (-0.25+0j), (-0.25+0j), (-0.25+0j), (-0.25+0j), (0.25+0j), (0.25+0j), (0.25+0j), (-0.25+0j), (0.25+0j)]
# paulis = ['X 0 X 1 X 2 X 3', 'X 0 X 1 X 2', 'X 0 X 1 Y 2 Y 3', 'X 0 Y 1 Y 2', 'X 0 Z 1 Z 3', 'X 0 Z 1', 'X 0 Z 3', 'X 0', 'Y 0 X 1 Y 2 Z 3', 'Y 0 Y 1 X 2 X 3', 'Y 0 Y 1 X 2 Z 3', 'Y 0 Y 1 Y 2 Y 3', 'Z 0 X 1 X 2 Z 3', 'Z 0 Y 1 Y 2 Z 3', 'Z 0 Z 1 X 3', 'Z 0 X 3', 'X 1 X 2', 'Y 1 Y 2', 'Z 1 X 3', 'X 3']
# perm = np.array([11, 3, 9, 12, 5, 8, 13, 14, 15, 2, 1, 7, 10, 6, 4, 0])

invperm = np.argsort(perm)

subspace = [i for i in range(2**n) if constraint(0, helpers.int2bits(invperm[i], n))]
initial_state = np.zeros(2**n)
for state in subspace:
    initial_state[perm[state]] += 1
initial_state /= np.linalg.norm(initial_state)
    
# get hamiltonians for running QAOA
C = circuits.get_cost_hamiltonian(cost, range(n), n) #, penalty=penalty, lam=lam, nconstraints=nconstraints)
B = circuits.get_custom_mixer_hamiltonian(n, coefs, paulis)
opt_fn = circuits.get_optimization_fn(n, p, C, B, initial_state)

# optimize over angles to produce a circuit
F_max, gammas, betas = circuits.find_optimal_angles(opt_fn, trials, p)
circ = circuits.get_circuit(n, p, C, B, gammas, betas, entangle=False)
print(f"max <C> = {-F_max}")

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
prob_unfeasable = evaluate.prob_of_f_max(probabilities, overall_cost_fn, 0, n)
approx_ratio = evaluate.approx_ratio(f_obs, f_max, f_min)

top_cands = probabilities.argsort()[::-1]
# top_probs = np.sort(probabilities.sort)

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
print(f"p: {p}, lambda: {lam}, approximation ratio: {approx_ratio}, P(f_max) = {prob_of_max}, P(f_min) = {prob_unfeasable}")

