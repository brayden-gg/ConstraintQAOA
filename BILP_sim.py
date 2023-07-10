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
import rules_based_mixer

p = 1 # layers of QAOA
trials = 50 # how many times to run optimizer
lam = 0 # penalty weight
k = 3


def get_S(n, k):
    S = np.zeros((1, n))
    Si = np.arange(n)
    for i in range(k):
        S[0, Si % k == i] = i + 1
    return S


# Objective function: maximize c⋅z
def cost(_, z): 
    return np.dot(z, c)

# subtract |S⋅z - b|^2 to minimize violating constraints
# def penalty(alpha, z):
#     diff = -b[alpha, 0] + np.dot(z, S[alpha, :])
#     return diff * diff

def constraint(alpha, z):
    return np.dot(z, S[alpha, :]) == b[alpha, 0]




c = np.array([1, 2, 3, 4, 3, 2]) # vector to optimize with
n = c.size
# constraints
S = get_S(n, k)

b = np.array([[4]])

nconstraints = S.shape[0]

# start in superposition of feasable states
subspace = [i for i in range(2**n) if constraint(0, helpers.int2bits(i, n))]
initial_state = np.zeros(2**n)
for state in subspace:
    initial_state[state] += 1
initial_state /= np.linalg.norm(initial_state)
    
# get hamiltonians for running QAOA
C = circuits.get_cost_hamiltonian(cost, range(n), n) #, penalty=penalty, lam=lam, nconstraints=nconstraints)
# B = circuits.get_custom_mixer_hamiltonian(n, coefs, paulis)
M = rules_based_mixer.get_minimal_mixer(n, k)
opt_fn = circuits.get_optimization_fn(n, p, C, M, initial_state)

assert helpers.preserves_subspace(lambda b: circuits.evolve(M, b, n).get_matrix(), subspace, n)

# optimize over angles to produce a circuit
F_max, gammas, betas = circuits.find_optimal_angles(opt_fn, trials, p)
circ = circuits.get_circuit(n, p, C, M, gammas, betas, entangle=False)
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

