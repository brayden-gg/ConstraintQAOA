import qulacs as q
import numpy as np
import scipy as sp
import scipy.linalg
import scipy.optimize
import networkx as nx
from matplotlib import pyplot as plt
from qulacsvis import circuit_drawer

import circuits
import gates
import evaluate
import helpers
import optimize


# Objective function: maximize c⋅z
def cost(_, z): 
    return np.dot(z, c)

def constraint(alpha, z):
    return np.dot(z, S[alpha, :]) == b[alpha, 0]

# hyperparameters
p = 1 # layers of QAOA
k = 3
n = 6

# reward function coefficients
c = np.ones(n)

# constraint
S = np.array([[1, 2, 3] * (n//k)]) # constraint equation coeffieicnts
b = np.array([[7]])

nconstraints = S.shape[0]
initial_state = circuits.get_initial_state(constraint, n)
    
# get hamiltonians for running QAOA
C = circuits.get_cost_hamiltonian(cost, range(n), n)
opt_fn = gates.get_optimization_fn(n, k, c, p, initial_state, C)

# set initial guess for gammas and betas
# gammas = np.array([1.543602762273278, 0.12964459636345538, 6.191372076992037])
# betas = np.array([2.7424737009188735, 3.19068682207466, 10.952967842072791])
gammas = np.zeros(p)
betas = np.zeros(p)
F_max = 0

print("running optimizers")
# use some combination of the next few lines with different numbers of trials to improve gammas and betas
F_max, gammas, betas = optimize.scipy_optimize(opt_fn, p, trials=2, x0=[*gammas, *betas], method="BFGS")
F_max, gammas, betas = optimize.alternating_minimization(opt_fn, trials=3, p=p, optimizer=optimize.grid_search, x0=[*gammas, *betas], res=30)
F_max, gammas, betas = optimize.scipy_optimize(opt_fn, p, trials=15, x0=[*gammas, *betas])
print("finished optimizing")

print("\nFinal result:")
print("gammas, betas:")
print("gammas = " + str(gammas.tolist()))
print("betas = " + str(betas.tolist()))
print(f"max <C> = {-F_max}")

circ = gates.get_circuit(n, k, c, gammas, betas)

print("Circuit depth: " + str(circ.calculate_depth()))
# circuit_drawer(circ, "mpl")
plt.show()
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

# print("Top 10 most likely states:")

# for cand in top_cands[:10]:
#     bits = helpers.int2bits(cand, n)
#     s = q.QuantumState(n)
#     v = np.zeros(2 ** n)
#     v[cand] = 1
#     s.load(v)
#     EV = C.get_expectation_value(s).real
#     print(f"bitstring {bits} ({cand}) has value {overall_cost_fn(bits)} with probability {probabilities[cand]}, <C>: {EV}")


s = q.QuantumState(n)
v = np.zeros(2 ** n)
v[global_opt_ind] = 1
s.load(v)
EV = C.get_expectation_value(s).real
global_opt_bitstring = "".join([str(e) for e in global_opt_state])
print(f"Classical algorithm: global max |{global_opt_bitstring}> with cost {f_max}")
print(f"Quantum simulation: (p = {p}), Approximation Ratio: {approx_ratio}, P(optimal) = {prob_of_max}")
# print(f"{n},{p},{k},{approx_ratio},{prob_of_max.round(5)},{prob_min.round(5)}")

