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

p = 1 # layers of QAOA
k = 3

n = 9

# reward
c = np.ones(n)
# c = np.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3, 2, 3, 8, 4, 6, 2, 6, 4][:n]) # vector to optimize with

# constraint
S = get_S(n, k)
b = np.array([[7]])

nconstraints = S.shape[0]
initial_state = circuits.get_initial_state(constraint, n)
    
# get hamiltonians for running QAOA
C = circuits.get_cost_hamiltonian(cost, range(n), n)
opt_fn = gates.get_optimization_fn(n, k, c, p, initial_state, C)

# optimize over angles to produce a circuit
gammas = [0, 10.744952256280623, 10.2088602668362, 10.305926727639305, 2.857936663144709, 8.642103069526074]
betas = [0, 0.17501806415319396, 0.09740271666347408, 12.311337022282833, 3.4831858746485564, 6.495813767349775]

F_max, gammas, betas = optimize.scipy_optimize(opt_fn, p, trials=15, x0=[*gammas, *betas], method="BFGS")

# F_max, gammas, betas = optimize.alternating_minimization(opt_fn, trials=3, p=p, optimizer=optimize.grid_search, x0=[*gammas, *betas], res=30)

# F_max, gammas, betas = optimize.scipy_optimize(opt_fn, p, trials=15, x0=[*gammas, *betas])
# print(F_max)

print("gammas, betas:")
print("gammas = " + str(gammas.tolist()))
print("betas = " + str(betas.tolist()))
# print(f"max <C> = {-F_max}")

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
print(f"found global optimum {global_opt_state} with value {f_max} with probability {probabilities[global_opt_ind]}, <C>: {EV}")
print(f"p: {p}, approximation ratio: {approx_ratio}, P(f_max) = {prob_of_max}, P(f_min) = {prob_min}")
# print(f"{n},{p},{k},{approx_ratio},{prob_of_max.round(5)},{prob_min.round(5)}")

