import qulacs as q
import numpy as np
import scipy as sp
import scipy.linalg
import scipy.optimize
import networkx as nx
import qulacs_core
from matplotlib import pyplot as plt

import circuits
import evaluate
import helpers


n = 2 # number of vertices
k = 1 # degree of graph
p = 1 # layers of QAOA
trials = 50 # how many times to run optimizer

# Objective function: returns true iff an edge is not monochromayic
def cost(edges, state):
    res = 0
    for i, j in edges:
        # XOR them together (i + j - 2ij)
        res += state[i] + state[j] - state[i] * state[j] * 2
    return res

# start with random k-regular graph
G = nx.random_regular_graph(k, n)
edges = list(G.edges)

# get hamiltonians for running QAOA
C = circuits.get_cost_hamiltonian(cost, edges, n)
B = circuits.get_mixer_hamiltonian(n)
opt_fn = circuits.get_optimization_fn(n, p, C, B)

# optimize over angles to produce a circuit
F_max, gammas, betas = circuits.find_optimal_angles(opt_fn, trials, p)
circ = circuits.get_circuit(n, p, C, B, gammas, betas)
print(f"F_max: {-F_max}")

# run state through our circuit
state = q.QuantumState(n)
state.set_zero_state()
circ.update_quantum_state(state)
f_obs = C.get_expectation_value(state).real
probabilities = np.abs(state.get_vector()) ** 2

# evaluate performance of produced states
overall_cost_fn = circuits.get_cost_fn(edges, cost)
f_max, f_min, global_opt_state, global_opt_ind = evaluate.get_f_min_max(overall_cost_fn, n)
prob_of_max = evaluate.prob_of_f_max(probabilities, overall_cost_fn, f_max, n)
approx_ratio = evaluate.approx_ratio(f_obs, f_max, f_min)
print(f"n: {n}, p: {p}, approximation ratio: {approx_ratio}, P(f_max) = {prob_of_max}")




# draw top graph produced by our algorithm as well as global optimum
# top_occurance = helpers.int2state(probabilities.argmax(), n)

# colorings = [top_occurance, global_opt_state]
# for i in range(2):
#     plt.subplot(1, 2, i + 1)
#     coloring = colorings[i]
#     V = list(G.nodes)
#     E = list(G.edges)
#     node_color = ["red" if coloring[v] == 1 else "blue" for v in V]
#     edge_color = ["purple" if coloring[e[0]] != coloring[e[1]] else "black" for e in E]
#     width = [5 if coloring[e[0]] != coloring[e[1]] else 1 for e in E]
#     labels = {e: e for e in G.nodes}
#     nx.draw(G, node_color=node_color, edge_color=edge_color, width=width, pos=nx.circular_layout(G), labels=labels)

# plt.show()
