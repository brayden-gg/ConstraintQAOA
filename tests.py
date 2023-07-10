import numpy as np
import qulacs as q
import networkx as nx

import circuits
import helpers
import evaluate
import rules_based_mixer
import gates
import optimize

import scipy as sp
import scipy.linalg

# # Tests for helpers
# assert np.all(helpers.int2bits(1, 4) == [1, 0, 0, 0])
# assert np.all(helpers.int2bits(2, 4) == [0, 1, 0, 0])
# assert np.all(helpers.int2bits(3, 4) == [1, 1, 0, 0])
# assert np.all(helpers.int2bits(1234, 10) == [0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1])

# print("Helper tests passed!")
# # Tests for circuits

# # define some objective functions to test
# def cost_maxcut(edges, state):
#     res = 0
#     for i, j in edges:
#         # XOR them together (i + j - 2ij)
#         res += state[i] + state[j] - state[i] * state[j] * 2
#     return res

# def get_cost_BILP(c):
#     def cost(_, z): 
#         return np.dot(z, c)
#     return cost

# # define some constraints to test

# def get_constraints_BILP(S, b):
#     def penalty(alpha, z):
#         diff = -b[alpha, 0] + np.dot(z, S[alpha, :])
#         return diff * diff

#     def constraint(alpha, z):
#         return np.dot(z, S[alpha, :]) == b[alpha, 0]
    
#     return penalty, constraint

# # Graph 1: P2, a path of length 2
# G1 = nx.from_edgelist([(0, 1)])
# C1 = circuits.get_cost_hamiltonian(cost_maxcut, list(G1.edges), len(G1.nodes))
# assert np.all(C1.get_matrix().todense() == np.array([[0, 0, 0, 0],
#                                                     [0, 1, 0, 0], 
#                                                     [0, 0, 1, 0],
#                                                     [0, 0, 0, 0]]))

# B1 = circuits.get_mixer_hamiltonian(len(G1.nodes))
# assert np.all(B1.get_matrix().todense() == np.array([[0, 1, 1, 0],
#                                                     [1, 0, 0, 1], 
#                                                     [1, 0, 0, 1],
#                                                     [0, 1, 1, 0]]))


# F_max1_good, gammas1_good, betas1_good = (-1, [np.pi/2], [np.pi/8])

# opt_fn1 = circuits.get_optimization_fn(n=len(G1.nodes), p=1, C=C1, B=B1)
# F_max1, _, _ = optimize.scipy_optimize(opt_fn1, p=1, trials=100)
# assert np.isclose(F_max1, F_max1_good)

# gammas1 = [np.pi/2]
# betas1 = [np.pi/8]

# UC1 = circuits.evolve(C1, gammas1[0], len(G1.nodes))
# UB1 = circuits.evolve(B1, betas1[0], len(G1.nodes))

# assert np.all(np.isclose(UC1.get_matrix(), np.array([[1, 0, 0, 0], 
#                                          [0, -1j, 0, 0], 
#                                          [0, 0, -1j, 0], 
#                                          [0, 0, 0, 1]])))

# B = B1.get_matrix().todense()
# assert np.all(np.isclose(UB1.get_matrix(), 
#                 np.eye(4) - B@B/4 
#                 + 0.25*B@B*np.cos(-2*betas1[0]) 
#                 + 0.5j*B*np.sin(-2*betas1[0])))


# circ1 = circuits.get_circuit(n=len(G1.nodes), p=1, C=C1, B=B1, gammas=gammas1, betas=betas1)

# state1 = q.QuantumState(len(G1.nodes))
# state1.set_zero_state()
# circ1.update_quantum_state(state1)

# assert np.all(np.isclose(state1.get_vector(), np.array([0, -1j * np.sqrt(2)/2, -1j * np.sqrt(2)/2, 0])))
# f_obs = C1.get_expectation_value(state1).real
# probabilities = np.abs(state1.get_vector()) ** 2
# assert np.all(np.isclose(probabilities, np.array([0, 0.5, 0.5, 0])))

# # Graph 2: C3, a triangle
# G2 = nx.from_edgelist([(0, 1), (1, 2), (0, 2)])
# C2 = circuits.get_cost_hamiltonian(cost_maxcut, list(G2.edges), len(G2.nodes))
# B2 = circuits.get_mixer_hamiltonian(len(G2.nodes))

# opt_fn2 = circuits.get_optimization_fn(n=len(G2.nodes), p=1, C=C2, B=B2)
# F_max2, _, _ = optimize.scipy_optimize(opt_fn2, trials=100, p=1)
# assert np.isclose(F_max2, -2)
 
# gammas2 = [np.pi * -0.19591327540963036]
# betas2 = [np.pi * -9.7956640041929194E-002]

# UC2 = circuits.evolve(C2, gammas2[0], len(G2.nodes))
# UB2 = circuits.evolve(B2, betas2[0], len(G2.nodes))

# circ2 = circuits.get_circuit(n=len(G2.nodes), p=1, C=C2, B=B2, gammas=gammas2, betas=betas2)

# state2 = q.QuantumState(len(G2.nodes))
# state2.set_zero_state()
# circ2.update_quantum_state(state2)

# # second row of https://code.ornl.gov/qci/qaoa-dataset-version1/-/blob/master/States/p%3D1/state_n%3D3_p%3D1.txt
# state2_good = np.array([-7.4220765228227847E-009 + -1.2062882429053623E-009j, 
#                         -0.12366063457963243 + 0.38906903516217956j,
#                         -0.12366063457963240 + 0.38906903516217956j,    
#                         -0.12366063457963240 + 0.38906903516217956j,
#                         -0.12366063457963240 + 0.38906903516217956j,      
#                         -0.12366063457963240 + 0.38906903516217956j,
#                         -0.12366063457963243 +  0.38906903516217956j,
#                         -7.4220765228227847E-009 + -1.2062882429053623E-009j])


# assert np.all(np.isclose(state2.get_vector(), state2_good))
# print("MaxCut tests passed!")

# # Tests for BILP
# # very easy case

# c0 = np.array([1, 1])
# C_alpha_BILP0 = get_cost_BILP(c0)

# C3 = circuits.get_cost_hamiltonian(C_alpha_BILP0, range(len(c0)), len(c0))
# B3 = circuits.get_mixer_hamiltonian(len(c0))
# opt_fn3 = circuits.get_optimization_fn(n=len(c0), p=1, C=C3, B=B3)
# F_max3, _, _ = optimize.scipy_optimize(opt_fn3, trials=100, p=1)
# assert np.isclose(-F_max3, 2)

# gammas3 = [-1/2 * np.pi]
# betas3 = [-1/4 * np.pi]
# circ3 = circuits.get_circuit(len(c0), 1, C3, B3, gammas3, betas3)

# assert np.isclose(-opt_fn3(gammas3 + betas3), 2)
# state3 = q.QuantumState(2)
# state3.set_zero_state()
# circ3.update_quantum_state(state3)

# assert np.all(np.isclose(state3.get_vector(), [0, 0, 0, -1]))
# print("BILP tests passed!")

# # tests for split and fusion/fission operators
# n = 6

# initial_state = helpers.bits2state([1, 1, 0, 0, 0, 0])
# F012 = rules_based_mixer.get_F(0, 1, 2, n).get_matrix().todense()
# S25 = rules_based_mixer.get_S(2, 5, n).get_matrix().todense()
# S14 = rules_based_mixer.get_S(1, 4, n).get_matrix().todense()
# F045 = rules_based_mixer.get_F(0, 4, 5, n).get_matrix().todense()

# state_vec1 = S14 @ initial_state.get_vector()
# state1 = q.QuantumState(n)
# state1.load(np.array(state_vec1).squeeze().tolist())
# assert np.all(helpers.state2bits(state1) == [1, 0, 0, 0, 1, 0])

# state_vec2 = F012 @ initial_state.get_vector()
# state2 = q.QuantumState(n)
# state2.load(np.array(state_vec2).squeeze().tolist())
# assert np.all(helpers.state2bits(state2) == [0, 0, 1, 0, 0, 0])

# state_vec3 = S25 @ initial_state.get_vector()
# assert np.all(state_vec3 == 0)

# state_vec4 = F045 @ S14 @ initial_state.get_vector()
# state4 = q.QuantumState(n)
# state4.load(np.array(state_vec4).squeeze().tolist())
# assert np.all(helpers.state2bits(state4) == [0, 0, 0, 0, 0, 1])
# print("Swap and Fusion/Fission tests passed!")

# # Gates mixer preserves subspace

S = np.array([[1, 2, 3, 1, 2, 3]])
b = np.array([[7]])
k = 3
n = 6

def constraint_BILP(alpha, z):
    return np.dot(z, S[alpha, :]) == b[alpha, 0]

subspace = [i for i in range(2**n) if constraint_BILP(0, helpers.int2bits(i, n))]
initial_state = np.zeros(2**n)
for state in subspace:
    initial_state[state] += 1
initial_state /= np.linalg.norm(initial_state)


test_n = 6
# Make sure custom operators have expected behavior
def F(theta):
    XXX = (q.Observable(test_n) + q.PauliOperator("X 0 X 1 X 2", 1)).get_matrix().todense()
    XYY = (q.Observable(test_n) + q.PauliOperator("X 0 Y 1 Y 2", 1)).get_matrix().todense()
    YXY = (q.Observable(test_n) + q.PauliOperator("Y 0 X 1 Y 2", 1)).get_matrix().todense()
    YYX = (q.Observable(test_n) + q.PauliOperator("Y 0 Y 1 X 2", 1)).get_matrix().todense()
    F = 1/4 * (XXX + XYY + YXY - YYX)
    return sp.linalg.expm(-1j * F * theta * 2)

def S(theta):
    XX = (q.Observable(test_n) + q.PauliOperator("X 0 X 3", 1)).get_matrix().todense()
    YY = (q.Observable(test_n) + q.PauliOperator("Y 0 Y 3", 1)).get_matrix().todense()
    S = 1/2 * (XX + YY)
    return sp.linalg.expm(-1j * S * theta)

for i in range(2**test_n):
    for beta in np.linspace(0, np.pi*2, 10):
        F_state1 = helpers.int2state(i, 6)
        F_state2 = helpers.int2state(i, 6)
        F_sv = F_state2.get_vector()

        S_state1 = helpers.int2state(i, 6)
        S_state2 = helpers.int2state(i, 6)
        S_sv = S_state2.get_vector()

        F_circ1 = q.QuantumCircuit(6)
        F_circ2 = q.ParametricQuantumCircuit(6)
        F_c3 = F(beta)

        S_circ1 = q.QuantumCircuit(6)
        S_circ2 = q.ParametricQuantumCircuit(6)
        S_c3 = S(beta)

        gates.add_F_gate(F_circ1, 0, 1, 2, beta)
        gates.add_param_F_gate(F_circ2, 0, 1, 2, beta)
        gates.add_S_gate(S_circ1, 0, 3, beta)
        gates.add_param_S_gate(S_circ2, 0, 3, beta)

        F_circ1.update_quantum_state(F_state1)
        F_circ2.update_quantum_state(F_state2)
        F_sv = F_c3 @ F_sv

        S_circ1.update_quantum_state(S_state1)
        S_circ2.update_quantum_state(S_state2)
        S_sv = S_c3 @ S_sv

        

        assert np.all(np.isclose(F_state1.get_vector(), F_state2.get_vector()))
        assert np.all(np.isclose(F_state1.get_vector(), F_sv))

        assert np.all(np.isclose(S_state1.get_vector(), S_state2.get_vector()))
        assert np.all(np.isclose(S_state1.get_vector(), S_sv))
        
# assert helpers.preserves_subspace(lambda b: sp.linalg.expm(-1j * b * F012), subspace, n)

# assert helpers.preserves_subspace_gates(lambda c, b: gates.add_F_gate(c, 0, 1, 2, b), subspace, n)
# assert helpers.preserves_subspace_gates(lambda c, b: gates.add_S_gate(c, 0, 3, b), subspace, n)
# assert helpers.preserves_subspace_gates(lambda c, b: gates.add_S_gate(c, 0, 3, b), subspace, n)
print("Subspace tests passed")
# assert helpers.preserves_subspace_gates(lambda c, b: gates.add_param_F_gate(c, 0, 1, 2, b), subspace, n)
# assert helpers.preserves_subspace_gates(lambda c, b: gates.add_param_F_gate(c, 0, 3, 1, b), subspace, n)
# assert helpers.preserves_subspace_gates(lambda c, b: gates.add_param_S_gate(c, 0, 3, b), subspace, n)







