
import qulacs as q
import numpy as np
import scipy as sp

import networkx as nx
from matplotlib import pyplot as plt
import helpers

# returns a cost function based on C_alpha, cost function returns 0 not all constraints are met
def get_cost_fn(clauses, cost, nconstraints=0, constraint=None):
    def cost_fn(state):
        for alpha in range(nconstraints):
            if not constraint(alpha, state):
                return 0
        return cost(clauses, state)

    return cost_fn

# convert cost function to hamiltonian
def get_cost_hamiltonian(cost, clauses, n, penalty=None, lam=None, nconstraints=0):
    # apply transformation to z to get C as a function of pauli matrices
    z_trans = []
    for i in range(n):
        transformed = q.GeneralQuantumOperator(n)
        transformed += q.PauliOperator(f"I {i}", 0.5)
        transformed += q.PauliOperator(f"Z {i}", -0.5)
        z_trans.append(transformed)

    # use C_alpha to get C hamiltonian
    C = q.GeneralQuantumOperator(n)
    C += cost(clauses, z_trans)
    
    for alpha in range(nconstraints):
        C -= penalty(alpha, z_trans) * lam

    return C


# original QAOA mixer hamiltonian
def get_mixer_hamiltonian(n):
    B = q.GeneralQuantumOperator(n)
    for i in range(n):
        # QAOA mixing operator
        B.add_operator(q.PauliOperator(f"X {i}", 1))
    return B

# build-your-own QAOA mixer hamiltonian
def get_custom_mixer_hamiltonian(n, coeffs, paulis):
    M = q.GeneralQuantumOperator(n)
    for coef, pauli in zip(coeffs, paulis):
        M.add_operator(q.PauliOperator(pauli, coef))
    return M

def get_hamiltonian_terms(n, coeffs, paulis):
    ops = []
    for coef, pauli in zip(coeffs, paulis):
        mat  = (q.Observable(n) + q.PauliOperator(pauli, coef)).get_matrix().todense()
        ops.append(mat)
    return ops

def get_initial_state(constraint, n):
    # start in superposition of feasable states
    subspace = [i for i in range(2**n) if constraint(0, helpers.int2bits(i, n))]
    initial_state = np.zeros(2**n)
    for state in subspace:
        initial_state[state] += 1
    initial_state /= np.linalg.norm(initial_state)
    return initial_state

# returns e^{-i*t*H}
def evolve(H, t, n):
    return q.gate.SparseMatrix(range(n), sp.linalg.expm(-1j * t * H.get_matrix().todense()))

# use optimized paramaters to return final quantum circuit
def get_circuit(n, p, C, B, gammas, betas, entangle=True):
    circ = q.QuantumCircuit(n)
    if entangle:
        for i in range(n):
            circ.add_H_gate(i)

    for i in range(p):
        UC = evolve(C, gammas[i], n)
        UB = evolve(B, betas[i], n)

        circ.add_gate(UC)
        circ.add_gate(UB)
    return circ


# function to maximize expectation
def get_optimization_fn(n, p, C, B, initial_state=None):
    def get_C_expectation(gamma_betas):
        gammas, betas = gamma_betas[:p], gamma_betas[p:]
        state = q.QuantumState(n)
        if initial_state is not None:
            state.load(initial_state)

        circ = get_circuit(n, p, C, B, gammas, betas, entangle=(initial_state is None))
        circ.update_quantum_state(state)
        return -C.get_expectation_value(state).real
    
    return get_C_expectation
