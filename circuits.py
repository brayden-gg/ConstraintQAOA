
import qulacs as q
import numpy as np
import scipy as sp
import scipy.linalg
import scipy.optimize
import networkx as nx
from matplotlib import pyplot as plt

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
    B = q.GeneralQuantumOperator(n)
    for coef, pauli in zip(coeffs, paulis):
        B.add_operator(q.PauliOperator(pauli, coef))
    return B

# returns e^{-i*t*H}
def evolve(H, t, n):
    return q.gate.SparseMatrix(range(n), sp.linalg.expm(-1j * t * H.get_matrix()))

# use optimized paramaters to return final quantum circuit
def get_circuit(n, p, C, B, gammas, betas, entangle=True):

    circ = q.QuantumCircuit(n)
    if entangle:
        for i in range(n):
            circ.add_H_gate(i)

    for i in range(p):
        UC = evolve(C, gammas[i], n)
        circ.add_gate(UC)
        UB = evolve(B, betas[i], n)
        circ.add_gate(UB)
    
    return circ

# function to maximize expectation
def get_optimization_fn(n, p, C, B, initial_state=None):
    
    def get_C_expectation(gamma_betas):
        gammas, betas = gamma_betas[:p], gamma_betas[p:]
        state = q.QuantumState(n)
        if initial_state is not None:
            state.load(initial_state)
        # state.set_zero_state()
        circ = get_circuit(n, p, C, B, gammas, betas, entangle=(initial_state is None))
        circ.update_quantum_state(state)
        return -C.get_expectation_value(state).real
    
    return get_C_expectation

# find optimal gammas and betas to maximize the optimization function
def find_optimal_angles(opt_fn, trials, p):
    gammas = None
    betas = None
    F_max = np.inf
    bounds = [(0, np.pi * 2)] * (2 * p) # can be any angle between 0 and 2pi
    for _ in range(trials):
        x0 = np.random.rand(2*p) * np.pi * 2
        f = opt_fn(x0)
        if f < F_max:    
            optim_res = sp.optimize.minimize(opt_fn, x0=x0, bounds=bounds)
            F_max = optim_res["fun"]
            gammas, betas = optim_res["x"][:p], optim_res["x"][p:]

    return F_max, gammas, betas