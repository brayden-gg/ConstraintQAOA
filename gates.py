import qulacs as q
import numpy as np
import circuits

def add_RXX_gate(circ, i, j, theta):
    circ.add_CNOT_gate(j, i)
    circ.add_RX_gate(j, -theta)
    circ.add_CNOT_gate(j, i)

def add_param_RXX_gate(circ, i, j, theta):
    circ.add_parametric_multi_Pauli_rotation_gate([i, j], [1, 1], -theta)

def add_RYY_gate(circ, i, j, theta):
    circ.add_RZ_gate(j, -np.pi/2)
    circ.add_RZ_gate(i, -np.pi/2)
    add_RXX_gate(circ, j, i, theta)
    circ.add_RZ_gate(i, np.pi/2)
    circ.add_RZ_gate(j, np.pi/2)

def add_param_RYY_gate(circ, i, j, theta):
    circ.add_parametric_multi_Pauli_rotation_gate([i, j], [2, 2], -theta)

def add_RXXX_gate(circ, i, j, k, theta):
    circ.add_CNOT_gate(k, j)
    circ.add_CNOT_gate(k, i)
    circ.add_RX_gate(k, theta)
    circ.add_CNOT_gate(k, i)
    circ.add_CNOT_gate(k, j)

def add_param_RXXX_gate(circ, i, j, k, theta):
    circ.add_parametric_multi_Pauli_rotation_gate([i, j, k], [1, 1, 1], theta)

def add_RXYY_gate(circ, i, j, k, theta):
    circ.add_RZ_gate(k, -np.pi/2)
    circ.add_RZ_gate(j, -np.pi/2)
    add_RXXX_gate(circ, i, j, k, theta)
    circ.add_RZ_gate(j, np.pi/2)
    circ.add_RZ_gate(k, np.pi/2)

def add_param_RXYY_gate(circ, i, j, k, theta):
    circ.add_parametric_multi_Pauli_rotation_gate([i, j, k], [1, 2, 2], theta)

def add_RYXY_gate(circ, i, j, k, theta):
    add_RXYY_gate(circ, j, i, k, theta)

def add_param_RYXY_gate(circ, i, j, k, theta):
    circ.add_parametric_multi_Pauli_rotation_gate([i, j, k], [2, 1, 2], theta)

def add_RYYX_gate(circ, i, j, k, theta):
    add_RXYY_gate(circ, k, j, i, theta)

def add_param_RYYX_gate(circ, i, j, k, theta):
    circ.add_parametric_multi_Pauli_rotation_gate([i, j, k], [2, 2, 1], theta)

def add_F_gate(circ, i, j, k, theta):
    # add_RXXX_gate(circ, i, j, k, -theta)
    # add_RXYY_gate(circ, i, j, k, -theta)
    # add_RYXY_gate(circ, i, j, k, -theta)
    # add_RYYX_gate(circ, i, j, k, theta)

    # XXX
    circ.add_CNOT_gate(k, j)
    circ.add_CNOT_gate(k, i)
    circ.add_RX_gate(k, -theta)
    circ.add_CNOT_gate(k, i)
    circ.add_CNOT_gate(k, j)

    # XYY
    circ.add_RZ_gate(k, -np.pi/2)
    circ.add_RZ_gate(j, -np.pi/2)
    add_RXXX_gate(circ, i, j, k, -theta)
    circ.add_RZ_gate(j, np.pi/2)
    # circ.add_RZ_gate(k, np.pi/2)

    # YXY
    # circ.add_RZ_gate(k, -np.pi/2)
    circ.add_RZ_gate(i, -np.pi/2)
    add_RXXX_gate(circ, i, j, k, -theta)
    circ.add_RZ_gate(k, np.pi/2)
    # circ.add_RZ_gate(i, np.pi/2)
    
    # YYX
    # circ.add_RZ_gate(i, -np.pi/2)
    circ.add_RZ_gate(j, -np.pi/2)
    add_RXXX_gate(circ, k, j, i, theta)
    circ.add_RZ_gate(j, np.pi/2)
    circ.add_RZ_gate(i, np.pi/2)

def add_param_F_gate(circ, i, j, k, theta):
    add_param_RXXX_gate(circ, i, j, k, -theta)
    add_param_RXYY_gate(circ, i, j, k, -theta)
    add_param_RYXY_gate(circ, i, j, k, -theta)
    add_param_RYYX_gate(circ, i, j, k, theta)

def add_S_gate(circ, i, j, theta):
    add_RXX_gate(circ, i, j, theta)
    add_RYY_gate(circ, i, j, theta)

def add_param_S_gate(circ, i, j, theta):
    add_param_RXX_gate(circ, i, j, theta)
    add_param_RYY_gate(circ, i, j, theta)

# BILP mixer
def add_UM(circ, beta, n, k):
    for i in range(0, n - k, 2*k):
        for j in range(0, k):
            add_S_gate(circ, i + j, i + j + k, beta)

    for i in range(0, n - k, 2*k):
        for j in range(0, k):
            add_S_gate(circ, i + j, i + j + k, beta)
    
    for i in range(1, k - 1):
        add_F_gate(circ, 0, i, i + 1, beta)
    
    if n > k: # self-add only possible if repeats
        add_F_gate(circ, 0, k, 1, beta)

# BILP mixer
def add_UM_medium_dense(circ, beta, n, k):
    # split S gates into two sections to overlap zi = z(i + k)
    for i in range(0, n - k, 2*k):
        for j in range(0, k):
            add_S_gate(circ, i + j, i + j + k, beta)
            # print(f"S_{i + j},{i + j + k}")

    for i in range(0, n - k, 2*k):
        for j in range(0, k):
            add_S_gate(circ, i + j, i + j + k, beta)
            # print(f"S_{i + j},{i + j + k}")

    # only need one layer for z1 + zk -> z(k + 1)
    for i in range(0, n, k):
        for j in range(1, k - 1):
            add_F_gate(circ, i, i + j, i + j + 1, beta)
            # print(f"F_{i},{i + j},{i+j+1}")
    
    # two layers for 
    for i in range(0, n - k, 2*k):
        add_F_gate(circ, i, i + k, i + 1, beta)
        # print(f"F_{i},{i + k},{i+1}")

    for i in range(k, n - k, 2*k):
        add_F_gate(circ, i, i + k, i + 1, beta)
        # print(f"F_{i},{i +k},{i+1}")

# # BILP mixer
# def add_param_UM(circ, beta, n, k):
#     for j in range(1, k):
#         for i in range(0, n, k):
#             add_param_S_gate(circ, i + j, i + j + k, beta)

#     for i in range(1, k - 1):
#         add_param_F_gate(circ, 0, i, i + 1, beta)
    
#     if n > k: # self-add only possible if repeats
#         add_param_F_gate(circ, 0, k, 1, beta)

# BILP cost fn
def add_UC(circ, gamma, c):
    # currently ignoring identity term since it only adds an overall phase
    for i, ci in enumerate(c):
        circ.add_RZ_gate(i, -gamma * ci/2)


def get_circuit(n, k, c, gammas, betas, dense=True):
    circ = q.QuantumCircuit(n)
    for gamma, beta in zip(gammas, betas):
        add_UC(circ, gamma, c)
        if dense:
            add_UM_medium_dense(circ, beta, n, k)
        else:
            add_UM(circ, beta, n, k)
    
    return circ

def get_optimization_fn(n, k, c, p, initial_state, C):
    def get_C_expectation(gamma_betas):
        gammas, betas = gamma_betas[:p], gamma_betas[p:]
        state = q.QuantumState(n)
        state.load(initial_state)
        circ = get_circuit(n, k, c, gammas, betas)
        circ.update_quantum_state(state)
        return -C.get_expectation_value(state).real
    
    return get_C_expectation

