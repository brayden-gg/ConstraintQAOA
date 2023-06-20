import numpy as np
import qulacs as q
import qulacs_core
import itertools
import scipy as sp
import scipy.linalg
# convert a base 10 integer to an array of its binary representation (little-endinan) 
def int2bits(z, n):
    bits = bin(z)[2:].zfill(n)[::-1]
    return np.array([int(i) for i in bits])

def bits2state(bits):
    n = len(bits)
    tot = 0
    for i, b in enumerate(bits):
        tot += b * (2**i)
    
    state = q.QuantumState(n)
    vec = np.zeros(2**n)
    vec[tot] = 1
    state.load(vec)
    return state

def state2bits(state):
    n = int(np.log2(state.get_vector().size))
    (index,) = np.where(np.isclose(state.get_vector(), 1))
    if index.size != 1:
        raise Exception(f"Can't convert non-basis state {state.get_vector()} to bits!")
    
    return int2bits(index[0], n)

def int2state(z, n):
    bits = int2bits(z, n)
    return bits2state(bits)

# turns list of gates into a simplified string
def gate_string(gates, sep=" "):
    rep = " ".join([g + sep + str(i) for i, g in enumerate(gates) if g != "I"])
    if rep == "":
        return "I"
    return rep

def print_terms(M):
    coefs, paulis = get_terms(M)
    print(f"{len(coefs)} terms:")
    # for gates, coef in terms:
    #     print(f"{coef} * {gate_string(gates)}")
    print("coefs = " + str(coefs))
    print("paulis = " + str(paulis))


def get_terms(M):
    paulis = []
    coefs = []
    n = int(np.log2(M.shape[0]))
    for gates in itertools.product("XYZI", repeat=n):
        op = q.Observable(n)
        pauli_string = gate_string(gates)
        op.add_operator(q.PauliOperator(pauli_string, 1))
        pauli = op.get_matrix()
        coef = (pauli @ M).trace()/(2**n)
        if coef != 0:
            paulis.append(pauli_string)
            coefs.append(coef)

    return coefs, paulis

# make matrix composition and scalarm multiplication easier
def scalar_mul(A, b):
    A2 = A.copy()
    A2.multiply_scalar(b)
    return A2

def scalar_add(A, b):
    return A + q.PauliOperator("I 0", b)

qulacs_core.QuantumGateMatrix.__rmul__ = scalar_mul
qulacs_core.GeneralQuantumOperator.__radd__ = scalar_add
qulacs_core.ClsOneQubitRotationGate.__radd__ = lambda a, b: q.gate.add(a, b)
qulacs_core.ClsOneControlOneTargetGate.__radd__ = lambda a, b: q.gate.add(a, b)
qulacs_core.ClsOneQubitRotationGate.__rmul__ = lambda a, b: q.gate.merge(a, b)
qulacs_core.ClsOneControlOneTargetGate.__rmul__ = lambda a, b: q.gate.merge(a, b)



# sanity check
def preserves_subspace(get_mixer, feasable, n) -> bool:
    in_span = True
    unfeasable = list(range(2**n))
    for f in feasable:
        unfeasable.remove(f)

    for beta in np.linspace(0, np.pi * 2, 100):
        UM = get_mixer(beta)
        # get unfeasable components of feasable states
        out_of_span = UM[feasable, :][:, unfeasable]
        if not np.all(out_of_span == np.zeros(out_of_span.shape, dtype=np.complex128)):
            in_span = False
            print(f"For beta = {beta}")
            print("UM(beta) = ")
            print(UM)
            break
        
    return in_span

def preserves_subspace_gates(get_mixer, feasable, n) -> bool:
    in_span = True
    unfeasable = list(range(2**n))
    for f in feasable:
        unfeasable.remove(f)

    for beta in np.linspace(0, np.pi * 2, 100):
        circ = q.ParametricQuantumCircuit(n)
        get_mixer(circ, beta)

        for z in feasable:
            state = int2state(z, n)
            circ.update_quantum_state(state)
            sv = state.get_vector()
            cond = np.isclose(sv[unfeasable], np.zeros_like(sv[unfeasable]))
            if not np.all(cond):
                in_span = False
                print(f"For beta = {beta}")
                print(f"nonzero-unfeasable states = {sv[unfeasable][~cond]}")
                break
        # get unfeasable components of feasable states
        
    return in_span



def commutes(A, B):
    comm = (A * B - B * A)
    return np.all(comm.get_matrix().todense() == 0)


