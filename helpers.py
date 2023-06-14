import numpy as np
import qulacs as q
import qulacs_core

# convert a base 10 integer to an array of its binary representation (little-endinan) 
def int2bits(z, n):
    bits = bin(z)[2:].zfill(n)[::-1]
    return np.array([int(i) for i in bits])

# make matrix composition and scalarm multiplication easier
def scalar_mul(A, b):
    A2 = A.copy()
    A2.multiply_scalar(b)
    return A2

def scalar_add(A, b):
    return A + q.PauliOperator("I 0", b)

qulacs_core.QuantumGateMatrix.__rmul__ = scalar_mul
qulacs_core.GeneralQuantumOperator.__radd__ = scalar_add


