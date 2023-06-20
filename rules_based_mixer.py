import qulacs as q
import numpy as np
import scipy as sp
import scipy.linalg
import itertools
import networkx as nx
import helpers
import circuits
import gates


def get_minimal_mixer(n, k):
    M = q.GeneralQuantumOperator(n)
    for i in range(n - k):
        S = get_S(i, i + k, n)
        print(f"S{i},{i+k}")
        M += S

    for i in range(1, k - 1):
        F = get_F(0, i, i + 1, n)
        print(f"F{0},{k+i},{i+1}")
        M += F
    
    if n > k: # self-add only possible if repeats
        F = get_F(0, k, 1, n)
        print(f"F{0},{k},{1}")
        M += F
    
    return M

def get_mth_order_mixer(n, k, m):
    M = q.GeneralQuantumOperator(n)

    generators = []
    
    for i in range(n - k):
        S = get_S(i, i + k, n)
        generators.append(S)

    for i in range(n):
        for j in range(i + 1, n - i - 1):
            F = get_F(i, j, i + j + 1, n)
            generators.append(F)
    
    for i in range(m):
        for ops in itertools.product(generators, repeat=m):
            if len(set(ops)) != len(ops):
                continue # make sure no duplicates
            M += np.prod(ops)
            
    return M
            
    
def get_S(i, j, n):
    return circuits.get_custom_mixer_hamiltonian(n, [1/2, 1/2], [f"X {i} X {j}", f"Y {i} Y {j}"])

def get_F(i, j, k, n):
    return circuits.get_custom_mixer_hamiltonian(n, [1/4, 1/4, 1/4, -1/4],
                                          [f'X {i} X {j} X {k}', f'X {i} Y {j} Y {k}', f'Y {i} X {j} Y {k}', f'Y {i} Y {j} X {k}'])



    


if __name__ == "__main__":
    M = get_minimal_mixer(6, 3)
    # helpers.print_terms(M.get_matrix().todense())
    S1 = get_S(0, 3, 6)
    S2 = get_S(1, 4, 6)
    S3 = get_S(2, 5, 6)
    F1 = get_F(0, 3, 1, 6)
    F2 = get_F(0, 4, 2, 6)


    print(helpers.commutes(F1, F2))
    # F21 = get_F(0, 3, 1, 6)
    # F22 = get_F(0, 3, 4, 6)
    
    # ops1 = [S1, S2, S3, F21, F31]
    # ops2 = [S1, S2, S3, F22, F31]
    # ops3 = [S1, S2, S3, F21, F32]
    # ops4 = [S1, S2, S3, F22, F32]
    # names = ["S1", "S2", "S3", "F1", "F2"]

    
    # helpers.print_terms(sum(ops1).get_matrix())
    # helpers.print_terms(sum(ops2).get_matrix())
    # helpers.print_terms(sum(ops3).get_matrix())
    # helpers.print_terms(sum(ops4).get_matrix())


    # get_minimal_mixer(6, 3)
