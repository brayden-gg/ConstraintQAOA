import qulacs as q
import numpy as np
import scipy as sp
import scipy.linalg
import itertools
import networkx as nx
import helpers

def get_M_from_graph(feasable, G):
    M = np.zeros((2 ** n, 2**n), dtype=np.complex128)
    for i, j in G.edges:
        a = feasable[i]
        b = feasable[j]
        M[a, b] = 1
        M[b, a] = 1
    return M

# turns list of gates into a simplified string
def gate_string(gates, sep=" "):
    return " ".join([g + sep + str(i) for i, g in enumerate(gates) if g != "I"])

# sanity check
def preserves_subspace(M, feasable) -> bool:
    in_span = True
    unfeasable = list(range(2**n))
    for f in feasable:
        unfeasable.remove(f)

    for beta in np.linspace(0, np.pi * 2, 100):
        UM = sp.linalg.expm(-1j * M * beta)
        # get unfeasable components of feasable states
        out_of_span = UM[feasable, :][:, unfeasable]
        if not np.all(out_of_span == np.zeros(out_of_span.shape, dtype=np.complex128)):
            in_span = False
            print(f"For beta = {beta}")
            print("UM(beta) = ")
            print(UM)
            break
        
    return in_span

def get_terms(M):
    paulis = []
    coefs = []
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

def print_terms(M):
    coefs, paulis = get_terms(M)
    print(f"{len(coefs)} terms:")
    # for gates, coef in terms:
    #     print(f"{coef} * {gate_string(gates)}")
    print("coefs = " + str(coefs))
    print("paulis = " + str(paulis))

#  check if communites with Σ Zi
def check_commutator(M):
    Zs = q.Observable(n)
    for i in range(n):
        Zs.add_operator(q.PauliOperator(f"Z {i}", 1))
    Zs = Zs.get_matrix()

    commutator = Zs @ M - M @ Zs
    return np.all(commutator == 0)


def constraint(zs):
    S = np.zeros_like(zs)
    Si = np.arange(len(zs))
    S[Si % 3 == 0] = 1
    S[Si % 3 == 1] = 2
    S[Si % 3 == 2] = 3
    return np.dot(S, zs)

n = 6
XYZI = ["I", "X", "Y", "Z"]
# nfeasable =  6
# feasable = np.random.choice(range(2**n), nfeasable, replace=False).tolist()
# feasable = [0b001, 0b010, 0b100]
# nfeasable =  len(feasable)
# G = nx.cycle_graph(nfeasable)
# G = nx.path_graph(nfeasable)
# E = list(G.edges)
# G.remove_edge(*E[len(E)//2])
# G = nx.complete_graph(nfeasable)
# M = get_M_from_graph(feasable, G)
buckets = []
for i in range(3 * n//2 + 1):
    buckets.append([])

for i in range(2 ** n):
    bits = helpers.int2bits(i, n)
    b = constraint(bits)
    buckets[b].append(i)

b = 3
edges = []
bucket = buckets[b]
# bucket.append(bucket[0])
# for bucket in buckets:
# edges += [(a, b) for a, b in zip(bucket, bucket[1:] + [bucket[0]])]
edges = [
                  (1, 4), # b = 1
                  (2, 5), (5, 8), (8, 2),# b = 2
                  (3, 6), (6, 9), (9, 12), (12, 3),# b = 3
                  (7, 10), (10, 13), (13, 7), # b = 4
                  (7, 13), # b = 5
                  ]
best_M = None
best_perm = None
least_terms = np.inf
print(edges)
# for i in range(500):
perm = np.arange(2**n)
# np.random.shuffle(perm)
perm_edges = [(perm[a], perm[b]) for a, b in edges]
# print(perm)

G = nx.Graph()
G.add_nodes_from(range(2**n))
G.add_edges_from(perm_edges)
M = nx.adjacency_matrix(G)
feasable = perm[buckets[b]]
nfeasable = len(feasable)
# assert preserves_subspace(M, feasable)
coefs, paulis = get_terms(M)
if len(coefs) < least_terms:
    least_terms = len(coefs)
    best_M = M
    best_perm = perm
    

print(f"{nfeasable} states: {[bin(s)[2:].zfill(n) for s in feasable]}")
print_terms(best_M)
print(best_perm.tolist())


# nonzero = np.where(M0 != 0)
# best_entries = [1] * len(nonzero[0])
# min_terms = 99
# for i in range(200):
#     M1 = M0.copy()
#     entries = []
#     for r, c in zip(*nonzero):
#         entry = np.random.choice([1, -1])
#         M1[r, c] = entry
#         entries.append(entry)
#     terms = get_terms(M1)
#     if len(terms) < min_terms:
#         min_terms = len(terms)
#         best_entries = entries










# print(min_terms, best_entries)

# res = np.zeros((2**n, 2**n), dtype=np.complex128)

# for coef, pauli in zip(coeffs, paulis):
#     res += 1/ 8 * coef * pauli


# print(res)


