import numpy as np
import qulacs as q
import itertools
import helpers
n = 9
k = 3

def F(i, j, l):
    XXX = (q.Observable(n) + q.PauliOperator(f"X {i} X {j} X {l}", 1)).get_matrix().todense()
    XYY = (q.Observable(n) + q.PauliOperator(f"X {i} Y {j} Y {l}", 1)).get_matrix().todense()
    YXY = (q.Observable(n) + q.PauliOperator(f"Y {i} X {j} Y {l}", 1)).get_matrix().todense()
    YYX = (q.Observable(n) + q.PauliOperator(f"Y {i} Y {j} X {l}", 1)).get_matrix().todense()
    return 1/4 * (XXX + XYY + YXY - YYX)

def S(i, j):
    XX = (q.Observable(n) + q.PauliOperator(f"X {i} X {j}", 1)).get_matrix().todense()
    YY = (q.Observable(n) + q.PauliOperator(f"Y {i} Y {j}", 1)).get_matrix().todense()
    return 1/2 * (XX + YY)

# def get_uniform(b):
#     vec = np.zeros(2**n)
#     valid = [e for e in range(2**n) if np.dot(helpers.int2bits(e, n), coefs) == b]
#     vec[valid] = 1
#     vec /= np.linalg.norm(vec)
#     return vec
# def powerset(iterable):
#     "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
#     s = list(iterable)
#     return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))

min_fam_S = set()
for i in range(n):
    if (i, (i+k) % n) not in min_fam_S and ((i+k) % n, i) not in min_fam_S:
        min_fam_S.add((i, (i+k) % n))

min_fam_F = set() #{(0, i, i+1) for i in range(1, k-1)}.union({(0, k, 1)})

coefs = [j + 1 for i in range(n//k) for j in range(k)]

additional_S = {(i, j) for i, j in itertools.product(range(n), repeat=2) if coefs[i] == coefs[j] and i < j}
additional_S = additional_S.difference(min_fam_S)
additional_S = additional_S.difference({(b, a) for a, b in min_fam_S})
additional_F = {(i, j, l) for i, j, l in itertools.product(range(n), repeat=3) if i < j and coefs[i] + coefs[j] == coefs[l]}
additional_F = additional_F.difference(min_fam_F)



count = 0
min_unique = np.inf
unique_terms = []

def get_b(vec):
    vec = np.array(vec).round(10).squeeze()
    nonzero = [helpers.int2bits(z, n) for z in np.where(vec != 0)[0]]
    coefs = [j + 1 for i in range(n//k) for j in range(k)]
    val = np.dot(nonzero[0], coefs)
    for state in nonzero[1:]:
        if np.dot(state, coefs) != val:
            return -1
        
    return val


# A_F= {(1, 3, 2), (1, 3, 5)}
bs = {i for i in range(k*(n-1) + 1)}
# nicest = np.inf
# best_A = None
for r2 in range(1): #, len(additional_F)+1):
    # combos_S = itertools.combinations(additional_S, r1)
    combos_F = itertools.combinations(additional_F, r2)
   
    for A in combos_F:

# if True:
#     if True:
        # A = ((1, 3, 5), (0, 3, 4), (3, 4, 2), (0, 4, 5))
        fam_S = min_fam_S #.union(set(A_S))
        fam_F = min_fam_F.union(set(A))
        not_nice = 0
        M = sum(S(*x) for x in fam_S) + sum(F(*x) for x in fam_F)
        for b in range(1, n * (k - 1) + 1):
            if b not in bs:
                continue

            valid = [e for e in range(2**n) if np.dot(helpers.int2bits(e, n), coefs) == b]
            M_S = M[valid, :]
            M_S = M_S[:, valid]
            vals, vecs = np.linalg.eig(M_S)
            i = np.where(np.isclose(vals, vals.max()))[0]
            v = np.array(vecs[:, i]).round(10)
            unique = np.unique(v)
            if unique.size == 1:
                bs.remove(b)
                print(f"found b = {b}, with terms {unique}", A)
                if len(bs) == 0:
                    print("We did it boys!")
            

        # if not_nice < nicest:
        #     nicest = not_nice
        #     best_A = A
        #     print(best_A, nicest)

            # print(A_F)
            # top = np.where(np.isclose(vals, vals.max()))[0]
            # if top.size == 1:
            #     vec = np.array(vecs[:, top])
            #     top_vec = np.zeros(64)
            #     top_vec[valid] = vec.flatten()
            #     print(f"for b = {b}, top eigenvec is:")
            #     print(vec)
            # else:
            #     print(f"b = {b} has eigenspace dim {top.size}")
    




# vals, vecs = np.linalg.eig(M)
# i = np.where(np.isclose(vals, vals.max()))[0]
# v = np.array(vecs[:, i]).round(10)
# # assert np.allclose(M @ v, vals[i] * v)
# unique = np.unique(v)

# nonzero = [bin(z)[2:].zfill(n)[::-1] for z in np.where(v != 0)[0]]
# if nonzero != nine:
#     print(nonzero)
#     print(A_S, A_F)
#     quit()
# if unique.size == 2:# or (unique.size == min_unique and not np.all(unique_terms == unique)):
#     # print(vals[i] ** 2)
#     # print("new min unique")
#     min_unique = unique.size
#     unique_terms = unique
#     # print(min_unique)
#     # print("non-zero coefs (1/e)^2")
#     # print(np.array(vecs[:, i]).squeeze())
#     # print("sets:")
#     # [(a + 1, b + 1) for (a, b) in A_S], 
#     print([(a + 1, b + 1, c + 1) for a, b, c in A_F])
#     # print("non-zero")
#     # print([bin(z)[2:].zfill(n)[::-1] for z in np.where(v != 0)[0]])
#     # break

