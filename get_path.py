import numpy as np



def S(z, i, j):
    if f"S_{i},{j}" not in valid_S:
        raise Exception(f"invalid S_{i},{j}")
    i, j = pi(i), pi(j)
    z = z.copy()
    if z[i] == z[j]:
        raise Exception(f"Swap can't swap equal z_{pinv(i)} = z_{pinv(j)} = {z[i]}")
    z[i], z[j] = 1 - z[i], 1 - z[j]
    return z

def F(z, i, j, l):
    if f"F_{i},{j},{l}" not in valid_F:
        raise Exception(f"invalid F_{i},{j},{l}")
    i, j, l = pi(i), pi(j), pi(l)
    z = z.copy()
    if not z[i] == z[j] == 1 - z[l]:
        raise Exception(f"Fuse can't understand {pinv(i)},{pinv(j)},{pinv(l)} = {z[i]},{z[j]},{z[l]}")
    z[i], z[j], z[l] = 1 - z[i], 1 - z[j], 1 - z[l]
    return z

def pi(i):
    r = (-i - 1) % k
    c = i//k
    return  r * cols + c

def pinv(i):
    r = (-i - 1) % cols
    c = i//cols
    return n - 1 - (r * k + c)

def eval(z):
    
    return np.dot(z, coefs)

def get_w(z, b=None):
    if b is None:
        b = eval(z)
    w = [0] * len(z)
    for i in range(len(z)):
        if b - coefs[i] >= 0:
            w[i] = 1
            b -= coefs[i]
        
    return w

def d(z, w):
    for i in range(len(z)):
        if w[i] != z[i]: #w[i] == 1 and z[i] == 0:
            return 1/(i+1)
    return 0

def get_path(z):
    w = get_w(z)
    P = []
    iters = 0
    z = shift_all_left(z, P)
    while d(z, w) != 0:
        # get row with too many ones in z
        upward_row = -1
        for r in range(1, k):
            r_prev = r-1
            if sum(w[r*cols:(r+1)*cols]) < sum(z[r*cols:(r+1)*cols]) or \
                sum(z[r*cols:(r+1)*cols]) == cols and sum(w[r_prev*cols:(r_prev+1)*cols]) > sum(z[r_prev*cols:(r_prev+1)*cols]) :
                upward_row = r
                break

        if upward_row == -1:
            raise Exception("could not find upward row?")
        
        z = ensure_ones_available(z, P)

        z = shift_all_left(z, P)
        z = unfill_left_col(z, P, upward_row - 1) # make room for one

        if upward_row == k - 1:
            P.append(f"F_{0},{k},{1}")
            z = F(z, 0, k, 1)
        else:
            P.append(f"F_{0},{(k - 1) - upward_row},{(k - 1) - upward_row + 1}")
            z = F(z, 0, (k - 1) - upward_row, (k - 1) - upward_row + 1)

        z = shift_all_left(z, P)

        iters += 1
        if iters > 100000:
            print(k, cols, n)
            print(np.array(z0).reshape(k, cols))
            raise Exception("took too long")
    
    return P, z

def ensure_ones_available(z, P):
    has_ones = np.any(z[-cols:] == np.ones(cols))
        
    while not has_ones: # no ones, make one
        last_row = -1
        for r in range(k):
            if z[pi(r)] == 1:
                last_row = r
                break
        if last_row == 1:
            P.append(f"F_{0},{k},{1}")
            z = F(z, 0, k, 1)
        else:
            P.append(f"F_{0},{last_row-1},{last_row}")
            z = F(z, 0, last_row-1, last_row)
        has_ones = np.any(z[-cols:] == np.ones(cols))

    return z

def unfill_left_col(z, P, r):
    if np.all(z[r*cols:(r+1)*cols] == np.ones(cols)):
        raise Exception(f"Row {r} is filled :(")
    
    for c in range(cols - 1, 0, -1):
        i = r * cols + c
        if z[i] != z[i - 1]:
            P.append(f"S_{pinv(i)},{pinv(i - 1)}")
            z[:] = S(z, pinv(i - 1), pinv(i))
    return z


def shift_all_left(z, P):
    for r in range(k):
        # run bubble sort on each row
        for l in range(cols):
            for j in range(r*cols, (r+1)*cols-l-1):
                if z[j] < z[j+1]:
                    P.append(f"S_{pinv(j)},{pinv(j+1)}")
                    z = S(z, pinv(j), pinv(j+1))
    return z       



for i in range(5000):
    cols = np.random.randint(2, 15)
    k = np.random.randint(2, 15)
    n = k * cols
    z0 = np.random.randint(2, size=k * cols)

# z0 = np.array([0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0,
#        1, 0])
# cols = 3
# k = 8
# n = n = k * cols

    coefs = [i for i in range(k,0,-1) for j in range(cols)]
    valid_S = [f"S_{i},{i+k}" for i in range(n - k)]
    valid_F = [f"F_{0},{i},{i+1}" for i in range(1, k)]
    valid_F.append(f"F_0,{k},1")
    P, z1 = get_path(z0)
    w = get_w(z0)
    assert np.all(z1 == w)

print("done!")


