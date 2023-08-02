import numpy as np



def S(z, i, j):
    if f"S_{i},{j}" not in valid_S:
        raise Exception(f"invalid S_{i},{j}")
    z = z.copy()
    if z[i] == z[j]:
        raise Exception(f"Swap can't swap equal z_{i} = z_{j} = {z[i]}")
    z[i], z[j] = 1 - z[i], 1 - z[j]
    return z

def F(z, i, j, l):
    if f"F_{i},{j},{l}" not in valid_F:
        raise Exception(f"invalid F_{i},{j},{l}")
    z = z.copy()
    if not z[i] == z[j] == 1 - z[l]:
        raise Exception(f"Fuse can't understand {i},{j},{l} = {z[i]},{z[j]},{z[l]}")
    z[i], z[j], z[l] = 1 - z[i], 1 - z[j], 1 - z[l]
    return z

# def pi(i):
#     r = (-i - 1) % k
#     c = i//k
#     return  r * cols + c

# def pinv(i):
#     r = (-i - 1) % cols
#     c = i//cols
#     return n - 1 - (r * k + c)

def tau(i, j):
    return k*(j+1) - i - 1

def phi(i, j):
    return i*cols + j + 1

def eval(z):
    return np.dot(z, coefs)

def get_w(z, b=None):
    if b is None:
        b = eval(z)
    w = [0] * len(z)
    for r in range(k):
        for c in range(cols):
            if b - coefs[tau(r, c)] >= 0:
                w[tau(r, c)] = 1
                b -= coefs[tau(r, c)]
        
    return w

def d(z, w):
    for i in range(len(z)):
        if w[i] != z[i]: #w[i] == 1 and z[i] == 0:
            return 1/(i+1)
    return 0

def score(z):
    sc = 0
    for i in range(k):
        for j in range(cols):
            sc += i/(i+1) * z[tau(i, j)]

    return sc

def get_path(z):
    w = get_w(z)
    P = []
    iters = 0
    z = shift_all_left(z, P)
    while score(z) > score(w):
        # get row with too many ones in z
        upward_row = -1
        for r in range(1, k):
            c_w = sum([w[tau(r, i)] for i in range(cols)])
            c_z = sum([z[tau(r, i)] for i in range(cols)])
            c_wprev = sum([w[tau(r-1, i)] for i in range(cols)])
            c_zprev = sum([z[tau(r-1, i)] for i in range(cols)])
            if c_w < c_z or c_z == cols and c_wprev > c_zprev:
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
            P.append(f"F_{0},{tau(upward_row, 0)},{tau(upward_row - 1, 0),}")
            z = F(z, 0, tau(upward_row, 0), tau(upward_row - 1, 0))

        z = shift_all_left(z, P)

        iters += 1
        if iters > 100000:
            print(k, cols, n)
            print(np.array(z0).reshape(k, cols))
            raise Exception("took too long")
    
    return P, z

def ensure_ones_available(z, P):
    row =  [z[tau(k-1, i)] for i in range(cols)]
    has_ones = np.any(row == np.ones(cols))
    ntimes = 0
    while not has_ones: # no ones, make one
        last_row = -1
        for r in range(k):
            if z[tau(r, 0)] == 1:
                last_row = r

        if last_row == k - 2:
            P.append(f"F_{0},{k},{1}")
            z = F(z, 0, k, 1)
        else:
            P.append(f"F_{0},{tau(last_row+1, 0)},{tau(last_row, 0)}")
            z = F(z, 0, tau(last_row+1, 0), tau(last_row, 0))

        row =  [z[tau(k-1, i)] for i in range(cols)]
        has_ones = np.any(row == np.ones(cols))
        ntimes += 1
        if ntimes > 1:
            print("WTF")
    return z

def unfill_left_col(z, P, r):
    row = [z[tau(r, i)] for i in range(cols)]
    if np.all(row == np.ones(cols)):
        raise Exception(f"Row {r} is filled :(")
    
    for i in range(cols - 1, 0, -1):
        if z[tau(r, i)] != z[tau(r, i-1)]:
            P.append(f"S_{tau(r, i-1)},{tau(r, i)}")
            z[:] = S(z, tau(r, i-1), tau(r, i))
    return z


def shift_all_left(z, P):
    for r in range(k):
        # run bubble sort on each row
        for l in range(cols):
            for j in range(cols - 1):
                if z[tau(r, j)] < z[tau(r, j+1)]:
                    P.append(f"S_{tau(r, j)},{tau(r, j+1)}")
                    z = S(z, tau(r, j), tau(r, j+1))
    return z       



for i in range(5000):
    # cols = 4
    # k = 6
    cols = np.random.randint(2, 15)
    k = np.random.randint(2, 15)
    n = k * cols
    z0 = np.random.randint(2, size=k * cols)
    # z0 = np.array([1, 0, 1, 0,
    #                0, 0, 0, 0,
    #                0, 0, 0, 0,
    #                0, 0, 0, 0])
    coefs = [j+1 for i in range(cols) for j in range(k)]
    valid_S = [f"S_{i},{i+k}" for i in range(n - k)]
    valid_F = [f"F_{0},{i},{i+1}" for i in range(1, k)]
    valid_F.append(f"F_0,{k},1")
    P, z1 = get_path(z0)
    w = get_w(z0)
    assert np.all(z1 == w)

print("done!")


