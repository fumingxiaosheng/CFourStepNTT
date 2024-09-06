def polymul(a, b, n, q):
    assert(len(a) == len(b) == n)
    Zq = GF(q)
    c = [Zq(0)] * n
    for i in range(n):
        for j in range(n):
            if i+j > n-1:
                c[i+j-n] += -a[i] * b[j]
            else:
                c[i+j] += a[i] * b[j]
    return c


def find_prime(n1, n2, bit_len):
    n = n1 * n2
    largest_value = 2 ^ bit_len
    value = previous_prime(largest_value)
    while True:
        if value % (2*n) == 1 and value % (2*n1) == 1 and value % (2*n2) == 1:
            return value
        value = previous_prime(value)


def znorder(a, q):
    mul = 1
    if xgcd(a, q)[0] == 1:
        for i in range(1, q):
            mul = (mul * a) % q
            if mul % q == 1:
                return i


def get_root_of_unity(n, q):
    for i in range(2, q):
        if znorder(i, q) == n:
            return i


def bit_reverse(index, bit_len):
    return int('{:0{width}b}'.format(index, width=bit_len)[::-1], 2)


def forward_ntt(src, n, q, root, is_4step=False):
    Zq = GF(q)

    a = src[:]
    length = 1
    while(length < n):
        for tid in range(n >> 1):
            step = int(n / length / 2)
            psi_step = int(tid / step)
            tar_idx = psi_step * step * 2 + (tid % step)
            step_group = length + psi_step
            psi = root ^ bit_reverse(step_group, log(n, 2))
            if is_4step:
                step_group = psi_step
                psi = root ^ bit_reverse(step_group, log(n, 2) - 1)
            U, V = Zq(a[tar_idx]), Zq(a[tar_idx + step])
            a[tar_idx], a[tar_idx + step] = U + V*psi, U - V*psi
        length <<= 1
    return a


def inverse_ntt(src, n, q, inv_root, is_4step=False):
    Zq = GF(q)

    a = src[:]
    step = 1
    while(step < n):
        for tid in range(n >> 1):
            len = int(n / step / 2)
            psi_step = int(tid / step)
            tar_idx = psi_step * step * 2 + (tid % step)
            step_group = len + psi_step
            psi = inv_root ^ bit_reverse(step_group, log(n, 2))
            if is_4step:
                step_group = psi_step
                psi = inv_root ^ bit_reverse(step_group, log(n, 2) - 1)
            U, V = Zq(a[tar_idx]), Zq(a[tar_idx + step])
            a[tar_idx], a[tar_idx + step] = (U + V) / 2, (U - V) * psi / 2
        step <<= 1
    return a


def forward_ntt_4step(src, n1, n2, q):
    n = n1 * n2

    Zq = GF(q)
    root_n = Zq(get_root_of_unity(n, q))
    root_n1 = root_n ^ n2
    root_n2 = root_n ^ n1

    M12 = MatrixSpace(Zq, n1, n2)
    M21 = MatrixSpace(Zq, n2, n1)
    M11 = MatrixSpace(Zq, n1)
    M22 = MatrixSpace(Zq, n2)

    # step 1
    a = M12(src)
    for j in range(a.ncols()):
        a[:,j] = column_matrix(forward_ntt(a[:,j].list(), n1, q, root_n1, is_4step=True))

    # step 2
    W_n = M12()
    for i in range(W_n.nrows()):
        for j in range(W_n.ncols()):
            br_i = bit_reverse(i, log(n1, 2))
            W_n[i, j] = root_n ^ (br_i * j)
    a = a.elementwise_product(W_n)

    # step 3
    a = a.transpose()

    # step 4
    for j in range(a.ncols()):
        a[:,j] = column_matrix(forward_ntt(a[:,j].list(), n2, q, root_n2, is_4step=True))

    return a.list()


def inverse_ntt_4step(src, n1, n2, q):
    n = n1 * n2
    Zq = GF(q)
    root_n = Zq(get_root_of_unity(n, q))
    root_n1 = root_n ^ n2
    root_n2 = root_n ^ n1
    inv_root_n = 1 / root_n
    inv_root_n1 = 1 / root_n1
    inv_root_n2 = 1 / root_n2
    inv_n1_mod_q = 1 / Zq(n1)
    inv_n2_mod_q = 1 / Zq(n2)

    M12 = MatrixSpace(Zq, n1, n2)
    M21 = MatrixSpace(Zq, n2, n1)
    M11 = MatrixSpace(Zq, n1)
    M22 = MatrixSpace(Zq, n2)

    # step 1
    #a = M21(src)

    atmp = M21(src) #按章
    atmp = atmp.transpose()
    srctmp = atmp.list()
    a = M21(srctmp)

    for j in range(a.ncols()):
        a[:,j] = column_matrix(inverse_ntt(a[:,j].list(), n2, q, inv_root_n2, is_4step=True))

    # step 2
    a = a.transpose()

    # step 3
    W_n = M12()
    for i in range(W_n.nrows()):
        for j in range(W_n.ncols()):
            br_i = bit_reverse(i, log(n1, 2))
            W_n[i, j] = (inv_root_n ^ (br_i * j))
    a = a.elementwise_product(W_n)

    # step 4
    for j in range(a.ncols()):
        a[:,j] = column_matrix(inverse_ntt(a[:,j].list(), n1, q, inv_root_n1, is_4step=True))

    return a.list()


n = 2^9
n1 = 2^4
n2 = 2^5
# q = find_prime(n1, n2, 15)
q = 12289
assert(q % (2*n) == 1)
assert(q % (2*n1) == 1)
assert(q % (2*n2) == 1)
Zq = GF(q)
"""
a = []
for i in range(n/2):
    a.append(Zq(i))
for i in range(n/2):
    a.append(Zq(0))

b = []
for i in range(n/2):
    b.append(Zq(i))
for i in range(n/2):
    b.append(Zq(0))

a.clear()
b.clear()
for i in range(n):
    a.append(Zq(i))
    b.append(Zq(i))
"""
a = []
for i in range(n):
    a.append(Zq(i))
    #a.append( Zq.random_element())

b = []
for i in range(n):
    b.append(Zq(i))
    #b.append( Zq.random_element())

c1 = polymul(a, b, n ,q)
# print(c)

root_n = Zq(get_root_of_unity(2 * n, q))
inv_root_n = 1 / root_n
a_hat = forward_ntt(a, n, q, root_n)
# print(a_hat)
b_hat = forward_ntt(b, n, q, root_n)
c_hat = [a_hat[i] * b_hat[i] for i in range(n)]
c2 = inverse_ntt(c_hat, n, q, inv_root_n)
# print(c_prime)
#print(c_prime == c)
#a_prime = inverse_ntt(a_hat, n, q, inv_root_n)
#print(a_prime == a)


a_hat_prime = forward_ntt_4step(a, n1, n2, q)
# print(a_hat_prime)
b_hat_prime = forward_ntt_4step(b, n1, n2, q)
c_hat_prime = [a_hat_prime[i] * b_hat_prime[i] for i in range(n)]
c3 = inverse_ntt_4step(c_hat_prime, n1, n2, q)
# print(c_prime)
print('c1 == c2', c1 == c2)
print('c1 == c3', c1 == c3)
a_prime = inverse_ntt_4step(a_hat_prime, n1, n2, q)
#print('a_prime == a', a_prime == a)
