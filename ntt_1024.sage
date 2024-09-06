#计算a*b mod Zq/x^n+1
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


#is_4step=0时，得到负折叠卷积的结果，整体过程符合FFT-trick,左边的子树分解为减
#is_4step=1时，计算的结果为循环卷积
def forward_ntt(src, n, q, root, is_4step=False):
    Zq = GF(q)

    a = src[:]
    length = 1
    while(length < n):
        #print("length=",length)
        for tid in range(n >> 1): #只关注上半部分
            step = int(n / length / 2) #代表的是步长 每一个组的半长
            psi_step = int(tid / step) #代表的是位于当前层的第几组中
            tar_idx = psi_step * step * 2 + (tid % step) #tar_idx代表的是第tid个蝴蝶操作开始的位置
            step_group = length + psi_step
            #print(step_group, bit_reverse(step_group, log(n, 2)))
            psi = root ^ bit_reverse(step_group, log(n, 2))
            if is_4step:
                step_group = psi_step
                psi = root ^ bit_reverse(step_group, log(n, 2) - 1) #旋转因子
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

#循环卷积NTT
def postive_NTT(src,n,q,root_n):
    
    a = forward_ntt(src, n, q, root_n, is_4step=True)
    return a

#逆循环卷积NTT
def postive_NTT_inv(src,n,q,root_n):
    
    inv_root = 1 / root_n
    a = inverse_ntt(src, n, q, inv_root, is_4step=True)
    return a

def forward_ntt_4step(src, n1, n2, q, root_n):
    n = n1 * n2

    print("[in forward]",root_n)

    
    root_n1 = root_n ^ n2
    root_n2 = root_n ^ n1

    M12 = MatrixSpace(Zq, n1, n2) #MatrixSpace 是用于定义矩阵空间的一个类
    M21 = MatrixSpace(Zq, n2, n1)
    M11 = MatrixSpace(Zq, n1) #n1*n1的方阵
    M22 = MatrixSpace(Zq, n2) #n2*n2的方阵

    # step 1
    a = M12(src) #将src中的值按行填充到矩阵中
    for j in range(a.ncols()): #遍历矩阵a中的每一列
        a[:,j] = column_matrix(forward_ntt(a[:,j].list(), n1, q, root_n1, is_4step=True)) #column_matrix用于将一个向量转变为列矩阵 a[:,j]代表的矩阵a的第j列

    
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

    a = a.transpose()

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


def inverse_ntt_4step_v2(src, n1, n2, q ,root_n):
    n = n1 * n2
    
    
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
    a = M12(src) #按行展开

    a = a.transpose()
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
            #W_n[i, j] = 1 / tmp
    a = a.elementwise_product(W_n)


    # step 4
    for j in range(a.ncols()):
        a[:,j] = column_matrix(inverse_ntt(a[:,j].list(), n1, q, inv_root_n1, is_4step=True))

    return a.list()




#####school book计算在Zq[x] / x^n + 1上的NTT#######
def schoolbook(a, b, n, n1, n2, q):
    c = polymul(a, b, n ,q)
    return c
# print(c)
##################################################

###########负折叠卷积NTT的计算####################
def negative_MUL(a, b, n, n1, n2, q ,root_n):
    inv_root_n = 1 / root_n
    #print("root_n",root_n)
    print("inv_root_n",inv_root_n)
    a_hat = forward_ntt(a, n, q, root_n)
    # print(a_hat)
    b_hat = forward_ntt(b, n, q, root_n)
    c_hat = [a_hat[i] * b_hat[i] for i in range(n)]
    c_prime = inverse_ntt(c_hat, n, q, inv_root_n)
    return c_prime,a_hat
    # print(c_prime)
    #print("negative",c_prime == c)
    #a_prime = inverse_ntt(a_hat, n, q, inv_root_n)
    #print("negative",a_prime == a)
###############################################

#4-step的循环卷积NTT
def ntt4step(a, b, n, n1, n2, q):
    Zq = GF(q)
    root_n = Zq(get_root_of_unity(n, q))

    a_hat_prime = forward_ntt_4step(a, n1, n2, q,root_n)
    # print(a_hat_prime)
    b_hat_prime = forward_ntt_4step(b, n1, n2, q,root_n)
    c_hat_prime = [a_hat_prime[i] * b_hat_prime[i] for i in range(n)]
    c_prime = inverse_ntt_4step_v2(c_hat_prime, n1, n2, q,root_n)
    
    a_hat_prime_2 = inverse_ntt_4step_v2(a_hat_prime, n1, n2, q,root_n)
    # print(c_prime)
    print("a == a_hat_prime_2",a == a_hat_prime_2)
    return c_prime,a_hat_prime
    #return [],a_hat_prime
    
    #print('c_prime == c', c_prime == c)
    #a_prime = inverse_ntt_4step(a_hat_prime, n1, n2, q)
    #print('a_prime == a', a_prime == a)

def neg_forward_ntt_4step(src, n1, n2, q, root_2n):
    n = n1 * n2
    M12 = MatrixSpace(Zq, n1, n2) #MatrixSpace 是用于定义矩阵空间的一个类
    M21 = MatrixSpace(Zq, n2, n1)

    root_2n1 = root_2n ^ n2

    a = M12(src)

    for j in range(a.ncols()):
        a[:,j] = column_matrix(forward_ntt(a[:,j].list(), n1, q, root_2n1, is_4step=False))

    W_n = M12()
    for i in range(W_n.nrows()):
        for j in range(W_n.ncols()):
            br_i = bit_reverse(i, log(n1, 2))
            W_n[i, j] = root_2n ^ ( (2 *br_i + 1 )* j)
    a = a.elementwise_product(W_n)

    a = a.transpose()

    root_n2 = root_2n ^ (2 * n1)
    for j in range(a.ncols()):
        a[:,j] = column_matrix(forward_ntt(a[:,j].list(), n2, q, root_n2, is_4step=True))

    a = a.transpose()
    return a.list()

def negative_inverse_ntt_4step(src, n1, n2, q ,root_2n):
    n = n1 * n2
    

    root_n2 = root_2n ^ (2 * n1)
    inv_root_2n = 1 / root_2n
    inv_root_n2 = 1 / root_n2
    M12 = MatrixSpace(Zq, n1, n2)
    M21 = MatrixSpace(Zq, n2, n1)

    # step 1
    a = M12(src) #按行展开

    a = a.transpose()
    for j in range(a.ncols()):
        a[:,j] = column_matrix(inverse_ntt(a[:,j].list(), n2, q, inv_root_n2, is_4step=True))

    # step 2
    a = a.transpose()

    # step 3
    W_n = M12()
    for i in range(W_n.nrows()):
        for j in range(W_n.ncols()):
            br_i = bit_reverse(i, log(n1, 2))
            W_n[i, j] = inv_root_2n ^ ( (2 *br_i + 1 )* j)
    a = a.elementwise_product(W_n)


    # step 4
    root_2n1 = root_2n ^ n2
    inv_root_2n1 = 1 / root_2n1
    for j in range(a.ncols()):
        a[:,j] = column_matrix(inverse_ntt(a[:,j].list(), n1, q, inv_root_2n1, is_4step=False))

    return a.list()

def negative_ntt4step(a, b, n, n1, n2, q ,root_2n):

    a_hat_prime = neg_forward_ntt_4step(a, n1, n2, q, root_2n)
    b_hat_prime = neg_forward_ntt_4step(b, n1, n2, q, root_2n)

    c_hat_prime = [a_hat_prime[i] * b_hat_prime[i] for i in range(n)]
    c_hat_prime_2 = negative_inverse_ntt_4step(c_hat_prime, n1, n2, q ,root_2n)

    
    
    a_hat_prime_2 = negative_inverse_ntt_4step(a_hat_prime, n1, n2, q ,root_2n)
    print("a_hat_prime_2 == a" ,a_hat_prime_2 == a)
    return c_hat_prime_2,a_hat_prime


if __name__ == "__main__":
    n = 2^10
    n1 = 2^5
    n2 = 2^5
    # q = find_prime(n1, n2, 15)
    q = 12289
    
    assert(q % (2*n) == 1)
    assert(q % (2*n1) == 1)
    assert(q % (2*n2) == 1)
    
    Zq = GF(q) #定义含有q个元素的优先域

    a = []
    for i in range(n):
        a.append(Zq(i))

    b = []
    for i in range(n):
        b.append(Zq(0))

    root_n = Zq(get_root_of_unity(n, q))

    root_2n = Zq(get_root_of_unity(2 * n, q)) #计算Zq上的2n次本原单位根

    c1 = schoolbook(a, b, n, n1, n2, q)
    c2 ,a1 = negative_MUL(a, b, n, n1, n2, q,root_2n)
    c3 ,a2 = ntt4step(a, b, n, n1, n2, q)


    a3 = postive_NTT(a,n,q,root_n)
    a4 = postive_NTT_inv(a3,n,q,root_n)
    print("a4 == a",a4 == a)

    c4,a5 = negative_ntt4step(a, b, n, n1, n2, q ,root_2n)
    print("a1 == a5",a1 == a5)

    print(c1 == c2 , c1 == c3,c4 == c2)

    print(a3 == a2)
