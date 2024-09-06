def paper_method():
    bdim = 1024
    kc = 3
    #koc = [11,11,2]
    koc = [11,7,6]
    n = (1 << 24)
    bc = n / (2 * bdim)

    #print("hxw",bc)
    olc = 24
    oc = -1

    ko =[0,0,0]
    kgs  = [[0,0],[0,0],[0,0]]
    kbs  = [[0,0],[0,0],[0,0]]


    for i in range(0,kc):
        oc = oc + koc[i]
        #print(oc)
        ko[i] = 1 << oc
        olc = olc - koc[i]

        if i == 0:
            kgs[i] = [1,bc]
            kbs[i] = [bdim/ko[i],ko[i]]

        else:
            kgs[i] = [bc/(1 << olc),(1<<olc)]
            kbs[i][1] = (2 * ko[i-1]) / kgs[i][1]
            kbs[i][0] = bdim / kbs[i][1]

    for i in range(0,3):
        print(kgs[i],kbs[i])

"""
对hxw_method()的测试表明,其和论文里给出的例子比较贴合
"""
def hxw_method():
    bdim = 256
    kc = 3
    koc = [9,5,4]
    #koc = [11,7,6]
    #koc = [3,3,2]
    n = (1 << 18)
    bc = n / (2 * bdim)

    #print("hxw",bc)
    olc = 18
    oc = -1

    ko =[0,0,0]
    kgs  = [[0,0],[0,0],[0,0]]
    kbs  = [[0,0],[0,0],[0,0]]

    for i in range(0,kc):
        olc = olc - koc[i] #该kernel开始时,所分成的组数

        kgs[i] = [bc / (1 << olc), (1 << olc)] #首先依据组数(使用的旋转因子是否相同),对整个数据进行分组,每个block处理该组的 2 * bDim个系数

        #下面考虑 每个block处理 2 * bDim个系数 应该如何从全局内存中取出来

        #该kernel需要做koc[i]层CT,因此需要 (1 << koc[i]) 个系数块 ,因此组会被分为 1 << (koc[i] - 1)个数据块，每个数据块被分为bc / (1 << olc)(gridDim.x)个片段，每个片段代表了连续读取的bdim / (1 << (koc[i] - 1))个系数

        kbs[i] = [bdim / (1 << (koc[i] - 1)),1 << (koc[i] - 1)]

        #下面判断组数和连续判断的长度的乘积是否等于 每组长度的一半
        #print ((bdim / (1 << (koc[i] - 1))) *  (bc / (1 << olc)) * (1 << (koc[i] - 1)),  n / (2 *(1 << olc) )) #后半部分是每组一半的长度
        print((bdim / (1 << (koc[i] - 1))) *  (bc / (1 << olc)) * (1 << (koc[i] - 1)) ==  n / (2 *(1 << olc) ))
        #print(kgs[i],"|",kbs[i])
        #print("+++++++++++++++++++++++++++")

    for i in range(0,kc): #调整输出的顺序，从第一个kernel开始
        print(kgs[kc - 1 -i],"|",kbs[kc - 1 -i])

hxw_method()
        