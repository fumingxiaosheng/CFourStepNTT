"""
out = 0
print("16个后面padding两个")
for i in range(1,257):
    print(i,end=" ")
    out = out + 1
    if(out % 16 == 0):
        print("\n")
    if(i % 16 == 0):
        print("*",end=" ")
        out = out + 1
        if(out % 16 == 0):
            print("\n")
        print("*",end=" ")
        out = out + 1
        if(out % 16 == 0):
            print("\n")

out = 0
print("8个后面padding一个")
for i in range(1,257):
    print(i,end=" ")
    out = out + 1
    if(out % 16 == 0):
        print("\n")
    if(i % 8 == 0):
        print("*",end=" ")
        out = out + 1
        if(out % 16 == 0):
            print("\n")
out = 0
print("16个后面padding三个")
for i in range(1,257):
    print(i,end=" ")
    out = out + 1
    if(out % 16 == 0):
        print("\n")
    if(i % 16 == 0):
        print("*",end=" ")
        out = out + 1
        if(out % 16 == 0):
            print("\n")
        print("*",end=" ")
        out = out + 1
        if(out % 16 == 0):
            print("\n")
        print("*",end=" ")
        out = out + 1
        if(out % 16 == 0):
            print("\n")


out = 0
print("64维NTT存储体冲突解决")
for i in range(1,65):
    print("{:>3}".format(i),"|",end=" ")
    out = out + 1
    if(out % 16 == 0):
        print("\n")
    if(i%2 == 0):
        print("   *|",end=" ")
        out = out + 1
        if(out % 16 == 0):
            print("\n")

out = 0
print("64维初始NTT存储")
for i in range(1,65):
    print("{:>3}".format(i),"|",end=" ")
    out = out + 1
    if(out % 16 == 0):
        print("\n")

out = 0
print("64维3层NTT存储体冲突解决")
for i in range(0,64):
    print("{:>3}".format(out),",",((int)(i /16)) * 24 + (i % 16) +  ((int)(((int)(i % 16)) / 8)) * 8,"|",end=" ")
    out = out + 1
    if(out % 16 == 0):
        print("\n")
    if((i+1)%8 == 0 and (i+1)% 16 !=0):
        for j in range(0,8):
            print("   *|",end=" ")
            out = out + 1
            if(out % 16 == 0):
                print("\n")

out = 0
print("64维2层NTT存储体冲突解决")
for i in range(0,64):
    print("{:>3}".format(out),",",((int)(i /8)) * 12 + (i % 8) +  ((int)(((int)(i % 8)) / 4)) * 4,"|",end=" ")
    out = out + 1
    if(out % 16 == 0):
        print("\n")
    if((i+1)%4 == 0 and (i+1)%8 !=0):
        for j in range(0,4):
            print("   *|",end=" ")
            out = out + 1
            if(out % 16 == 0):
                print("\n")
out = 0
print("64维1层NTT存储体冲突解决")
for i in range(0,64):
    print("{:>3}".format(out),",",((int)(i /4)) * 6 + (i % 4) +  ((int)(((int)(i % 4)) / 2)) * 2,"|",end=" ")
    out = out + 1
    if(out % 16 == 0):
        print("\n")
    if((i+1)%2 == 0 and (i+1)%4 !=0):
        for j in range(0,2):
            print("   *|",end=" ")
            out = out + 1
            if(out % 16 == 0):
                print("\n")

out = 0
print("64维0层NTT存储体冲突解决")
for i in range(1,65):
    print("{:>3}".format(i),"|",end=" ")
    out = out + 1
    if(out % 16 == 0):
        print("\n")
    if(i%1 == 0 and i%2 !=0):
        for j in range(0,1):
            print("   *|",end=" ")
            out = out + 1
            if(out % 16 == 0):
                print("\n")


out = 0
print("8个线程8个系数存取的模拟")
for i in range(1,65):
    print("{:>3}".format(i),"|",end=" ")
    out = out + 1
    if(out % 16 == 0):
        print("\n")
    if(i%2 == 0):
        for j in range(0,1):
            print("   *|",end=" ")
            out = out + 1
            if(out % 16 == 0):
                print("\n")
"""

"""
out = 0
for i in range(1,257):
    print("{:>3}".format(i),"|",end=" ")
    out = out + 1
    if(out % 16 == 0):
        print("\n")
"""


n1 = (1 << 8)
n2 = (1 <<16)
print(n2 / 512 ,n1,(1 << 18) / n2 , n2 / (1<<10))