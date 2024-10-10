#!/bin/bash

# 初始化总和变量和计数器
sum1=0
sum2=0
count=10

# 定义 a 和 b 的范围
as=(12 15 17 18 21 23)
# 13 14 15 16 17 18 19 20 21 22 23 24)
#(12 13 14 15 16 17)
bs=(2 5 8)
#(1 4 8 16 32 64)
#(1 4 8 16 32 64)

# 对于 a 和 b 的每一个组合
for a in "${as[@]}"
do
    for b in "${bs[@]}"
    do
        # 重置每轮循环中的总和
        sum2=0
        
        # 运行可执行文件 count 次并累计输出值
        for ((i=1; i<=count; i++)); do
            output=$( ./build/bin/gpu_4step_negative_ntt_examples $a $b )
            sum2=$(echo "$sum2 + $output" | bc)
        done
        
        # 计算平均值
        average2=$(echo "scale=2; $sum2 / $count / $b" | bc)
        
        # 输出结果
        echo "For a=$a and b=$b, 4-step is: $average2"
    done
done