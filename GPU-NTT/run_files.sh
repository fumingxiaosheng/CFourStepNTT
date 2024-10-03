#!/bin/bash

# 初始化总和变量和计数器
sum1=0
sum2=0
count=50

a=$1
b=$2

# 运行可执行文件 count 次并累计输出值
for ((i=1; i<=count; i++)); do

    output=$( ./build/bin/gpu_4step_ntt_examples $a $b)
    sum2=$(echo "$sum2 + $output" | bc)
    
    #output=$( ./build/bin/gpu_merge_ntt_examples $a $b)
    #sum1=$(echo "$sum1 + $output" | bc)

    
done

# 计算平均值
average1=$(echo "scale=2; $sum1 / $count" | bc)
average2=$(echo "scale=2; $sum2 / $count" | bc)

# 输出结果
echo "merge is: $average1"
echo "4-step is: $average2"