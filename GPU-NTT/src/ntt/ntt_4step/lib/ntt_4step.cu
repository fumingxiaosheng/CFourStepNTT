// (C) Ulvetanna Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan
// Paper: https://eprint.iacr.org/2023/1410

#include "ntt_4step.cuh"

#include "hxw_constant.cuh" 

//extern __constant__ Root Csitable64[32];
__device__ void CooleyTukeyUnit_hxw(Data& U_out, Data& V_out,Data& U, Data& V, Root& root, Modulus& modulus)
{
    Data u_ = U;
    Data v_ = VALUE_GPU::mult(V, root, modulus);

    U_out = VALUE_GPU::add(u_, v_, modulus);
    V_out = VALUE_GPU::sub(u_, v_, modulus);

    /*Data u_ = 0;
    Data v_ = 0;

    U_out = u_;
    V_out = v_;*/ //实验结果表明,基于GPU实现的加、减等，耗时占比不多

}

__device__ void CooleyTukeyUnit_(Data& U, Data& V, Root& root, Modulus& modulus)
{
    Data u_ = U;
    Data v_ = VALUE_GPU::mult(V, root, modulus);

    U = VALUE_GPU::add(u_, v_, modulus);
    V = VALUE_GPU::sub(u_, v_, modulus);
}

__device__ void GentlemanSandeUnit_(Data& U, Data& V, Root& root, Modulus& modulus)
{
    Data u_ = U;
    Data v_ = V;

    U = VALUE_GPU::add(u_, v_, modulus);

    v_ = VALUE_GPU::sub(u_, v_, modulus);
    V = VALUE_GPU::mult(v_, root, modulus);
}

/*2024-8-7:
完成矩阵转置的操作
输入:polynomial_in是一个row*col的矩阵,polynomial_out是一个col*row的矩阵
操作流程:每一个block负责读出16*16的矩阵,然后将该子矩阵放回到polynomial_out中
线程组织形式:dim3(col >> 4, row >> 4, batch_size), dim3(16, 16)*/
__global__ void Transpose_Batch(Data* polynomial_in, Data* polynomial_out, const int row,
                                const int col, int n_power)
{
    int idx_x = threadIdx.x;  // 16 对应col
    int idx_y = threadIdx.y;  // 16 对应row

    int block_x = blockIdx.x * blockDim.x;
    int block_y = blockIdx.y * blockDim.y;

    int divindex = blockIdx.z << n_power;  //z维度由batch_size指定，用于定位polynominal_in的起始位置
    

    __shared__ Data sharedmemorys[16][18]; //TODO:32*32的形式更加合适(padding的计算) -> 考虑只padding1个

    sharedmemorys[idx_y][idx_x] =
        polynomial_in[((block_y + idx_y) * col) + block_x + idx_x + divindex]; //从row*col中取出(block.x,block.y)处的16*16矩阵 TODO:跨步内存访问，是否会存在相应的问题
    __syncthreads();

    polynomial_out[((block_x + idx_y) * row) + block_y + idx_x + divindex] =
        sharedmemorys[idx_x][idx_y]; //16*16的特质使得能够互换idx_x和y

    //TODO: 64 * 64 -> 转置的工作借鉴一下 切片的 , 共享内存的大小 -> x * 32 （trade off 对于全局内存的读写 一个block多多处理一下 或者32 * 32，考虑1个线程处理多个系数 优先解决访存问题）

    //TODO:上传git；分支维护 tag(版本号、自定义) wip: 未完成的commit ；由commit message、tag来寻找具体版本；CI（服务器，需要使用自己的服务器）
}

/*2024-8-7:
输入:input为输入的BATCH个多项式
    row代表的是行数
    col代表列数
    n_power代表logn
    batch_size代表的是并发的ntt个数*/
__host__ void GPU_Transpose(Data* input, Data* output, const int row, const int col,
                            const int n_power, const int batch_size)
{
    Transpose_Batch<<<dim3(col >> 4, row >> 4, batch_size), dim3(16, 16)>>>(input, output, row, col, n_power);
    THROW_IF_CUDA_ERROR(cudaGetLastError());
}

/*2024-8-7:
参数:polynomial_in:n2*n1维矩阵
    polynomial_out:n1*n2维矩阵
    mod_count:模数的modulus个数
    index1:TODO猜测是logn2
    n1_root_of_unity_table:n1阶本原单位根的次方的比特翻转的顺序

线程组织形式:dim3(4, batch_size), dim3(32, 8) //4096分为4份,每份1024个系数，在共享内存中以32*32的形式存在，32*8个线程完成32个32维NTT

处理流程:计算NTT结果，并完成转置操作，以n1*n2维（在线性空间中的跨度为n2）的形式存储在polynomial_out中
*/
__global__ void FourStepForwardCoreT1(Data* polynomial_in, Data* polynomial_out,
                                      Root* n1_root_of_unity_table, Modulus* modulus, int index1, int index2, int n_power, int mod_count)
{
    int idx_x = threadIdx.x;
    int idx_y = threadIdx.y;
    int block_x = blockIdx.x;
    int block_y = blockIdx.y; //和batch_size相对应

    __shared__ Data sharedmemorys[32][32 + 1]; //每一个block用于填满sharedmemorys

    int q_index = block_y % mod_count; //moduls共用一个
    Modulus q_thread = modulus[q_index];//q_thread代表的是模数

    //把4096分成4段,每一段对应一个block,(block_x << 10)对应每个半区的开始,idx_x代表一个block中的线程数(256),因此，每一个线程需要load4个数进来

    int idx_index = idx_x + (idx_y << 5); //0-256
    int global_addresss = idx_index + (block_x << 10); //分成4个半区
    int divindex = block_y << n_power; //用于区分是第几个batch

    // Load data from global & store to shared
    
    //每一个共享内存的一行,对应着在多项式中连续的32个系数
    sharedmemorys[idx_y][idx_x] = polynomial_in[global_addresss + divindex];
    sharedmemorys[idx_y + 8][idx_x] = polynomial_in[global_addresss + 256 + divindex];
    sharedmemorys[idx_y + 16][idx_x] = polynomial_in[global_addresss + 512 + divindex];
    sharedmemorys[idx_y + 24][idx_x] = polynomial_in[global_addresss + 768 + divindex];
    __syncthreads();

    int global_index1 = idx_x >> 4; //32以内的数除以16
    int global_index2 = idx_x % 16;

    int t_ = 4;
    int t = 1 << t_;
    int in_shared_address = ((global_index2 >> t_) << t_) + global_index2; //本质上就等于global_index2

    //NTT操作:in_shared_address对应着矩阵sharedmemorys的一列，该列需要完成32个CT操作。in_shared_address的范围为0-16，因此两个idx_x映射到同一个列。因此global_index2对应了哪一列，global_index1对应着该idx_x是处理奇数行还是偶数行（因此行位置为(idx_y << 1) + global_index1，又因为每一列16个线程，因此每个线程做两个CT，再加上16），(global_index2 >> t_代表着做的是第几组CT，又因为n1_root_of_unity_table已经是比特翻转顺序，因此直接索引其下标即可
    CooleyTukeyUnit_(sharedmemorys[(idx_y << 1) + global_index1][in_shared_address],
                    sharedmemorys[(idx_y << 1) + global_index1][in_shared_address + t],
                    n1_root_of_unity_table[(global_index2 >> t_)], q_thread);
    CooleyTukeyUnit_(sharedmemorys[(idx_y << 1) + global_index1 + 16][in_shared_address],
                    sharedmemorys[(idx_y << 1) + global_index1 + 16][in_shared_address + t],
                    n1_root_of_unity_table[(global_index2 >> t_)], q_thread);
    __syncthreads();

    for(int i = 0; i < 4; i++)
    {
        t = t >> 1;
        t_ -= 1;

        in_shared_address = ((global_index2 >> t_) << t_) + global_index2;//二分

        CooleyTukeyUnit_(sharedmemorys[(idx_y << 1) + global_index1][in_shared_address],
                        sharedmemorys[(idx_y << 1) + global_index1][in_shared_address + t],
                        n1_root_of_unity_table[(global_index2 >> t_)], q_thread);
        CooleyTukeyUnit_(sharedmemorys[(idx_y << 1) + global_index1 + 16][in_shared_address],
                        sharedmemorys[(idx_y << 1) + global_index1 + 16][in_shared_address + t],
                        n1_root_of_unity_table[(global_index2 >> t_)], q_thread);
        __syncthreads();
    }
    __syncthreads();

    //假设polynomial_out中的下标为(a,b),那么写入到polynomial_out中的位置为(block_x*32+a) + b * 32
    //每个线程负责4个数的写入
    polynomial_out[idx_x + (idx_y << index1) + (block_x << 5) + divindex] =
        sharedmemorys[idx_x][idx_y];
    polynomial_out[idx_x + (idx_y << index1) + (block_x << 5) + index2 + divindex] =
        sharedmemorys[idx_x][idx_y + 8];
    polynomial_out[idx_x + (idx_y << index1) + (block_x << 5) + (index2 * 2) + divindex] =
        sharedmemorys[idx_x][idx_y + 16];
    polynomial_out[idx_x + (idx_y << index1) + (block_x << 5) + (index2 * 3) + divindex] =
        sharedmemorys[idx_x][idx_y + 24];
}

__global__ void FourStepForwardCoreT1(Data* polynomial_in, Data* polynomial_out,
                                      Root* n1_root_of_unity_table, Modulus modulus, int index1,
                                      int index2, int n_power)
{
    int idx_x = threadIdx.x;
    int idx_y = threadIdx.y;
    int block_x = blockIdx.x;
    int block_y = blockIdx.y;

    __shared__ Data sharedmemorys[32][32 + 1];

    Modulus q_thread = modulus;

    int idx_index = idx_x + (idx_y << 5);
    int global_addresss = idx_index + (block_x << 10);
    int divindex = block_y << n_power;

    // Load data from global & store to shared
    sharedmemorys[idx_y][idx_x] = polynomial_in[global_addresss + divindex];
    sharedmemorys[idx_y + 8][idx_x] = polynomial_in[global_addresss + 256 + divindex];
    sharedmemorys[idx_y + 16][idx_x] = polynomial_in[global_addresss + 512 + divindex];
    sharedmemorys[idx_y + 24][idx_x] = polynomial_in[global_addresss + 768 + divindex];
    __syncthreads();

    int global_index1 = idx_x >> 4;
    int global_index2 = idx_x % 16;

    int t_ = 4;
    int t = 1 << t_;
    int in_shared_address = ((global_index2 >> t_) << t_) + global_index2;

    CooleyTukeyUnit_(sharedmemorys[(idx_y << 1) + global_index1][in_shared_address],
                    sharedmemorys[(idx_y << 1) + global_index1][in_shared_address + t],
                    n1_root_of_unity_table[(global_index2 >> t_)], q_thread);
    CooleyTukeyUnit_(sharedmemorys[(idx_y << 1) + global_index1 + 16][in_shared_address],
                    sharedmemorys[(idx_y << 1) + global_index1 + 16][in_shared_address + t],
                    n1_root_of_unity_table[(global_index2 >> t_)], q_thread);
    __syncthreads();

    for(int i = 0; i < 4; i++)
    {
        t = t >> 1;
        t_ -= 1;

        in_shared_address = ((global_index2 >> t_) << t_) + global_index2;

        CooleyTukeyUnit_(sharedmemorys[(idx_y << 1) + global_index1][in_shared_address],
                        sharedmemorys[(idx_y << 1) + global_index1][in_shared_address + t],
                        n1_root_of_unity_table[(global_index2 >> t_)], q_thread);
        CooleyTukeyUnit_(sharedmemorys[(idx_y << 1) + global_index1 + 16][in_shared_address],
                        sharedmemorys[(idx_y << 1) + global_index1 + 16][in_shared_address + t],
                        n1_root_of_unity_table[(global_index2 >> t_)], q_thread);
        __syncthreads();
    }
    __syncthreads();

    polynomial_out[idx_x + (idx_y << index1) + (block_x << 5) + divindex] =
        sharedmemorys[idx_x][idx_y];
    polynomial_out[idx_x + (idx_y << index1) + (block_x << 5) + index2 + divindex] =
        sharedmemorys[idx_x][idx_y + 8];
    polynomial_out[idx_x + (idx_y << index1) + (block_x << 5) + (index2 * 2) + divindex] =
        sharedmemorys[idx_x][idx_y + 16];
    polynomial_out[idx_x + (idx_y << index1) + (block_x << 5) + (index2 * 3) + divindex] =
        sharedmemorys[idx_x][idx_y + 24];
}

/*2024-8-19:
完成64维NTT操作*/
__global__ void FourStepForwardCoreT2(Data* polynomial_in, Data* polynomial_out,
                                      Root* n1_root_of_unity_table, Modulus* modulus, int index1,
                                      int index2, int index3, int n_power, int mod_count)
{
    int idx_x = threadIdx.x;
    int idx_y = threadIdx.y;
    int block_x = blockIdx.x;
    int block_y = blockIdx.y;

    __shared__ Data sharedmemorys[16][64 + 1];

    int q_index = block_y % mod_count;
    Modulus q_thread = modulus[q_index];

    int idx_index = idx_x + (idx_y << 5);
    int global_addresss = idx_index + (block_x << 10);
    int divindex = block_y << n_power;

    int shr_index1 = idx_y >> 1;
    int shr_index2 = idx_y % 2;

    // Load data from global & store to shared
    sharedmemorys[shr_index1][idx_x + (shr_index2 << 5)] =
        polynomial_in[global_addresss + divindex];
    sharedmemorys[shr_index1 + 4][idx_x + (shr_index2 << 5)] =
        polynomial_in[global_addresss + 256 + divindex];
    sharedmemorys[shr_index1 + 8][idx_x + (shr_index2 << 5)] =
        polynomial_in[global_addresss + 512 + divindex];
    sharedmemorys[shr_index1 + 12][idx_x + (shr_index2 << 5)] =
        polynomial_in[global_addresss + 768 + divindex];
    __syncthreads();

    int t_ = 5;
    int t = 1 << t_;
    int in_shared_address = ((idx_x >> t_) << t_) + idx_x;

    CooleyTukeyUnit_(sharedmemorys[idx_y][in_shared_address],
                    sharedmemorys[idx_y][in_shared_address + t],
                    n1_root_of_unity_table[(idx_x >> t_)], q_thread);
    CooleyTukeyUnit_(sharedmemorys[idx_y + 8][in_shared_address],
                    sharedmemorys[idx_y + 8][in_shared_address + t],
                    n1_root_of_unity_table[(idx_x >> t_)], q_thread);
    __syncthreads();

    for(int i = 0; i < 5; i++)
    {
        t = t >> 1;
        t_ -= 1;

        in_shared_address = ((idx_x >> t_) << t_) + idx_x;
        ;

        CooleyTukeyUnit_(sharedmemorys[idx_y][in_shared_address],
                        sharedmemorys[idx_y][in_shared_address + t],
                        n1_root_of_unity_table[(idx_x >> t_)], q_thread);
        CooleyTukeyUnit_(sharedmemorys[idx_y + 8][in_shared_address],
                        sharedmemorys[idx_y + 8][in_shared_address + t],
                        n1_root_of_unity_table[(idx_x >> t_)], q_thread);
        __syncthreads();
    }
    __syncthreads();

    int global_index1 = idx_x >> 4;
    int global_index2 = idx_x % 16;

    polynomial_out[global_index2 + (global_index1 << index1) + (idx_y << index2) + (block_x << 4) +
                   divindex] = sharedmemorys[global_index2][global_index1 + (idx_y << 1)];
    polynomial_out[global_index2 + (global_index1 << index1) + (idx_y << index2) + (block_x << 4) +
                   index3 + divindex] =
        sharedmemorys[global_index2][global_index1 + (idx_y << 1) + 16];
    polynomial_out[global_index2 + (global_index1 << index1) + (idx_y << index2) + (block_x << 4) +
                   (index3 * 2) + divindex] =
        sharedmemorys[global_index2][global_index1 + (idx_y << 1) + 32];
    polynomial_out[global_index2 + (global_index1 << index1) + (idx_y << index2) + (block_x << 4) +
                   (index3 * 3) + divindex] =
        sharedmemorys[global_index2][global_index1 + (idx_y << 1) + 48];
}

__global__ void FourStepForwardCoreT2(Data* polynomial_in, Data* polynomial_out,
                                      Root* n1_root_of_unity_table, Modulus modulus, int index1,
                                      int index2, int index3, int n_power)
{
    int idx_x = threadIdx.x;
    int idx_y = threadIdx.y;
    int block_x = blockIdx.x;
    int block_y = blockIdx.y;

    __shared__ Data sharedmemorys[16][64 + 1];

    Modulus q_thread = modulus;

    int idx_index = idx_x + (idx_y << 5);
    int global_addresss = idx_index + (block_x << 10);
    int divindex = block_y << n_power;

    int shr_index1 = idx_y >> 1;
    int shr_index2 = idx_y % 2;

    // Load data from global & store to shared
    sharedmemorys[shr_index1][idx_x + (shr_index2 << 5)] =
        polynomial_in[global_addresss + divindex];
    sharedmemorys[shr_index1 + 4][idx_x + (shr_index2 << 5)] =
        polynomial_in[global_addresss + 256 + divindex];
    sharedmemorys[shr_index1 + 8][idx_x + (shr_index2 << 5)] =
        polynomial_in[global_addresss + 512 + divindex];
    sharedmemorys[shr_index1 + 12][idx_x + (shr_index2 << 5)] =
        polynomial_in[global_addresss + 768 + divindex];
    __syncthreads();

    int t_ = 5;
    int t = 1 << t_;
    int in_shared_address = ((idx_x >> t_) << t_) + idx_x;

    CooleyTukeyUnit_(sharedmemorys[idx_y][in_shared_address],
                    sharedmemorys[idx_y][in_shared_address + t],
                    n1_root_of_unity_table[(idx_x >> t_)], q_thread);
    CooleyTukeyUnit_(sharedmemorys[idx_y + 8][in_shared_address],
                    sharedmemorys[idx_y + 8][in_shared_address + t],
                    n1_root_of_unity_table[(idx_x >> t_)], q_thread);
    __syncthreads();

    for(int i = 0; i < 5; i++)
    {
        t = t >> 1;
        t_ -= 1;

        in_shared_address = ((idx_x >> t_) << t_) + idx_x;
        ;

        CooleyTukeyUnit_(sharedmemorys[idx_y][in_shared_address],
                        sharedmemorys[idx_y][in_shared_address + t],
                        n1_root_of_unity_table[(idx_x >> t_)], q_thread);
        CooleyTukeyUnit_(sharedmemorys[idx_y + 8][in_shared_address],
                        sharedmemorys[idx_y + 8][in_shared_address + t],
                        n1_root_of_unity_table[(idx_x >> t_)], q_thread);
        __syncthreads();
    }
    __syncthreads();

    int global_index1 = idx_x >> 4;
    int global_index2 = idx_x % 16;

    polynomial_out[global_index2 + (global_index1 << index1) + (idx_y << index2) + (block_x << 4) +
                   divindex] = sharedmemorys[global_index2][global_index1 + (idx_y << 1)];
    polynomial_out[global_index2 + (global_index1 << index1) + (idx_y << index2) + (block_x << 4) +
                   index3 + divindex] =
        sharedmemorys[global_index2][global_index1 + (idx_y << 1) + 16];
    polynomial_out[global_index2 + (global_index1 << index1) + (idx_y << index2) + (block_x << 4) +
                   (index3 * 2) + divindex] =
        sharedmemorys[global_index2][global_index1 + (idx_y << 1) + 32];
    polynomial_out[global_index2 + (global_index1 << index1) + (idx_y << index2) + (block_x << 4) +
                   (index3 * 3) + divindex] =
        sharedmemorys[global_index2][global_index1 + (idx_y << 1) + 48];
}

/*2024-8-19:
完成128维NTT操作*/
__global__ void FourStepForwardCoreT3(Data* polynomial_in, Data* polynomial_out,
                                      Root* n1_root_of_unity_table, Modulus* modulus, int index1,
                                      int index2, int index3, int n_power, int mod_count)
{
    int idx_x = threadIdx.x;
    int idx_y = threadIdx.y;
    int block_x = blockIdx.x;
    int block_y = blockIdx.y;

    __shared__ Data sharedmemorys[8][128 + 1];

    int q_index = block_y % mod_count;
    Modulus q_thread = modulus[q_index];

    int idx_index = idx_x + (idx_y << 5);
    int global_addresss = idx_index + (block_x << 10);
    int divindex = block_y << n_power;

    int shr_index1 = idx_y >> 2;
    int shr_index2 = idx_y % 4;

    // Load data from global & store to shared
    sharedmemorys[shr_index1][idx_x + (shr_index2 << 5)] =
        polynomial_in[global_addresss + divindex];
    sharedmemorys[shr_index1 + 2][idx_x + (shr_index2 << 5)] =
        polynomial_in[global_addresss + 256 + divindex];
    sharedmemorys[shr_index1 + 4][idx_x + (shr_index2 << 5)] =
        polynomial_in[global_addresss + 512 + divindex];
    sharedmemorys[shr_index1 + 6][idx_x + (shr_index2 << 5)] =
        polynomial_in[global_addresss + 768 + divindex];
    __syncthreads();

    int shr_address = idx_x + ((idx_y % 2) << 5);
    int shr_in = idx_y >> 1;

    int t_ = 6;
    int t = 1 << t_;
    int in_shared_address = ((shr_address >> t_) << t_) + shr_address;

    CooleyTukeyUnit_(sharedmemorys[shr_in][in_shared_address],
                    sharedmemorys[shr_in][in_shared_address + t],
                    n1_root_of_unity_table[(shr_address >> t_)], q_thread);
    CooleyTukeyUnit_(sharedmemorys[shr_in + 4][in_shared_address],
                    sharedmemorys[shr_in + 4][in_shared_address + t],
                    n1_root_of_unity_table[(shr_address >> t_)], q_thread);
    __syncthreads();

    for(int i = 0; i < 6; i++)
    {
        t = t >> 1;
        t_ -= 1;

        in_shared_address = ((shr_address >> t_) << t_) + shr_address;

        CooleyTukeyUnit_(sharedmemorys[shr_in][in_shared_address],
                        sharedmemorys[shr_in][in_shared_address + t],
                        n1_root_of_unity_table[(shr_address >> t_)], q_thread);
        CooleyTukeyUnit_(sharedmemorys[shr_in + 4][in_shared_address],
                        sharedmemorys[shr_in + 4][in_shared_address + t],
                        n1_root_of_unity_table[(shr_address >> t_)], q_thread);
        __syncthreads();
    }
    __syncthreads();

    int global_index1 = idx_x >> 3;
    int global_index2 = idx_x % 8;

    polynomial_out[global_index2 + (global_index1 << index1) + (idx_y << index2) + (block_x << 3) +
                   divindex] = sharedmemorys[global_index2][global_index1 + (idx_y << 2)];
    polynomial_out[global_index2 + (global_index1 << index1) + (idx_y << index2) + (block_x << 3) +
                   index3 + divindex] =
        sharedmemorys[global_index2][global_index1 + (idx_y << 2) + 32];
    polynomial_out[global_index2 + (global_index1 << index1) + (idx_y << index2) + (block_x << 3) +
                   (index3 * 2) + divindex] =
        sharedmemorys[global_index2][global_index1 + (idx_y << 2) + 64];
    polynomial_out[global_index2 + (global_index1 << index1) + (idx_y << index2) + (block_x << 3) +
                   (index3 * 3) + divindex] =
        sharedmemorys[global_index2][global_index1 + (idx_y << 2) + 96];
}

__global__ void FourStepForwardCoreT3(Data* polynomial_in, Data* polynomial_out,
                                      Root* n1_root_of_unity_table, Modulus modulus, int index1,
                                      int index2, int index3, int n_power)
{
    int idx_x = threadIdx.x;
    int idx_y = threadIdx.y;
    int block_x = blockIdx.x;
    int block_y = blockIdx.y;

    __shared__ Data sharedmemorys[8][128 + 1];

    Modulus q_thread = modulus;

    int idx_index = idx_x + (idx_y << 5);
    int global_addresss = idx_index + (block_x << 10);
    int divindex = block_y << n_power;

    int shr_index1 = idx_y >> 2;
    int shr_index2 = idx_y % 4;

    // Load data from global & store to shared
    sharedmemorys[shr_index1][idx_x + (shr_index2 << 5)] =
        polynomial_in[global_addresss + divindex];
    sharedmemorys[shr_index1 + 2][idx_x + (shr_index2 << 5)] =
        polynomial_in[global_addresss + 256 + divindex];
    sharedmemorys[shr_index1 + 4][idx_x + (shr_index2 << 5)] =
        polynomial_in[global_addresss + 512 + divindex];
    sharedmemorys[shr_index1 + 6][idx_x + (shr_index2 << 5)] =
        polynomial_in[global_addresss + 768 + divindex];
    __syncthreads();

    int shr_address = idx_x + ((idx_y % 2) << 5);
    int shr_in = idx_y >> 1;

    int t_ = 6;
    int t = 1 << t_;
    int in_shared_address = ((shr_address >> t_) << t_) + shr_address;

    CooleyTukeyUnit_(sharedmemorys[shr_in][in_shared_address],
                    sharedmemorys[shr_in][in_shared_address + t],
                    n1_root_of_unity_table[(shr_address >> t_)], q_thread);
    CooleyTukeyUnit_(sharedmemorys[shr_in + 4][in_shared_address],
                    sharedmemorys[shr_in + 4][in_shared_address + t],
                    n1_root_of_unity_table[(shr_address >> t_)], q_thread);
    __syncthreads();

    for(int i = 0; i < 6; i++)
    {
        t = t >> 1;
        t_ -= 1;

        in_shared_address = ((shr_address >> t_) << t_) + shr_address;

        CooleyTukeyUnit_(sharedmemorys[shr_in][in_shared_address],
                        sharedmemorys[shr_in][in_shared_address + t],
                        n1_root_of_unity_table[(shr_address >> t_)], q_thread);
        CooleyTukeyUnit_(sharedmemorys[shr_in + 4][in_shared_address],
                        sharedmemorys[shr_in + 4][in_shared_address + t],
                        n1_root_of_unity_table[(shr_address >> t_)], q_thread);
        __syncthreads();
    }
    __syncthreads();

    int global_index1 = idx_x >> 3;
    int global_index2 = idx_x % 8;

    polynomial_out[global_index2 + (global_index1 << index1) + (idx_y << index2) + (block_x << 3) +
                   divindex] = sharedmemorys[global_index2][global_index1 + (idx_y << 2)];
    polynomial_out[global_index2 + (global_index1 << index1) + (idx_y << index2) + (block_x << 3) +
                   index3 + divindex] =
        sharedmemorys[global_index2][global_index1 + (idx_y << 2) + 32];
    polynomial_out[global_index2 + (global_index1 << index1) + (idx_y << index2) + (block_x << 3) +
                   (index3 * 2) + divindex] =
        sharedmemorys[global_index2][global_index1 + (idx_y << 2) + 64];
    polynomial_out[global_index2 + (global_index1 << index1) + (idx_y << index2) + (block_x << 3) +
                   (index3 * 3) + divindex] =
        sharedmemorys[global_index2][global_index1 + (idx_y << 2) + 96];
}

/*2024-8-19:
完成256维NTT操作*/
__global__ void FourStepForwardCoreT4(Data* polynomial_in, Data* polynomial_out,
                                      Root* n1_root_of_unity_table, Modulus* modulus, int index1,
                                      int index2, int index3, int n_power, int mod_count)
{
    int idx_x = threadIdx.x;
    int idx_y = threadIdx.y;
    int block_x = blockIdx.x;
    int block_y = blockIdx.y;

    __shared__ Data sharedmemorys[4][256 + 1];

    int q_index = block_y % mod_count;
    Modulus q_thread = modulus[q_index];

    int idx_index = idx_x + (idx_y << 5);
    int global_addresss = idx_index + (block_x << 10);
    int divindex = block_y << n_power;

    int shr_index1 = idx_y >> 3;
    int shr_index2 = idx_y % 8;

    // Load data from global & store to shared
    sharedmemorys[shr_index1][idx_x + (shr_index2 << 5)] =
        polynomial_in[global_addresss + divindex];
    sharedmemorys[shr_index1 + 1][idx_x + (shr_index2 << 5)] =
        polynomial_in[global_addresss + 256 + divindex];
    sharedmemorys[shr_index1 + 2][idx_x + (shr_index2 << 5)] =
        polynomial_in[global_addresss + 512 + divindex];
    sharedmemorys[shr_index1 + 3][idx_x + (shr_index2 << 5)] =
        polynomial_in[global_addresss + 768 + divindex];
    __syncthreads();

    int shr_address = idx_x + ((idx_y % 4) << 5);
    int shr_in = idx_y >> 2;

    int t_ = 7;
    int t = 1 << t_;
    int in_shared_address = ((shr_address >> t_) << t_) + shr_address;

    CooleyTukeyUnit_(sharedmemorys[shr_in][in_shared_address],
                    sharedmemorys[shr_in][in_shared_address + t],
                    n1_root_of_unity_table[(shr_address >> t_)], q_thread);
    CooleyTukeyUnit_(sharedmemorys[shr_in + 2][in_shared_address],
                    sharedmemorys[shr_in + 2][in_shared_address + t],
                    n1_root_of_unity_table[(shr_address >> t_)], q_thread);
    __syncthreads();

    for(int i = 0; i < 7; i++)
    {
        t = t >> 1;
        t_ -= 1;

        in_shared_address = ((shr_address >> t_) << t_) + shr_address;

        CooleyTukeyUnit_(sharedmemorys[shr_in][in_shared_address],
                        sharedmemorys[shr_in][in_shared_address + t],
                        n1_root_of_unity_table[(shr_address >> t_)], q_thread);
        CooleyTukeyUnit_(sharedmemorys[shr_in + 2][in_shared_address],
                        sharedmemorys[shr_in + 2][in_shared_address + t],
                        n1_root_of_unity_table[(shr_address >> t_)], q_thread);
        __syncthreads();
    }
    __syncthreads();

    int global_index1 = idx_x >> 2;
    int global_index2 = idx_x % 4;

    polynomial_out[global_index2 + (global_index1 << index1) + (idx_y << index2) + (block_x << 2) +
                   divindex] = sharedmemorys[global_index2][global_index1 + (idx_y << 3)];
    polynomial_out[global_index2 + (global_index1 << index1) + (idx_y << index2) + (block_x << 2) +
                   index3 + divindex] =
        sharedmemorys[global_index2][global_index1 + (idx_y << 3) + 64];
    polynomial_out[global_index2 + (global_index1 << index1) + (idx_y << index2) + (block_x << 2) +
                   (index3 * 2) + divindex] =
        sharedmemorys[global_index2][global_index1 + (idx_y << 3) + 128];
    polynomial_out[global_index2 + (global_index1 << index1) + (idx_y << index2) + (block_x << 2) +
                   (index3 * 3) + divindex] =
        sharedmemorys[global_index2][global_index1 + (idx_y << 3) + 192];
}

__global__ void FourStepForwardCoreT4(Data* polynomial_in, Data* polynomial_out,
                                      Root* n1_root_of_unity_table, Modulus modulus, int index1,
                                      int index2, int index3, int n_power)
{
    int idx_x = threadIdx.x;
    int idx_y = threadIdx.y;
    int block_x = blockIdx.x;
    int block_y = blockIdx.y;

    __shared__ Data sharedmemorys[4][256 + 1];

    Modulus q_thread = modulus;

    int idx_index = idx_x + (idx_y << 5);
    int global_addresss = idx_index + (block_x << 10);
    int divindex = block_y << n_power;

    int shr_index1 = idx_y >> 3;
    int shr_index2 = idx_y % 8;

    // Load data from global & store to shared
    sharedmemorys[shr_index1][idx_x + (shr_index2 << 5)] =
        polynomial_in[global_addresss + divindex];
    sharedmemorys[shr_index1 + 1][idx_x + (shr_index2 << 5)] =
        polynomial_in[global_addresss + 256 + divindex];
    sharedmemorys[shr_index1 + 2][idx_x + (shr_index2 << 5)] =
        polynomial_in[global_addresss + 512 + divindex];
    sharedmemorys[shr_index1 + 3][idx_x + (shr_index2 << 5)] =
        polynomial_in[global_addresss + 768 + divindex];
    __syncthreads();

    int shr_address = idx_x + ((idx_y % 4) << 5);
    int shr_in = idx_y >> 2;

    int t_ = 7;
    int t = 1 << t_;
    int in_shared_address = ((shr_address >> t_) << t_) + shr_address;

    CooleyTukeyUnit_(sharedmemorys[shr_in][in_shared_address],
                    sharedmemorys[shr_in][in_shared_address + t],
                    n1_root_of_unity_table[(shr_address >> t_)], q_thread);
    CooleyTukeyUnit_(sharedmemorys[shr_in + 2][in_shared_address],
                    sharedmemorys[shr_in + 2][in_shared_address + t],
                    n1_root_of_unity_table[(shr_address >> t_)], q_thread);
    __syncthreads();

    for(int i = 0; i < 7; i++)
    {
        t = t >> 1;
        t_ -= 1;

        in_shared_address = ((shr_address >> t_) << t_) + shr_address;

        CooleyTukeyUnit_(sharedmemorys[shr_in][in_shared_address],
                        sharedmemorys[shr_in][in_shared_address + t],
                        n1_root_of_unity_table[(shr_address >> t_)], q_thread);
        CooleyTukeyUnit_(sharedmemorys[shr_in + 2][in_shared_address],
                        sharedmemorys[shr_in + 2][in_shared_address + t],
                        n1_root_of_unity_table[(shr_address >> t_)], q_thread);
        __syncthreads();
    }
    __syncthreads();

    int global_index1 = idx_x >> 2;
    int global_index2 = idx_x % 4;

    polynomial_out[global_index2 + (global_index1 << index1) + (idx_y << index2) + (block_x << 2) +
                   divindex] = sharedmemorys[global_index2][global_index1 + (idx_y << 3)];
    polynomial_out[global_index2 + (global_index1 << index1) + (idx_y << index2) + (block_x << 2) +
                   index3 + divindex] =
        sharedmemorys[global_index2][global_index1 + (idx_y << 3) + 64];
    polynomial_out[global_index2 + (global_index1 << index1) + (idx_y << index2) + (block_x << 2) +
                   (index3 * 2) + divindex] =
        sharedmemorys[global_index2][global_index1 + (idx_y << 3) + 128];
    polynomial_out[global_index2 + (global_index1 << index1) + (idx_y << index2) + (block_x << 2) +
                   (index3 * 3) + divindex] =
        sharedmemorys[global_index2][global_index1 + (idx_y << 3) + 192];
}

/*2024-8-8:

线程组织形式:dim3(8, 32, batch_size), dim3(64, 4)
完成n1个n2维NTT操作中的前loop层

参数:polynomial_in:n1*n2维的多项式矩阵
    loc2:代表的是首次读取时的间隔的长度


[例子:32个4096操作]
12, 6, 2048, 3, 17, mod_count
线程组织结构:
dim3(8, 32, batch_size), dim3(64, 4)

[例子:] 32个2^{15}个NTT操作
FourStepPartialForwardCore1<<<dim3(64, 32, batch_size), dim3(8, 32)>>>(
                        device_out, n2_root_of_unity_table, W_root_of_unity_table, modulus, 15, 3,
                        16384, 6, 20, mod_count);

*/
__global__ void FourStepPartialForwardCore1(Data* polynomial_in, Root* n2_root_of_unity_table,
                                            Root* w_root_of_unity_table, Modulus* modulus,
                                            int small_npower, int loc1, int loc2, int loop,
                                            int n_power, int mod_count)
{
    int idx_x = threadIdx.x;
    int idx_y = threadIdx.y;
    int block_x = blockIdx.x;
    int block_y = blockIdx.y;
    int block_z = blockIdx.z;

    __shared__ Data sharedmemorys[512];//4096个系数 分成了8个半区; 32768个系数 分成64个区域

    int n_power__ = small_npower;
    int t_2 = n_power__ - 1;

    int q_index = block_z % mod_count;
    Modulus q_thread = modulus[q_index];

    int grid = (block_y << n_power__); //第几个n2
    int divindex = block_z << n_power; //第几个n

    int global_addresss = (idx_y << 9) + idx_x + (block_x << loc1); //在4096中的哪一个位置 //以512为间隔,取出连续的8个数,由于以512为间隔，因此最多能做6层NTT操作
    int shared_addresss = (idx_x + (idx_y << loc1)); //定位存储到共享内存中的哪一个位置 //一个idx_y对应了8个系数,由idx_x进行标识

    int load_store_address = global_addresss + grid; //能够去形成合并内存的访问

    Data mult_1 = polynomial_in[load_store_address + divindex];
    Data mult_2 = polynomial_in[load_store_address + loc2 + divindex];//loc2是间隔，第一次读的时候以2048为间隔大小来读

    mult_1 = VALUE_GPU::mult(mult_1, w_root_of_unity_table[load_store_address], q_thread);
    mult_2 = VALUE_GPU::mult(mult_2, w_root_of_unity_table[load_store_address + loc2], q_thread);

    sharedmemorys[shared_addresss] = mult_1;
    sharedmemorys[shared_addresss + 256] = mult_2;

    //下面开始做NTT操作,64 * 4 = 256个线程共同完成512维的NTT操作
    int t_ = 8;
    int t = 1 << t_; //t代表的是在共享内存中的间隔大小
    int in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;

    for(int lp = 0; lp < loop; lp++) //512能够做9层的NTT操作,但是这里只做了3层
    {
        CooleyTukeyUnit_(sharedmemorys[in_shared_address], sharedmemorys[in_shared_address + t],
                        n2_root_of_unity_table[(global_addresss >> t_2)], q_thread);
        __syncthreads();

        t = t >> 1; //间隔取半
        t_2 -= 1; //旋转因子下标减1
        t_ -= 1; //用于二分每个线程负责的下标的位置

        in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
    }
    __syncthreads();

    // Load data from shared & store to global
    polynomial_in[load_store_address + divindex] = sharedmemorys[shared_addresss];
    polynomial_in[load_store_address + loc2 + divindex] = sharedmemorys[shared_addresss + 256];
}

__global__ void FourStepPartialForwardCore1(Data* polynomial_in, Root* n2_root_of_unity_table,
                                            Root* w_root_of_unity_table, Modulus modulus,
                                            int small_npower, int loc1, int loc2, int loop,
                                            int n_power)
{
    int idx_x = threadIdx.x;
    int idx_y = threadIdx.y;
    int block_x = blockIdx.x;
    int block_y = blockIdx.y;
    int block_z = blockIdx.z;

    __shared__ Data sharedmemorys[512];

    int n_power__ = small_npower;
    int t_2 = n_power__ - 1;

    Modulus q_thread = modulus;

    int grid = (block_y << n_power__);
    int divindex = block_z << n_power;

    int global_addresss = (idx_y << 9) + idx_x + (block_x << loc1);
    int shared_addresss = (idx_x + (idx_y << loc1));

    int load_store_address = global_addresss + grid;

    Data mult_1 = polynomial_in[load_store_address + divindex];
    Data mult_2 = polynomial_in[load_store_address + loc2 + divindex];

    mult_1 = VALUE_GPU::mult(mult_1, w_root_of_unity_table[load_store_address], q_thread);
    mult_2 = VALUE_GPU::mult(mult_2, w_root_of_unity_table[load_store_address + loc2], q_thread);

    sharedmemorys[shared_addresss] = mult_1;
    sharedmemorys[shared_addresss + 256] = mult_2;

    int t_ = 8;
    int t = 1 << t_;
    int in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;

    for(int lp = 0; lp < loop; lp++)
    {
        CooleyTukeyUnit_(sharedmemorys[in_shared_address], sharedmemorys[in_shared_address + t],
                        n2_root_of_unity_table[(global_addresss >> t_2)], q_thread);
        __syncthreads();

        t = t >> 1;
        t_2 -= 1;
        t_ -= 1;

        in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
    }
    __syncthreads();

    // Load data from shared & store to global
    polynomial_in[load_store_address + divindex] = sharedmemorys[shared_addresss];
    polynomial_in[load_store_address + loc2 + divindex] = sharedmemorys[shared_addresss + 256];
}

/*由32个线程同步完成*/
__device__ void six_NTT(Data * coes,Data * n64_root_of_unity_table, Modulus q_thread){
    CooleyTukeyUnit_(coes[threadIdx.y], coes[threadIdx.y + 32], n64_root_of_unity_table[0], q_thread);
    CooleyTukeyUnit_(coes[((threadIdx.y >> 4) << 5) + (threadIdx.y & 15)], coes[((threadIdx.y >> 4) << 5) + (threadIdx.y & 15) + 16], n64_root_of_unity_table[threadIdx.y >> 4], q_thread);
    CooleyTukeyUnit_(coes[((threadIdx.y >> 3) << 4) + (threadIdx.y & 7)], coes[((threadIdx.y >> 3) << 4) + (threadIdx.y & 7) + 8], n64_root_of_unity_table[threadIdx.y >> 3], q_thread);
    CooleyTukeyUnit_(coes[((threadIdx.y >> 2) << 3) + (threadIdx.y & 3)], coes[((threadIdx.y >> 2) << 3) + (threadIdx.y & 3) + 4], n64_root_of_unity_table[threadIdx.y >> 2], q_thread);
    CooleyTukeyUnit_(coes[((threadIdx.y >> 1) << 2) + (threadIdx.y & 1)], coes[((threadIdx.y >> 1) << 2) + (threadIdx.y & 1) + 2], n64_root_of_unity_table[threadIdx.y >> 1], q_thread);
    CooleyTukeyUnit_(coes[((threadIdx.y) << 1)], coes[((threadIdx.y) << 1) + 1], n64_root_of_unity_table[threadIdx.y], q_thread);
}
/*2024-8-9:
线程组织结构 <<<(n/4096,batch_size),(32,8)>>>
*/
__global__ void FourStepFowardCoreFull(Data* polynomial_in, Root * n64_root_of_unity_table, Root *n64_W_inverse_root_of_unity_table, Root * W_root_of_unity_table, int mod_count ,int n_power, Modulus* modulus){
    __shared__ Data coes[64][64];

    int q_index = blockIdx.y % mod_count; //moduls共用一个
    Modulus q_thread = modulus[q_index];//q_thread代表的是模数

    //一个block有256个线程，因此每一个线程需要load 16个系数

    int i = 0;

    int global_start = blockIdx.x * UNITY_SIZE1 *UNITY_SIZE2 + (blockIdx.y << n_power ); //注意:这里应该不会有溢出

    //256个线程一起放到共享内存
    //printf("x:%d y:%d\n",threadIdx.x,threadIdx.y);
#pragma unroll
    for(i=0;i<(UNITY_SIZE1 * UNITY_SIZE2 / (blockDim.x * blockDim.y));i++){
        coes[(threadIdx.x * 8 + threadIdx.y + i * (blockDim.x * blockDim.y)) / UNITY_SIZE2][(threadIdx.x * 8 + threadIdx.y + i * (blockDim.x * blockDim.y)) % UNITY_SIZE2] = VALUE_GPU::mult(polynomial_in[threadIdx.x * 8 + threadIdx.y + i * (blockDim.x * blockDim.y) + global_start], W_root_of_unity_table[threadIdx.x * 8 + threadIdx.y + i * (blockDim.x * blockDim.y) + blockIdx.x * UNITY_SIZE1 * UNITY_SIZE2],q_thread);//形成合并内存访问

        /*coes[(threadIdx.x * 8 + threadIdx.y + i * (blockDim.x * blockDim.y)) / UNITY_SIZE2][(threadIdx.x * 8 + threadIdx.y + i * (blockDim.x * blockDim.y)) % UNITY_SIZE2] = polynomial_in[threadIdx.x * 8 + threadIdx.y + i * (blockDim.x * blockDim.y) + global_start];*///形成合并内存访问
        //printf("%d,",threadIdx.x * 8 + threadIdx.y + i * (blockDim.x * blockDim.y));
    }

    __syncthreads();//DEBUG:必须要加一个内存栅栏
    

    //下面再64*64的矩阵上做操作
    Data a[8];
    Data b[8];


#pragma unroll
    for(i=0;i<8;i++){
        a[i] = coes[threadIdx.y + 8 * i][threadIdx.x];
        b[i] = coes[threadIdx.y + 8 * i][threadIdx.x + 32];
    }
    //第一层
    CooleyTukeyUnit_(a[0], a[4], n64_root_of_unity_table[0], q_thread);
    CooleyTukeyUnit_(a[1], a[5], n64_root_of_unity_table[0], q_thread);
    CooleyTukeyUnit_(a[2], a[6], n64_root_of_unity_table[0], q_thread);
    CooleyTukeyUnit_(a[3], a[7], n64_root_of_unity_table[0], q_thread);

    CooleyTukeyUnit_(b[0], b[4], n64_root_of_unity_table[0], q_thread);
    CooleyTukeyUnit_(b[1], b[5], n64_root_of_unity_table[0], q_thread);
    CooleyTukeyUnit_(b[2], b[6], n64_root_of_unity_table[0], q_thread);
    CooleyTukeyUnit_(b[3], b[7], n64_root_of_unity_table[0], q_thread);

    //第二层
    CooleyTukeyUnit_(a[0], a[2], n64_root_of_unity_table[0], q_thread);
    CooleyTukeyUnit_(a[1], a[3], n64_root_of_unity_table[0], q_thread);
    CooleyTukeyUnit_(a[4], a[6], n64_root_of_unity_table[1], q_thread);
    CooleyTukeyUnit_(a[5], a[7], n64_root_of_unity_table[1], q_thread);

    CooleyTukeyUnit_(b[0], b[2], n64_root_of_unity_table[0], q_thread);
    CooleyTukeyUnit_(b[1], b[3], n64_root_of_unity_table[0], q_thread);
    CooleyTukeyUnit_(b[4], b[6], n64_root_of_unity_table[1], q_thread);
    CooleyTukeyUnit_(b[5], b[7], n64_root_of_unity_table[1], q_thread);

    //第三层
    CooleyTukeyUnit_(a[0], a[1], n64_root_of_unity_table[0], q_thread);
    CooleyTukeyUnit_(a[2], a[3], n64_root_of_unity_table[1], q_thread);
    CooleyTukeyUnit_(a[4], a[5], n64_root_of_unity_table[2], q_thread);
    CooleyTukeyUnit_(a[6], a[7], n64_root_of_unity_table[3], q_thread);

    CooleyTukeyUnit_(b[0], b[1], n64_root_of_unity_table[0], q_thread);
    CooleyTukeyUnit_(b[2], b[3], n64_root_of_unity_table[1], q_thread);
    CooleyTukeyUnit_(b[4], b[5], n64_root_of_unity_table[2], q_thread);
    CooleyTukeyUnit_(b[6], b[7], n64_root_of_unity_table[3], q_thread);

#pragma unroll
    for(i=0;i<8;i++){
        coes[threadIdx.y + 8 * i][threadIdx.x] = a[i];
        coes[threadIdx.y + 8 * i][threadIdx.x + 32] = b[i];
    }

    __syncthreads();

#pragma unroll
    for(i=0;i<8;i++){
        a[i] = coes[threadIdx.y * 8 + i][threadIdx.x];
        b[i] = coes[threadIdx.y * 8 + i][threadIdx.x + 32];
    }

    //第四层NTT
    CooleyTukeyUnit_(a[0], a[4], n64_root_of_unity_table[threadIdx.y], q_thread);
    CooleyTukeyUnit_(a[1], a[5], n64_root_of_unity_table[threadIdx.y], q_thread);
    CooleyTukeyUnit_(a[2], a[6], n64_root_of_unity_table[threadIdx.y], q_thread);
    CooleyTukeyUnit_(a[3], a[7], n64_root_of_unity_table[threadIdx.y], q_thread);

    CooleyTukeyUnit_(b[0], b[4], n64_root_of_unity_table[threadIdx.y], q_thread);
    CooleyTukeyUnit_(b[1], b[5], n64_root_of_unity_table[threadIdx.y], q_thread);
    CooleyTukeyUnit_(b[2], b[6], n64_root_of_unity_table[threadIdx.y], q_thread);
    CooleyTukeyUnit_(b[3], b[7], n64_root_of_unity_table[threadIdx.y], q_thread);

    //第五层NTT
    CooleyTukeyUnit_(a[0], a[2], n64_root_of_unity_table[threadIdx.y * 2], q_thread);
    CooleyTukeyUnit_(a[1], a[3], n64_root_of_unity_table[threadIdx.y * 2], q_thread);
    CooleyTukeyUnit_(a[4], a[6], n64_root_of_unity_table[threadIdx.y * 2 + 1], q_thread);
    CooleyTukeyUnit_(a[5], a[7], n64_root_of_unity_table[threadIdx.y * 2 + 1], q_thread);

    CooleyTukeyUnit_(b[0], b[2], n64_root_of_unity_table[threadIdx.y * 2], q_thread);
    CooleyTukeyUnit_(b[1], b[3], n64_root_of_unity_table[threadIdx.y * 2], q_thread);
    CooleyTukeyUnit_(b[4], b[6], n64_root_of_unity_table[threadIdx.y * 2 + 1], q_thread);
    CooleyTukeyUnit_(b[5], b[7], n64_root_of_unity_table[threadIdx.y * 2 + 1], q_thread);

    //第六层NTT
    CooleyTukeyUnit_(a[0], a[1], n64_root_of_unity_table[threadIdx.y * 4], q_thread);
    CooleyTukeyUnit_(a[2], a[3], n64_root_of_unity_table[threadIdx.y * 4 + 1], q_thread);
    CooleyTukeyUnit_(a[4], a[5], n64_root_of_unity_table[threadIdx.y * 4 + 2], q_thread);
    CooleyTukeyUnit_(a[6], a[7], n64_root_of_unity_table[threadIdx.y * 4 + 3], q_thread);

    CooleyTukeyUnit_(b[0], b[1], n64_root_of_unity_table[threadIdx.y * 4], q_thread);
    CooleyTukeyUnit_(b[2], b[3], n64_root_of_unity_table[threadIdx.y * 4 + 1], q_thread);
    CooleyTukeyUnit_(b[4], b[5], n64_root_of_unity_table[threadIdx.y * 4 + 2], q_thread);
    CooleyTukeyUnit_(b[6], b[7], n64_root_of_unity_table[threadIdx.y * 4 + 3], q_thread);

    //乘上补偿因子，再写回到共享内存中

#pragma unroll
    for(i=0;i<8;i++){
        coes[threadIdx.y * 8 + i][threadIdx.x] = VALUE_GPU::mult(a[i], n64_W_inverse_root_of_unity_table[(threadIdx.y * 8 + i) * UNITY_SIZE2 + threadIdx.x], q_thread);
        coes[threadIdx.y * 8 + i][threadIdx.x + 32] = VALUE_GPU::mult(b[i], n64_W_inverse_root_of_unity_table[(threadIdx.y * 8 + i) * UNITY_SIZE2 + threadIdx.x + 32], q_thread);
    }
    __syncthreads();

    /*if(threadIdx.x == 0 && threadIdx.y ==0){
        for(int i=0;i<64;i++){
            for(int j=0;j<64;j++){
                printf("%lld,",coes[i][j]);
            }
            printf("\n");
        }
    }

    __syncthreads();*/


    //横着做64维NTT

#pragma unroll
    for(i=0;i<8;i++){
        a[i] = coes[threadIdx.x][threadIdx.y + 8 * i];
        b[i] = coes[threadIdx.x + 32][threadIdx.y + 8 * i];
    }
    //第一层
    CooleyTukeyUnit_(a[0], a[4], n64_root_of_unity_table[0], q_thread);
    CooleyTukeyUnit_(a[1], a[5], n64_root_of_unity_table[0], q_thread);
    CooleyTukeyUnit_(a[2], a[6], n64_root_of_unity_table[0], q_thread);
    CooleyTukeyUnit_(a[3], a[7], n64_root_of_unity_table[0], q_thread);

    CooleyTukeyUnit_(b[0], b[4], n64_root_of_unity_table[0], q_thread);
    CooleyTukeyUnit_(b[1], b[5], n64_root_of_unity_table[0], q_thread);
    CooleyTukeyUnit_(b[2], b[6], n64_root_of_unity_table[0], q_thread);
    CooleyTukeyUnit_(b[3], b[7], n64_root_of_unity_table[0], q_thread);

    //第二层
    CooleyTukeyUnit_(a[0], a[2], n64_root_of_unity_table[0], q_thread);
    CooleyTukeyUnit_(a[1], a[3], n64_root_of_unity_table[0], q_thread);
    CooleyTukeyUnit_(a[4], a[6], n64_root_of_unity_table[1], q_thread);
    CooleyTukeyUnit_(a[5], a[7], n64_root_of_unity_table[1], q_thread);

    CooleyTukeyUnit_(b[0], b[2], n64_root_of_unity_table[0], q_thread);
    CooleyTukeyUnit_(b[1], b[3], n64_root_of_unity_table[0], q_thread);
    CooleyTukeyUnit_(b[4], b[6], n64_root_of_unity_table[1], q_thread);
    CooleyTukeyUnit_(b[5], b[7], n64_root_of_unity_table[1], q_thread);

    //第三层
    CooleyTukeyUnit_(a[0], a[1], n64_root_of_unity_table[0], q_thread);
    CooleyTukeyUnit_(a[2], a[3], n64_root_of_unity_table[1], q_thread);
    CooleyTukeyUnit_(a[4], a[5], n64_root_of_unity_table[2], q_thread);
    CooleyTukeyUnit_(a[6], a[7], n64_root_of_unity_table[3], q_thread);

    CooleyTukeyUnit_(b[0], b[1], n64_root_of_unity_table[0], q_thread);
    CooleyTukeyUnit_(b[2], b[3], n64_root_of_unity_table[1], q_thread);
    CooleyTukeyUnit_(b[4], b[5], n64_root_of_unity_table[2], q_thread);
    CooleyTukeyUnit_(b[6], b[7], n64_root_of_unity_table[3], q_thread);

#pragma unroll
    for(i=0;i<8;i++){
        coes[threadIdx.x][threadIdx.y + 8 * i] = a[i];
        coes[threadIdx.x + 32][threadIdx.y + 8 * i] = b[i];
    }
    __syncthreads();


#pragma unroll
    for(i=0;i<8;i++){
        a[i] = coes[threadIdx.x][threadIdx.y * 8 + i];
        b[i] = coes[threadIdx.x + 32][threadIdx.y * 8 + i];
    }

    //第四层NTT
    CooleyTukeyUnit_(a[0], a[4], n64_root_of_unity_table[threadIdx.y], q_thread);
    CooleyTukeyUnit_(a[1], a[5], n64_root_of_unity_table[threadIdx.y], q_thread);
    CooleyTukeyUnit_(a[2], a[6], n64_root_of_unity_table[threadIdx.y], q_thread);
    CooleyTukeyUnit_(a[3], a[7], n64_root_of_unity_table[threadIdx.y], q_thread);

    CooleyTukeyUnit_(b[0], b[4], n64_root_of_unity_table[threadIdx.y], q_thread);
    CooleyTukeyUnit_(b[1], b[5], n64_root_of_unity_table[threadIdx.y], q_thread);
    CooleyTukeyUnit_(b[2], b[6], n64_root_of_unity_table[threadIdx.y], q_thread);
    CooleyTukeyUnit_(b[3], b[7], n64_root_of_unity_table[threadIdx.y], q_thread);

    //第五层NTT
    CooleyTukeyUnit_(a[0], a[2], n64_root_of_unity_table[threadIdx.y * 2], q_thread);
    CooleyTukeyUnit_(a[1], a[3], n64_root_of_unity_table[threadIdx.y * 2], q_thread);
    CooleyTukeyUnit_(a[4], a[6], n64_root_of_unity_table[threadIdx.y * 2 + 1], q_thread);
    CooleyTukeyUnit_(a[5], a[7], n64_root_of_unity_table[threadIdx.y * 2 + 1], q_thread);

    CooleyTukeyUnit_(b[0], b[2], n64_root_of_unity_table[threadIdx.y * 2], q_thread);
    CooleyTukeyUnit_(b[1], b[3], n64_root_of_unity_table[threadIdx.y * 2], q_thread);
    CooleyTukeyUnit_(b[4], b[6], n64_root_of_unity_table[threadIdx.y * 2 + 1], q_thread);
    CooleyTukeyUnit_(b[5], b[7], n64_root_of_unity_table[threadIdx.y * 2 + 1], q_thread);

    //第六层NTT
    CooleyTukeyUnit_(a[0], a[1], n64_root_of_unity_table[threadIdx.y * 4], q_thread);
    CooleyTukeyUnit_(a[2], a[3], n64_root_of_unity_table[threadIdx.y * 4 + 1], q_thread);
    CooleyTukeyUnit_(a[4], a[5], n64_root_of_unity_table[threadIdx.y * 4 + 2], q_thread);
    CooleyTukeyUnit_(a[6], a[7], n64_root_of_unity_table[threadIdx.y * 4 + 3], q_thread);

    CooleyTukeyUnit_(b[0], b[1], n64_root_of_unity_table[threadIdx.y * 4], q_thread);
    CooleyTukeyUnit_(b[2], b[3], n64_root_of_unity_table[threadIdx.y * 4 + 1], q_thread);
    CooleyTukeyUnit_(b[4], b[5], n64_root_of_unity_table[threadIdx.y * 4 + 2], q_thread);
    CooleyTukeyUnit_(b[6], b[7], n64_root_of_unity_table[threadIdx.y * 4 + 3], q_thread);

/*#pragma unroll
    for(i=0;i<8;i++){
        coes[threadIdx.x][threadIdx.y * 8 + i] = a[i];
        coes[threadIdx.x + 32][threadIdx.y * 8 + i] = b[i];
    }

    if(threadIdx.x == 0 && threadIdx.y ==0){
        for(int i=0;i<64;i++){
            for(int j=0;j<64;j++){
                printf("%lld,",coes[i][j]);
            }
            printf("\n");
        }
    }

    __syncthreads();*/

    //直接放回全局内存
    for(i=0;i<8;i++){
        polynomial_in[threadIdx.y * 8 + i + threadIdx.x * UNITY_SIZE2 + global_start] = a[i];
        polynomial_in[threadIdx.y * 8 + i + (threadIdx.x + 32) * UNITY_SIZE2 + global_start] = b[i];
    }

}

/*2024-8-11
线程组织结构<<<(n2/64 / 4),(32,4)>>> 复用多个64维的NTT
polynimal_in:n2*n1维度
polynomial_out:n1*n2维度
*/
__global__ void FourStepFowardCore12_1(Data* polynomial_in, Data* polynomial_out,Root * n64_root_of_unity_table, Root *n64_W_inverse_root_of_unity_table, Root * W_root_of_unity_table, int mod_count ,int n_power, Modulus* modulus){
    //TODO:写一下
    __shared__ Data coes[4][64];

    int i=0;
    int q_index = blockIdx.y % mod_count; //moduls共用一个
    Modulus q_thread = modulus[q_index];//q_thread代表的是模数

    int global_start = blockIdx.x * 64 * blockDim.y + (blockIdx.y << n_power );
    
    coes[threadIdx.y][threadIdx.x] = polynomial_in[global_start + (threadIdx.y << 6) + threadIdx.x]; //TODO:最开始的时候就应该转置
    coes[threadIdx.y][threadIdx.x + 32] = polynomial_in[global_start + (threadIdx.y << 6) + threadIdx.x + 32];

    //__syncthreads();
    /*if(blockIdx.x == 0){
        printf("%d,%d %lld %lld\n",threadIdx.y,threadIdx.x,coes[threadIdx.y][threadIdx.x],coes[threadIdx.y][threadIdx.x + 32]);
    }
    if(blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 3){
        printf("\n\n");
        for(int i=0;i<4096;i++){
            printf("%d,",polynomial_in[i]);
        }
        printf("%lld %lld\n\n\n",coes[1][0],coes[threadIdx.y][threadIdx.x]);
        for(int i=0;i<4;i++){
            for(int j=0;j<64;j++){
                printf("%lld,",coes[i][j]);
            }
            printf("\n");
        }
    }*/
    //__syncthreads();
    /*if(blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 3){
        for(int i=0;i<64;i++){
            printf("%lld," ,coes[threadIdx.y][i]);
        }
        printf("\n\n");
    }*/
#pragma unroll

    for(i=5;i>=0;i--){
        /*if(blockIdx.x == 0 && threadIdx.x == 31 && threadIdx.y == 3){
            printf("%lld %lld %lld ",coes[threadIdx.y][((threadIdx.x >> i) << (i+1)) + (threadIdx.x & ((1<<i)-1))],coes[threadIdx.y][((threadIdx.x >> i) << (i+1)) + (threadIdx.x & ((1<<i)-1)) + (1<<i)],n64_root_of_unity_table[threadIdx.x >> i]);
        }*/
        CooleyTukeyUnit_(coes[threadIdx.y][((threadIdx.x >> i) << (i+1)) + (threadIdx.x & ((1<<i)-1))], coes[threadIdx.y][((threadIdx.x >> i) << (i+1)) + (threadIdx.x & ((1<<i)-1)) + (1<<i)], n64_root_of_unity_table[threadIdx.x >> i], q_thread); //TODO:感觉后面全部都溢出了,在无符号的情况下是有点溢出的

        /*if(blockIdx.x == 0 && threadIdx.x == 31 && threadIdx.y == 3){
            printf("%lld %lld\n",coes[threadIdx.y][((threadIdx.x >> i) << (i+1)) + (threadIdx.x & ((1<<i)-1))],coes[threadIdx.y][((threadIdx.x >> i) << (i+1)) + (threadIdx.x & ((1<<i)-1)) + (1<<i)]);
        }

        if(blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 3){
        for(int i=0;i<64;i++){
            printf("%lld," ,coes[threadIdx.y][i]);
        }
        printf("\n\n");
    }*/

       // __syncthreads();
    }
    //32个线程完成,无需进行syncthreads();
    
    
    //(a,b)对应着矩阵里的(a + blockIdx.x * blockDim.y ,b) 对应反过来是(b,a + blockIdx.x * blockDim.y)
    polynomial_out[(threadIdx.x << 6) + threadIdx.y + blockIdx.x * blockDim.y] = VALUE_GPU::mult(coes[threadIdx.y][threadIdx.x],n64_W_inverse_root_of_unity_table[((threadIdx.x << 6) + threadIdx.y + blockIdx.x * blockDim.y)],q_thread);//DEBUG:这里应该使用n64_W_inverse_root_of_unity_table

    polynomial_out[((threadIdx.x + 32)<< 6) + threadIdx.y + blockIdx.x * blockDim.y] = VALUE_GPU::mult(coes[threadIdx.y][threadIdx.x + 32],n64_W_inverse_root_of_unity_table[(((threadIdx.x + 32)<< 6) + threadIdx.y + blockIdx.x * blockDim.y)],q_thread);//注意:这个补偿因子应该乘在哪里比较合适呢?
}

__global__ void FourStepFowardCore12_2_o_v1(Data* polynomial_in,Root * n64_root_of_unity_table, Root *n64_W_inverse_root_of_unity_table, Root * W_root_of_unity_table, int mod_count ,int n_power, Modulus* modulus){
    //TODO:写一下
    __shared__ Data coes[4][64]; //为了解决存储体冲突,在16he 48的后面padiing上一个Data

    int i=0;

    int q_index = blockIdx.y % mod_count; //moduls共用一个
    Modulus q_thread = modulus[q_index];//q_thread代表的是模数
    int global_start = blockIdx.x * 64 * blockDim.y + (blockIdx.y << n_power );

    coes[threadIdx.y][threadIdx.x] = polynomial_in[global_start + (threadIdx.y << 6) + threadIdx.x]; //TODO:最开始的时候就应该转置
    coes[threadIdx.y][threadIdx.x + 32] = polynomial_in[global_start + (threadIdx.y << 6) + threadIdx.x + 32];

    //TODO:注意删除 线程组织结构没有问题
#pragma unroll
    for(i=5;i>=0;i--){ //i为3的时候有存储体冲突
        CooleyTukeyUnit_(coes[threadIdx.y][((threadIdx.x >> i) << (i+1)) + (threadIdx.x & ((1<<i)-1))], coes[threadIdx.y][((threadIdx.x >> i) << (i+1)) + (threadIdx.x & ((1<<i)-1)) + (1<<i)], n64_root_of_unity_table[threadIdx.x >> i], q_thread);
    }
    //32个线程完成,无需进行syncthreads();
    
    polynomial_in[global_start + (threadIdx.y << 6) + threadIdx.x] = coes[threadIdx.y][threadIdx.x];
    polynomial_in[global_start + (threadIdx.y << 6) + threadIdx.x + 32] = coes[threadIdx.y][threadIdx.x + 32];

}

/*2024-8-17:
polynomial_in:n1*n2维，解决了load的存储体冲突的64维NTT版本，但是没有解决store的存储体冲突
线程组织形式:<<<(n1/64 / 4),(32,4)>>>*/
__global__ void FourStepFowardCore12_2_o_v2(Data* polynomial_in,Root * n64_root_of_unity_table, Root *n64_W_inverse_root_of_unity_table, Root * W_root_of_unity_table, int mod_count ,int n_power, Modulus* modulus){
    //TODO:写一下
    __shared__ Data coes[4][96]; //为了解决存储体冲突,在16he 48的后面padiing上一个Data
    
    int8_t idx1,idx2;

    int q_index = blockIdx.y % mod_count; //moduls共用一个
    Modulus q_thread = modulus[q_index];//q_thread代表的是模数
    int global_start = blockIdx.x * 64 * blockDim.y + (blockIdx.y << n_power );

    coes[threadIdx.y][threadIdx.x] = polynomial_in[global_start + (threadIdx.y << 6) + threadIdx.x]; //TODO:最开始的时候就应该转置
    coes[threadIdx.y][threadIdx.x + 32] = polynomial_in[global_start + (threadIdx.y << 6) + threadIdx.x + 32];

    //第一层蝴蝶操作
    //32个线程同步进行读，因此不会存在读写冲突的问题
    CooleyTukeyUnit_hxw(coes[threadIdx.y][threadIdx.x], coes[threadIdx.y][threadIdx.x + 32],coes[threadIdx.y][threadIdx.x], coes[threadIdx.y][threadIdx.x + 32], n64_root_of_unity_table[0], q_thread);
    
    //第二层蝴蝶操作
    idx1 = ((threadIdx.x >> 4)<<5) + (threadIdx.x & 15);
    idx2 = ((threadIdx.x >> 4)<<5) + (threadIdx.x & 15) + 16;
    CooleyTukeyUnit_hxw(coes[threadIdx.y][(idx1 >> 4) * 24 + (idx1 & 15) + (((idx1 & 15) >> 3) << 3)], coes[threadIdx.y][(idx2 >> 4) * 24 + (idx2 & 15) + (((idx2 & 15) >> 3) << 3)],coes[threadIdx.y][idx1], coes[threadIdx.y][idx2], n64_root_of_unity_table[threadIdx.x >> 4], q_thread);
    //store的时候开始出现存储体冲突 8192 二路冲突
    
    /*16个数中间插上8个空格 i -> (i /16) * 24 + (i % 16) +  ((i % 16) / 8) * 8
     1 |   2 |   3 |   4 |   5 |   6 |   7 |   8 |    *|    *|    *|    *|    *|    *|    *|    *| 

     9 |  10 |  11 |  12 |  13 |  14 |  15 |  16 |  17 |  18 |  19 |  20 |  21 |  22 |  23 |  24 | 

      *|    *|    *|    *|    *|    *|    *|    *|  25 |  26 |  27 |  28 |  29 |  30 |  31 |  32 | 

    33 |  34 |  35 |  36 |  37 |  38 |  39 |  40 |    *|    *|    *|    *|    *|    *|    *|    *| 

    41 |  42 |  43 |  44 |  45 |  46 |  47 |  48 |  49 |  50 |  51 |  52 |  53 |  54 |  55 |  56 | 

      *|    *|    *|    *|    *|    *|    *|    *|  57 |  58 |  59 |  60 |  61 |  62 |  63 |  64 | 
    */


    //第三层蝴蝶操作
    idx1 = ((threadIdx.x >> 3)<<4) + (threadIdx.x & 7);
    idx2 = ((threadIdx.x >> 3)<<4) + (threadIdx.x & 7) + 8;
    CooleyTukeyUnit_hxw(coes[threadIdx.y][(idx1 >> 3) * 12 + (idx1 & 7) + (((idx1 & 7) >> 2) << 2)], coes[threadIdx.y][(idx2 >> 3) * 12 + (idx2 & 7) + (((idx2 & 7) >> 2) << 2)],coes[threadIdx.y][(idx1 >> 4) * 24 + (idx1 & 15) + (((idx1 & 15) >> 3) << 3)], coes[threadIdx.y][(idx2 >> 4) * 24 + (idx2 & 15) + (((idx2 & 15) >> 3) << 3)], n64_root_of_unity_table[threadIdx.x >> 3], q_thread);
    // store的时候存在存储体冲突 16384

    /* 8个数中间插上4个空格 i -> (i / 8) * 12 + (i % 8) + ((i % 8) / 4) * 4
      1 |   2 |   3 |   4 |    *|    *|    *|    *|   5 |   6 |   7 |   8 |   9 |  10 |  11 |  12 | 

       *|    *|    *|    *|  13 |  14 |  15 |  16 |  17 |  18 |  19 |  20 |    *|    *|    *|    *| 

     21 |  22 |  23 |  24 |  25 |  26 |  27 |  28 |    *|    *|    *|    *|  29 |  30 |  31 |  32 | 

     33 |  34 |  35 |  36 |    *|    *|    *|    *|  37 |  38 |  39 |  40 |  41 |  42 |  43 |  44 | 

       *|    *|    *|    *|  45 |  46 |  47 |  48 |  49 |  50 |  51 |  52 |    *|    *|    *|    *| 

     53 |  54 |  55 |  56 |  57 |  58 |  59 |  60 |    *|    *|    *|    *|  61 |  62 |  63 |  64 | 
    */

    //第四层蝴蝶操作
    idx1 = ((threadIdx.x >> 2)<<3) + (threadIdx.x & 3);
    idx2 = ((threadIdx.x >> 2)<<3) + (threadIdx.x & 3) + 4;
    CooleyTukeyUnit_hxw(coes[threadIdx.y][(idx1 >> 2) * 6 + (idx1 & 3) + (((idx1 & 3) >> 1) << 1)],coes[threadIdx.y][(idx2 >> 2) * 6 + (idx2 & 3) + (((idx2 & 3) >> 1) << 1)],coes[threadIdx.y][(idx1 >> 3) * 12 + (idx1 & 7) + (((idx1 & 7) >> 2) << 2)] , coes[threadIdx.y][(idx2 >> 3) * 12 + (idx2 & 7) + (((idx2 & 7) >> 2) << 2)], n64_root_of_unity_table[threadIdx.x >> 2], q_thread);

    //store的时候存在存储体冲突24576

    /* 4个数中间插上2个空格 i -> (i / 4) * 6 + (i % 4) + ((i % 4) / 2) * 2
      1 |   2 |    *|    *|   3 |   4 |   5 |   6 |    *|    *|   7 |   8 |   9 |  10 |    *|    *| 

     11 |  12 |  13 |  14 |    *|    *|  15 |  16 |  17 |  18 |    *|    *|  19 |  20 |  21 |  22 | 

       *|    *|  23 |  24 |  25 |  26 |    *|    *|  27 |  28 |  29 |  30 |    *|    *|  31 |  32 | 

     33 |  34 |    *|    *|  35 |  36 |  37 |  38 |    *|    *|  39 |  40 |  41 |  42 |    *|    *| 

     43 |  44 |  45 |  46 |    *|    *|  47 |  48 |  49 |  50 |    *|    *|  51 |  52 |  53 |  54 | 

       *|    *|  55 |  56 |  57 |  58 |    *|    *|  59 |  60 |  61 |  62 |    *|    *|  63 |  64 | 
    */


   //第5层蝴蝶操作
    idx1 = ((threadIdx.x >> 1)<<2) + (threadIdx.x & 1);
    idx2 = ((threadIdx.x >> 1)<<2) + (threadIdx.x & 1) + 2;
    CooleyTukeyUnit_hxw(coes[threadIdx.y][(idx1 >> 1) * 3 + ((idx1 & 1) << 1)],coes[threadIdx.y][(idx2 >> 1) * 3 + ((idx2 & 1) << 1)],coes[threadIdx.y][(idx1 >> 2) * 6 + (idx1 & 3) + (((idx1 & 3) >> 1) << 1)],coes[threadIdx.y][(idx2 >> 2) * 6 + (idx2 & 3) + (((idx2 & 3) >> 1) << 1)], n64_root_of_unity_table[threadIdx.x >> 1], q_thread);

    //store的冲突 32768
    /*两个数中间插上1个空格 i -> (i / 2) * 3 + (i % 2) +  (i % 2)
      1 |    *|   2 |   3 |    *|   4 |   5 |    *|   6 |   7 |    *|   8 |   9 |    *|  10 |  11 | 

       *|  12 |  13 |    *|  14 |  15 |    *|  16 |  17 |    *|  18 |  19 |    *|  20 |  21 |    *| 

     22 |  23 |    *|  24 |  25 |    *|  26 |  27 |    *|  28 |  29 |    *|  30 |  31 |    *|  32 | 

     33 |    *|  34 |  35 |    *|  36 |  37 |    *|  38 |  39 |    *|  40 |  41 |    *|  42 |  43 | 

       *|  44 |  45 |    *|  46 |  47 |    *|  48 |  49 |    *|  50 |  51 |    *|  52 |  53 |    *| 

     54 |  55 |    *|  56 |  57 |    *|  58 |  59 |    *|  60 |  61 |    *|  62 |  63 |    *|  64 |
    */

    //第6层蝴蝶操作
    idx1 = threadIdx.x << 1;
    idx2 = (threadIdx.x << 1) + 1;
    CooleyTukeyUnit_hxw(coes[threadIdx.y][(idx1 >> 1) * 3 + ((idx1 & 1) << 1)],coes[threadIdx.y][(idx2 >> 1) * 3 + ((idx2 & 1) << 1)],coes[threadIdx.y][(idx1 >> 1) * 3 + ((idx1 & 1) << 1)],coes[threadIdx.y][(idx2 >> 1) * 3 + ((idx2 & 1) << 1)], n64_root_of_unity_table[threadIdx.x], q_thread);

    polynomial_in[global_start + (threadIdx.y << 6) + threadIdx.x] = coes[threadIdx.y][(threadIdx.x >> 1) * 3 + ((threadIdx.x & 1) << 1)];
    polynomial_in[global_start + (threadIdx.y << 6) + threadIdx.x + 32] = coes[threadIdx.y][((threadIdx.x + 32) >> 1) * 3 + (((threadIdx.x + 32) & 1) << 1)];

}


/*2024-8-18:
线程组织形式:<<<(n1/64 / 4),(32,4)
解决了存储体冲突的版本
*/
__global__ void FourStepFowardCore12_2(Data* polynomial_in,Root * n64_root_of_unity_table, Root *n64_W_inverse_root_of_unity_table, Root * W_root_of_unity_table, int mod_count ,int n_power, Modulus* modulus){
    
    __shared__ Data coes[4][96]; 

    int q_index = blockIdx.y % mod_count; //moduls共用一个
    Modulus q_thread = modulus[q_index];//q_thread代表的是模数
    int global_start = blockIdx.x * 64 * blockDim.y + (blockIdx.y << n_power );

    coes[threadIdx.y][threadIdx.x] = polynomial_in[global_start + (threadIdx.y << 6) + threadIdx.x]; //TODO:最开始的时候就应该转置
    coes[threadIdx.y][threadIdx.x + 32] = polynomial_in[global_start + (threadIdx.y << 6) + threadIdx.x + 32];

    //level 1
    CooleyTukeyUnit_hxw(coes[threadIdx.y][threadIdx.x], coes[threadIdx.y][threadIdx.x + 32],coes[threadIdx.y][threadIdx.x], coes[threadIdx.y][threadIdx.x + 32],n64_root_of_unity_table[0], q_thread);

    //level 2
    CooleyTukeyUnit_hxw(coes[threadIdx.y][(threadIdx.x >> 4) * 48 + (threadIdx.x & 15)], coes[threadIdx.y][(threadIdx.x >> 4) * 48 + (threadIdx.x & 15) + 24],coes[threadIdx.y][((threadIdx.x >> 4) << 5) + (threadIdx.x & 15)], coes[threadIdx.y][((threadIdx.x >> 4) << 5) + (threadIdx.x & 15) + 16],n64_root_of_unity_table[threadIdx.x >> 4], q_thread);

    //level 3
    CooleyTukeyUnit_hxw(coes[threadIdx.y][(threadIdx.x >> 3) * 24 + (threadIdx.x & 7)], coes[threadIdx.y][(threadIdx.x >> 3) * 24 + (threadIdx.x & 7) + 12],coes[threadIdx.y][(threadIdx.x >> 3) * 24 + (threadIdx.x & 7)], coes[threadIdx.y][(threadIdx.x >> 3) * 24 + (threadIdx.x & 7) + 8],n64_root_of_unity_table[threadIdx.x >> 3], q_thread);

    //level 4
    CooleyTukeyUnit_hxw(coes[threadIdx.y][(threadIdx.x >> 2) * 12 + (threadIdx.x & 3)], coes[threadIdx.y][(threadIdx.x >> 2) * 12 + (threadIdx.x & 3) + 6],coes[threadIdx.y][(threadIdx.x >> 2) * 12 + (threadIdx.x & 3)], coes[threadIdx.y][(threadIdx.x >> 2) * 12 + (threadIdx.x & 3) + 4], n64_root_of_unity_table[threadIdx.x >> 2], q_thread);

    //level 5
    CooleyTukeyUnit_hxw(coes[threadIdx.y][(threadIdx.x >> 1) * 6 + (threadIdx.x & 1)], coes[threadIdx.y][(threadIdx.x >> 1) * 6 + (threadIdx.x & 1) + 3],coes[threadIdx.y][(threadIdx.x >> 1) * 6 + (threadIdx.x & 1)], coes[threadIdx.y][(threadIdx.x >> 1) * 6 + (threadIdx.x & 1) + 2], n64_root_of_unity_table[threadIdx.x >> 1], q_thread);

    //level 6
    //CooleyTukeyUnit_hxw(coes[threadIdx.y][threadIdx.x * 3], coes[threadIdx.y][threadIdx.x * 3 + 1],coes[threadIdx.y][threadIdx.x * 3], coes[threadIdx.y][threadIdx.x * 3 + 1],root,q_thread);// n64_root_of_unity_table[threadIdx.x], q_thread);

    CooleyTukeyUnit_hxw(polynomial_in[global_start + (threadIdx.y << 6) + (threadIdx.x << 1)], polynomial_in[global_start + (threadIdx.y << 6) + (threadIdx.x << 1) + 1],coes[threadIdx.y][threadIdx.x * 3], coes[threadIdx.y][threadIdx.x * 3 + 1], n64_root_of_unity_table[threadIdx.x], q_thread); //这样确实没有存储体冲突了

    
    //合并内存访问，且不引起load的冲突 ->这里会引起共享内存的存储体冲突 TODO:为什么这里会有bank conflict?
    //polynomial_in[global_start + (threadIdx.y << 6) + threadIdx.x] = coes[threadIdx.y][(threadIdx.x >> 1) * 3 + (threadIdx.x & 1)];
    //polynomial_in[global_start + (threadIdx.y << 6) + threadIdx.x + 32] = coes[threadIdx.y][(threadIdx.x >> 1) * 3 + (threadIdx.x & 1) + 48];
}

/*2024-8-18:
使用寄存器代替共享内存,存在占用率过低的问题
(8,4)*/
__global__ void FourStepFowardCore12_2_o_v4(Data* polynomial_in,Root * n64_root_of_unity_table, Root *n64_W_inverse_root_of_unity_table, Root * W_root_of_unity_table, int mod_count ,int n_power, Modulus* modulus){
    
    __shared__ Data coes[4][64];

    Data coesl[8]; 
    int i;

    int q_index = blockIdx.y % mod_count; //moduls共用一个
    Modulus q_thread = modulus[q_index];//q_thread代表的是模数
    int global_start = blockIdx.x * 64 * blockDim.y + (blockIdx.y << n_power );

    coesl[0] = polynomial_in[global_start + (threadIdx.y << 6) + threadIdx.x];
    coesl[1] = polynomial_in[global_start + (threadIdx.y << 6) + threadIdx.x + 8];
    coesl[2] = polynomial_in[global_start + (threadIdx.y << 6) + threadIdx.x + 16];
    coesl[3] = polynomial_in[global_start + (threadIdx.y << 6) + threadIdx.x + 24];
    coesl[4] = polynomial_in[global_start + (threadIdx.y << 6) + threadIdx.x + 32];
    coesl[5] = polynomial_in[global_start + (threadIdx.y << 6) + threadIdx.x + 40];
    coesl[6] = polynomial_in[global_start + (threadIdx.y << 6) + threadIdx.x + 48];
    coesl[7] = polynomial_in[global_start + (threadIdx.y << 6) + threadIdx.x + 56];

    //level 1
    CooleyTukeyUnit_(coesl[0],coesl[4], n64_root_of_unity_table[0], q_thread);
    CooleyTukeyUnit_(coesl[1],coesl[5], n64_root_of_unity_table[0], q_thread);
    CooleyTukeyUnit_(coesl[2],coesl[6], n64_root_of_unity_table[0], q_thread);
    CooleyTukeyUnit_(coesl[3],coesl[7], n64_root_of_unity_table[0], q_thread);

    //level 2
    CooleyTukeyUnit_(coesl[0],coesl[2], n64_root_of_unity_table[0], q_thread);
    CooleyTukeyUnit_(coesl[1],coesl[3], n64_root_of_unity_table[0], q_thread);
    CooleyTukeyUnit_(coesl[4],coesl[6], n64_root_of_unity_table[1], q_thread);
    CooleyTukeyUnit_(coesl[5],coesl[7], n64_root_of_unity_table[1], q_thread);

    //level 3
    CooleyTukeyUnit_(coesl[0],coesl[1], n64_root_of_unity_table[0], q_thread);
    CooleyTukeyUnit_(coesl[2],coesl[3], n64_root_of_unity_table[1], q_thread);
    CooleyTukeyUnit_(coesl[4],coesl[5], n64_root_of_unity_table[2], q_thread);
    CooleyTukeyUnit_(coesl[6],coesl[7], n64_root_of_unity_table[3], q_thread);
#pragma unroll
    for(i = 0;i<8;i++){
        coes[threadIdx.y][threadIdx.x + i * 8] = coesl[i]; 
    }

#pragma unroll
    for(int i=0;i<8;i++){
        coesl[i] = coes[threadIdx.y][threadIdx.x * 8 + i];
    }
    
    //level 4
    CooleyTukeyUnit_(coesl[0],coesl[4], n64_root_of_unity_table[threadIdx.x], q_thread);
    CooleyTukeyUnit_(coesl[1],coesl[5], n64_root_of_unity_table[threadIdx.x], q_thread);
    CooleyTukeyUnit_(coesl[2],coesl[6], n64_root_of_unity_table[threadIdx.x], q_thread);
    CooleyTukeyUnit_(coesl[3],coesl[7], n64_root_of_unity_table[threadIdx.x], q_thread);

    //level 5
    CooleyTukeyUnit_(coesl[0],coesl[2], n64_root_of_unity_table[threadIdx.x * 2], q_thread);
    CooleyTukeyUnit_(coesl[1],coesl[3], n64_root_of_unity_table[threadIdx.x * 2], q_thread);
    CooleyTukeyUnit_(coesl[4],coesl[6], n64_root_of_unity_table[threadIdx.x * 2 + 1], q_thread);
    CooleyTukeyUnit_(coesl[5],coesl[7], n64_root_of_unity_table[threadIdx.x * 2 + 1], q_thread);

    //level 6
    CooleyTukeyUnit_(coesl[0],coesl[1], n64_root_of_unity_table[threadIdx.x * 4], q_thread);
    CooleyTukeyUnit_(coesl[2],coesl[3], n64_root_of_unity_table[threadIdx.x * 4 + 1], q_thread);
    CooleyTukeyUnit_(coesl[4],coesl[5], n64_root_of_unity_table[threadIdx.x * 4 + 2], q_thread);
    CooleyTukeyUnit_(coesl[6],coesl[7], n64_root_of_unity_table[threadIdx.x * 4 + 3], q_thread);
    
    #pragma unroll
    for(int i=0;i<8;i++){
         polynomial_in[global_start + (threadIdx.y << 6) + threadIdx.x * 8 + i] = coesl[i];
    }
    
}


/*2024-8-24:
线程组织形式:<<<(n1/64 / 4),(32,4)>>>
在解决了存储体冲突的基础上，使用了常量内存来存储64维NTT的旋转因子表
*/
__global__ void FourStepFowardCore12_2_o_v5(Data* polynomial_in,Root * n64_root_of_unity_table, Root *n64_W_inverse_root_of_unity_table, Root * W_root_of_unity_table, int mod_count ,int n_power, Modulus* modulus){
    
    __shared__ Data coes[4][96]; 

    int q_index = blockIdx.y % mod_count; //moduls共用一个
    Modulus q_thread = modulus[q_index];//q_thread代表的是模数
    int global_start = blockIdx.x * 64 * blockDim.y + (blockIdx.y << n_power );

    coes[threadIdx.y][threadIdx.x] = polynomial_in[global_start + (threadIdx.y << 6) + threadIdx.x]; //TODO:最开始的时候就应该转置
    coes[threadIdx.y][threadIdx.x + 32] = polynomial_in[global_start + (threadIdx.y << 6) + threadIdx.x + 32];

    //level 1
    CooleyTukeyUnit_hxw(coes[threadIdx.y][threadIdx.x], coes[threadIdx.y][threadIdx.x + 32],coes[threadIdx.y][threadIdx.x], coes[threadIdx.y][threadIdx.x + 32],Csitable64[0], q_thread);

    //level 2
    CooleyTukeyUnit_hxw(coes[threadIdx.y][(threadIdx.x >> 4) * 48 + (threadIdx.x & 15)], coes[threadIdx.y][(threadIdx.x >> 4) * 48 + (threadIdx.x & 15) + 24],coes[threadIdx.y][((threadIdx.x >> 4) << 5) + (threadIdx.x & 15)], coes[threadIdx.y][((threadIdx.x >> 4) << 5) + (threadIdx.x & 15) + 16],Csitable64[threadIdx.x >> 4], q_thread);

    //level 3
    CooleyTukeyUnit_hxw(coes[threadIdx.y][(threadIdx.x >> 3) * 24 + (threadIdx.x & 7)], coes[threadIdx.y][(threadIdx.x >> 3) * 24 + (threadIdx.x & 7) + 12],coes[threadIdx.y][(threadIdx.x >> 3) * 24 + (threadIdx.x & 7)], coes[threadIdx.y][(threadIdx.x >> 3) * 24 + (threadIdx.x & 7) + 8],Csitable64[threadIdx.x >> 3], q_thread);

    //level 4
    CooleyTukeyUnit_hxw(coes[threadIdx.y][(threadIdx.x >> 2) * 12 + (threadIdx.x & 3)], coes[threadIdx.y][(threadIdx.x >> 2) * 12 + (threadIdx.x & 3) + 6],coes[threadIdx.y][(threadIdx.x >> 2) * 12 + (threadIdx.x & 3)], coes[threadIdx.y][(threadIdx.x >> 2) * 12 + (threadIdx.x & 3) + 4], Csitable64[threadIdx.x >> 2], q_thread);

    //level 5
    CooleyTukeyUnit_hxw(coes[threadIdx.y][(threadIdx.x >> 1) * 6 + (threadIdx.x & 1)], coes[threadIdx.y][(threadIdx.x >> 1) * 6 + (threadIdx.x & 1) + 3],coes[threadIdx.y][(threadIdx.x >> 1) * 6 + (threadIdx.x & 1)], coes[threadIdx.y][(threadIdx.x >> 1) * 6 + (threadIdx.x & 1) + 2], Csitable64[threadIdx.x >> 1], q_thread);

    //level 6
    //CooleyTukeyUnit_hxw(coes[threadIdx.y][threadIdx.x * 3], coes[threadIdx.y][threadIdx.x * 3 + 1],coes[threadIdx.y][threadIdx.x * 3], coes[threadIdx.y][threadIdx.x * 3 + 1],root,q_thread);// Csitable64[threadIdx.x], q_thread);

    CooleyTukeyUnit_hxw(polynomial_in[global_start + (threadIdx.y << 6) + (threadIdx.x << 1)], polynomial_in[global_start + (threadIdx.y << 6) + (threadIdx.x << 1) + 1],coes[threadIdx.y][threadIdx.x * 3], coes[threadIdx.y][threadIdx.x * 3 + 1], Csitable64[threadIdx.x], q_thread); //这样确实没有存储体冲突了

    
    //合并内存访问，且不引起load的冲突 ->这里会引起共享内存的存储体冲突 TODO:为什么这里会有bank conflict?
    //polynomial_in[global_start + (threadIdx.y << 6) + threadIdx.x] = coes[threadIdx.y][(threadIdx.x >> 1) * 3 + (threadIdx.x & 1)];
    //polynomial_in[global_start + (threadIdx.y << 6) + threadIdx.x + 32] = coes[threadIdx.y][(threadIdx.x >> 1) * 3 + (threadIdx.x & 1) + 48];
}

/*2024-8-18:
256维NTT，线程组织形式(32,4)
最后将计算的结果转置一下
*/
__global__ void FourStepFowardCore256(Data* polynomial_in,Root * n1_root_of_unity_table, int mod_count ,int n_power, Modulus* modulus){
    
    __shared__ Data coes[4][256];

    Data coesl[8]; 
    int i;

    int q_index = blockIdx.y % mod_count; //moduls共用一个
    Modulus q_thread = modulus[q_index];//q_thread代表的是模数
    int global_start = blockIdx.x * 256 * blockDim.y + (blockIdx.y << n_power );

    coesl[0] = polynomial_in[global_start + (threadIdx.y << 8) + threadIdx.x];
    coesl[1] = polynomial_in[global_start + (threadIdx.y << 8) + threadIdx.x + 32];
    coesl[2] = polynomial_in[global_start + (threadIdx.y << 8) + threadIdx.x + 64];
    coesl[3] = polynomial_in[global_start + (threadIdx.y << 8) + threadIdx.x + 96];
    coesl[4] = polynomial_in[global_start + (threadIdx.y << 8) + threadIdx.x + 128];
    coesl[5] = polynomial_in[global_start + (threadIdx.y << 8) + threadIdx.x + 160];
    coesl[6] = polynomial_in[global_start + (threadIdx.y << 8) + threadIdx.x + 192];
    coesl[7] = polynomial_in[global_start + (threadIdx.y << 8) + threadIdx.x + 224];

    //level 1
    CooleyTukeyUnit_(coesl[0],coesl[4], n1_root_of_unity_table[0], q_thread);
    CooleyTukeyUnit_(coesl[1],coesl[5], n1_root_of_unity_table[0], q_thread);
    CooleyTukeyUnit_(coesl[2],coesl[6], n1_root_of_unity_table[0], q_thread);
    CooleyTukeyUnit_(coesl[3],coesl[7], n1_root_of_unity_table[0], q_thread);

    //level 2
    CooleyTukeyUnit_(coesl[0],coesl[2], n1_root_of_unity_table[0], q_thread);
    CooleyTukeyUnit_(coesl[1],coesl[3], n1_root_of_unity_table[0], q_thread);
    CooleyTukeyUnit_(coesl[4],coesl[6], n1_root_of_unity_table[1], q_thread);
    CooleyTukeyUnit_(coesl[5],coesl[7], n1_root_of_unity_table[1], q_thread);

    //level 3
    CooleyTukeyUnit_(coesl[0],coesl[1], n1_root_of_unity_table[0], q_thread);
    CooleyTukeyUnit_(coesl[2],coesl[3], n1_root_of_unity_table[1], q_thread);
    CooleyTukeyUnit_(coesl[4],coesl[5], n1_root_of_unity_table[2], q_thread);
    CooleyTukeyUnit_(coesl[6],coesl[7], n1_root_of_unity_table[3], q_thread);

#pragma unroll
    for(i = 0;i<8;i++){
        coes[threadIdx.y][threadIdx.x + i * 32] = coesl[i]; //无存储体冲突
    }

#pragma unroll
    for(int i=0;i<8;i++){
        coesl[i] = coes[threadIdx.y][(threadIdx.x >> 2) * 32 + 4 * i + (threadIdx.x & 3)];
    }
    
    //level 4
    CooleyTukeyUnit_(coesl[0],coesl[4], n1_root_of_unity_table[threadIdx.x >> 2], q_thread);
    CooleyTukeyUnit_(coesl[1],coesl[5], n1_root_of_unity_table[threadIdx.x >> 2], q_thread);
    CooleyTukeyUnit_(coesl[2],coesl[6], n1_root_of_unity_table[threadIdx.x >> 2], q_thread);
    CooleyTukeyUnit_(coesl[3],coesl[7], n1_root_of_unity_table[threadIdx.x >> 2], q_thread);

    //level 5
    CooleyTukeyUnit_(coesl[0],coesl[2], n1_root_of_unity_table[(threadIdx.x >> 2) * 2 ], q_thread);
    CooleyTukeyUnit_(coesl[1],coesl[3], n1_root_of_unity_table[(threadIdx.x >> 2) * 2 ], q_thread);
    CooleyTukeyUnit_(coesl[4],coesl[6], n1_root_of_unity_table[(threadIdx.x >> 2) * 2 + 1], q_thread);
    CooleyTukeyUnit_(coesl[5],coesl[7], n1_root_of_unity_table[(threadIdx.x >> 2) * 2 + 1], q_thread);

    //level 6
    CooleyTukeyUnit_(coesl[0],coesl[1], n1_root_of_unity_table[(threadIdx.x >> 2) * 4], q_thread);
    CooleyTukeyUnit_(coesl[2],coesl[3], n1_root_of_unity_table[(threadIdx.x >> 2) * 4 + 1], q_thread);
    CooleyTukeyUnit_(coesl[4],coesl[5], n1_root_of_unity_table[(threadIdx.x >> 2) * 4 + 2], q_thread);
    CooleyTukeyUnit_(coesl[6],coesl[7], n1_root_of_unity_table[(threadIdx.x >> 2) * 4 + 3], q_thread);

#pragma unroll
    for(int i=0;i<8;i++){
        coes[threadIdx.y][(threadIdx.x >> 2) * 32 + 4 * i + (threadIdx.x & 3)] = coesl[i];
    }

#pragma unroll
    for(int i=0;i<8;i++){
        coesl[i] = coes[threadIdx.y][threadIdx.x *8 + i];
    }

    //level 7
    CooleyTukeyUnit_(coesl[0],coesl[2], n1_root_of_unity_table[threadIdx.x * 2], q_thread);
    CooleyTukeyUnit_(coesl[1],coesl[3], n1_root_of_unity_table[threadIdx.x * 2], q_thread);
    CooleyTukeyUnit_(coesl[4],coesl[6], n1_root_of_unity_table[threadIdx.x * 2 + 1], q_thread);
    CooleyTukeyUnit_(coesl[5],coesl[7], n1_root_of_unity_table[threadIdx.x * 2 + 1], q_thread);

    //level 8
    CooleyTukeyUnit_(coesl[0],coesl[1], n1_root_of_unity_table[threadIdx.x * 4], q_thread);
    CooleyTukeyUnit_(coesl[2],coesl[3], n1_root_of_unity_table[threadIdx.x * 4 + 1], q_thread);
    CooleyTukeyUnit_(coesl[4],coesl[5], n1_root_of_unity_table[threadIdx.x * 4 + 2], q_thread);
    CooleyTukeyUnit_(coesl[6],coesl[7], n1_root_of_unity_table[threadIdx.x * 4 + 3], q_thread);

#pragma unroll
    for(int i=0;i<8;i++){
        polynomial_in[global_start + (threadIdx.y << 8) + threadIdx.x * 8 + i] = coesl[i];
    }
    
}


__global__ void FourStepFowardCore256(Data* polynomial_in,Root * n1_root_of_unity_table, Root * W_root_of_unity_table ,int mod_count ,int n_power, Modulus* modulus){
    
    __shared__ Data coes[4][256];

    Data coesl[8]; 
    int i;

    int q_index = blockIdx.y % mod_count; //moduls共用一个
    Modulus q_thread = modulus[q_index];//q_thread代表的是模数
    int global_start = blockIdx.x * 256 * blockDim.y + (blockIdx.y << n_power );

    coesl[0] = VALUE_GPU::mult(polynomial_in[global_start + (threadIdx.y << 8) + threadIdx.x],W_root_of_unity_table[blockIdx.x * 256 * blockDim.y + (threadIdx.y << 8) + threadIdx.x], q_thread);
    coesl[1] = VALUE_GPU::mult(polynomial_in[global_start + (threadIdx.y << 8) + threadIdx.x + 32],W_root_of_unity_table[blockIdx.x * 256 * blockDim.y + (threadIdx.y << 8) + threadIdx.x + 32], q_thread);
    coesl[2] = VALUE_GPU::mult(polynomial_in[global_start + (threadIdx.y << 8) + threadIdx.x + 64],W_root_of_unity_table[blockIdx.x * 256 * blockDim.y + (threadIdx.y << 8) + threadIdx.x + 64], q_thread);
    coesl[3] = VALUE_GPU::mult(polynomial_in[global_start + (threadIdx.y << 8) + threadIdx.x + 96],W_root_of_unity_table[blockIdx.x * 256 * blockDim.y + (threadIdx.y << 8) + threadIdx.x + 96], q_thread);
    coesl[4] = VALUE_GPU::mult(polynomial_in[global_start + (threadIdx.y << 8) + threadIdx.x + 128],W_root_of_unity_table[blockIdx.x * 256 * blockDim.y + (threadIdx.y << 8) + threadIdx.x + 128], q_thread);
    coesl[5] = VALUE_GPU::mult(polynomial_in[global_start + (threadIdx.y << 8) + threadIdx.x + 160],W_root_of_unity_table[blockIdx.x * 256 * blockDim.y + (threadIdx.y << 8) + threadIdx.x + 160], q_thread);
    coesl[6] = VALUE_GPU::mult(polynomial_in[global_start + (threadIdx.y << 8) + threadIdx.x + 192],W_root_of_unity_table[blockIdx.x * 256 * blockDim.y + (threadIdx.y << 8) + threadIdx.x + 192], q_thread);
    coesl[7] = VALUE_GPU::mult(polynomial_in[global_start + (threadIdx.y << 8) + threadIdx.x + 224],W_root_of_unity_table[blockIdx.x * 256 * blockDim.y + (threadIdx.y << 8) + threadIdx.x + 224], q_thread);

    //level 1
    CooleyTukeyUnit_(coesl[0],coesl[4], n1_root_of_unity_table[0], q_thread);
    CooleyTukeyUnit_(coesl[1],coesl[5], n1_root_of_unity_table[0], q_thread);
    CooleyTukeyUnit_(coesl[2],coesl[6], n1_root_of_unity_table[0], q_thread);
    CooleyTukeyUnit_(coesl[3],coesl[7], n1_root_of_unity_table[0], q_thread);

    //level 2
    CooleyTukeyUnit_(coesl[0],coesl[2], n1_root_of_unity_table[0], q_thread);
    CooleyTukeyUnit_(coesl[1],coesl[3], n1_root_of_unity_table[0], q_thread);
    CooleyTukeyUnit_(coesl[4],coesl[6], n1_root_of_unity_table[1], q_thread);
    CooleyTukeyUnit_(coesl[5],coesl[7], n1_root_of_unity_table[1], q_thread);

    //level 3
    CooleyTukeyUnit_(coesl[0],coesl[1], n1_root_of_unity_table[0], q_thread);
    CooleyTukeyUnit_(coesl[2],coesl[3], n1_root_of_unity_table[1], q_thread);
    CooleyTukeyUnit_(coesl[4],coesl[5], n1_root_of_unity_table[2], q_thread);
    CooleyTukeyUnit_(coesl[6],coesl[7], n1_root_of_unity_table[3], q_thread);

#pragma unroll
    for(i = 0;i<8;i++){
        coes[threadIdx.y][threadIdx.x + i * 32] = coesl[i]; //无存储体冲突
    }

#pragma unroll
    for(int i=0;i<8;i++){
        coesl[i] = coes[threadIdx.y][(threadIdx.x >> 2) * 32 + 4 * i + (threadIdx.x & 3)];
    }
    
    //level 4
    CooleyTukeyUnit_(coesl[0],coesl[4], n1_root_of_unity_table[threadIdx.x >> 2], q_thread);
    CooleyTukeyUnit_(coesl[1],coesl[5], n1_root_of_unity_table[threadIdx.x >> 2], q_thread);
    CooleyTukeyUnit_(coesl[2],coesl[6], n1_root_of_unity_table[threadIdx.x >> 2], q_thread);
    CooleyTukeyUnit_(coesl[3],coesl[7], n1_root_of_unity_table[threadIdx.x >> 2], q_thread);

    //level 5
    CooleyTukeyUnit_(coesl[0],coesl[2], n1_root_of_unity_table[(threadIdx.x >> 2) * 2 ], q_thread);
    CooleyTukeyUnit_(coesl[1],coesl[3], n1_root_of_unity_table[(threadIdx.x >> 2) * 2 ], q_thread);
    CooleyTukeyUnit_(coesl[4],coesl[6], n1_root_of_unity_table[(threadIdx.x >> 2) * 2 + 1], q_thread);
    CooleyTukeyUnit_(coesl[5],coesl[7], n1_root_of_unity_table[(threadIdx.x >> 2) * 2 + 1], q_thread);

    //level 6
    CooleyTukeyUnit_(coesl[0],coesl[1], n1_root_of_unity_table[(threadIdx.x >> 2) * 4], q_thread);
    CooleyTukeyUnit_(coesl[2],coesl[3], n1_root_of_unity_table[(threadIdx.x >> 2) * 4 + 1], q_thread);
    CooleyTukeyUnit_(coesl[4],coesl[5], n1_root_of_unity_table[(threadIdx.x >> 2) * 4 + 2], q_thread);
    CooleyTukeyUnit_(coesl[6],coesl[7], n1_root_of_unity_table[(threadIdx.x >> 2) * 4 + 3], q_thread);

#pragma unroll
    for(int i=0;i<8;i++){
        coes[threadIdx.y][(threadIdx.x >> 2) * 32 + 4 * i + (threadIdx.x & 3)] = coesl[i];
    }

#pragma unroll
    for(int i=0;i<8;i++){
        coesl[i] = coes[threadIdx.y][threadIdx.x *8 + i];
    }

    //level 7
    CooleyTukeyUnit_(coesl[0],coesl[2], n1_root_of_unity_table[threadIdx.x * 2], q_thread);
    CooleyTukeyUnit_(coesl[1],coesl[3], n1_root_of_unity_table[threadIdx.x * 2], q_thread);
    CooleyTukeyUnit_(coesl[4],coesl[6], n1_root_of_unity_table[threadIdx.x * 2 + 1], q_thread);
    CooleyTukeyUnit_(coesl[5],coesl[7], n1_root_of_unity_table[threadIdx.x * 2 + 1], q_thread);

    //level 8
    CooleyTukeyUnit_(coesl[0],coesl[1], n1_root_of_unity_table[threadIdx.x * 4], q_thread);
    CooleyTukeyUnit_(coesl[2],coesl[3], n1_root_of_unity_table[threadIdx.x * 4 + 1], q_thread);
    CooleyTukeyUnit_(coesl[4],coesl[5], n1_root_of_unity_table[threadIdx.x * 4 + 2], q_thread);
    CooleyTukeyUnit_(coesl[6],coesl[7], n1_root_of_unity_table[threadIdx.x * 4 + 3], q_thread);

#pragma unroll
    for(int i=0;i<8;i++){
        polynomial_in[global_start + (threadIdx.y << 8) + threadIdx.x * 8 + i] = coesl[i];
    }
    
}

/*2024-8-18:
带转置版本
*/
__global__ void FourStepFowardCore256(Data* polynomial_in,Root * n1_root_of_unity_table, int mod_count ,int n_power, Modulus* modulus,Data* polynomial_out){
    
    __shared__ Data coes[4][256];

    Data coesl[8]; 
    int i;

    int q_index = blockIdx.y % mod_count; //moduls共用一个
    Modulus q_thread = modulus[q_index];//q_thread代表的是模数
    int global_start = blockIdx.x * 256 * blockDim.y + (blockIdx.y << n_power );

    coesl[0] = polynomial_in[global_start + (threadIdx.y << 8) + threadIdx.x];
    coesl[1] = polynomial_in[global_start + (threadIdx.y << 8) + threadIdx.x + 32];
    coesl[2] = polynomial_in[global_start + (threadIdx.y << 8) + threadIdx.x + 64];
    coesl[3] = polynomial_in[global_start + (threadIdx.y << 8) + threadIdx.x + 96];
    coesl[4] = polynomial_in[global_start + (threadIdx.y << 8) + threadIdx.x + 128];
    coesl[5] = polynomial_in[global_start + (threadIdx.y << 8) + threadIdx.x + 160];
    coesl[6] = polynomial_in[global_start + (threadIdx.y << 8) + threadIdx.x + 192];
    coesl[7] = polynomial_in[global_start + (threadIdx.y << 8) + threadIdx.x + 224];

    //level 1
    CooleyTukeyUnit_(coesl[0],coesl[4], n1_root_of_unity_table[0], q_thread);
    CooleyTukeyUnit_(coesl[1],coesl[5], n1_root_of_unity_table[0], q_thread);
    CooleyTukeyUnit_(coesl[2],coesl[6], n1_root_of_unity_table[0], q_thread);
    CooleyTukeyUnit_(coesl[3],coesl[7], n1_root_of_unity_table[0], q_thread);

    //level 2
    CooleyTukeyUnit_(coesl[0],coesl[2], n1_root_of_unity_table[0], q_thread);
    CooleyTukeyUnit_(coesl[1],coesl[3], n1_root_of_unity_table[0], q_thread);
    CooleyTukeyUnit_(coesl[4],coesl[6], n1_root_of_unity_table[1], q_thread);
    CooleyTukeyUnit_(coesl[5],coesl[7], n1_root_of_unity_table[1], q_thread);

    //level 3
    CooleyTukeyUnit_(coesl[0],coesl[1], n1_root_of_unity_table[0], q_thread);
    CooleyTukeyUnit_(coesl[2],coesl[3], n1_root_of_unity_table[1], q_thread);
    CooleyTukeyUnit_(coesl[4],coesl[5], n1_root_of_unity_table[2], q_thread);
    CooleyTukeyUnit_(coesl[6],coesl[7], n1_root_of_unity_table[3], q_thread);

#pragma unroll
    for(i = 0;i<8;i++){
        coes[threadIdx.y][threadIdx.x + i * 32] = coesl[i]; //无存储体冲突
    }

#pragma unroll
    for(int i=0;i<8;i++){
        coesl[i] = coes[threadIdx.y][(threadIdx.x >> 2) * 32 + 4 * i + (threadIdx.x & 3)];
    }
    
    //level 4
    CooleyTukeyUnit_(coesl[0],coesl[4], n1_root_of_unity_table[threadIdx.x >> 2], q_thread);
    CooleyTukeyUnit_(coesl[1],coesl[5], n1_root_of_unity_table[threadIdx.x >> 2], q_thread);
    CooleyTukeyUnit_(coesl[2],coesl[6], n1_root_of_unity_table[threadIdx.x >> 2], q_thread);
    CooleyTukeyUnit_(coesl[3],coesl[7], n1_root_of_unity_table[threadIdx.x >> 2], q_thread);

    //level 5
    CooleyTukeyUnit_(coesl[0],coesl[2], n1_root_of_unity_table[(threadIdx.x >> 2) * 2 ], q_thread);
    CooleyTukeyUnit_(coesl[1],coesl[3], n1_root_of_unity_table[(threadIdx.x >> 2) * 2 ], q_thread);
    CooleyTukeyUnit_(coesl[4],coesl[6], n1_root_of_unity_table[(threadIdx.x >> 2) * 2 + 1], q_thread);
    CooleyTukeyUnit_(coesl[5],coesl[7], n1_root_of_unity_table[(threadIdx.x >> 2) * 2 + 1], q_thread);

    //level 6
    CooleyTukeyUnit_(coesl[0],coesl[1], n1_root_of_unity_table[(threadIdx.x >> 2) * 4], q_thread);
    CooleyTukeyUnit_(coesl[2],coesl[3], n1_root_of_unity_table[(threadIdx.x >> 2) * 4 + 1], q_thread);
    CooleyTukeyUnit_(coesl[4],coesl[5], n1_root_of_unity_table[(threadIdx.x >> 2) * 4 + 2], q_thread);
    CooleyTukeyUnit_(coesl[6],coesl[7], n1_root_of_unity_table[(threadIdx.x >> 2) * 4 + 3], q_thread);

#pragma unroll
    for(int i=0;i<8;i++){
        coes[threadIdx.y][(threadIdx.x >> 2) * 32 + 4 * i + (threadIdx.x & 3)] = coesl[i];
    }

#pragma unroll
    for(int i=0;i<8;i++){
        coesl[i] = coes[threadIdx.y][threadIdx.x *8 + i];
    }

    //level 7
    CooleyTukeyUnit_(coesl[0],coesl[2], n1_root_of_unity_table[threadIdx.x * 2], q_thread);
    CooleyTukeyUnit_(coesl[1],coesl[3], n1_root_of_unity_table[threadIdx.x * 2], q_thread);
    CooleyTukeyUnit_(coesl[4],coesl[6], n1_root_of_unity_table[threadIdx.x * 2 + 1], q_thread);
    CooleyTukeyUnit_(coesl[5],coesl[7], n1_root_of_unity_table[threadIdx.x * 2 + 1], q_thread);

    //level 8
    CooleyTukeyUnit_(coesl[0],coesl[1], n1_root_of_unity_table[threadIdx.x * 4], q_thread);
    CooleyTukeyUnit_(coesl[2],coesl[3], n1_root_of_unity_table[threadIdx.x * 4 + 1], q_thread);
    CooleyTukeyUnit_(coesl[4],coesl[5], n1_root_of_unity_table[threadIdx.x * 4 + 2], q_thread);
    CooleyTukeyUnit_(coesl[6],coesl[7], n1_root_of_unity_table[threadIdx.x * 4 + 3], q_thread);

#pragma unroll
    //coesl[i]位于256矩阵的(blockIdx.x * blockDim.y + threadIdx.y, threadIdx.x * 8 + i)
    for(int i=0;i<8;i++){
        polynomial_out[(blockIdx.y << n_power ) + ( threadIdx.x * 8 + i) * 256 + blockIdx.x * blockDim.y + threadIdx.y] = coesl[i];
    }
    
}


/*2024-8-19:
线程组织结构:<<<dim3(8, 32, batch_size), 256>>>

示例:
32个4096维NTT的后9层
传递参数；12, 17, mod_count
*/
__global__ void FourStepPartialForwardCore2(Data* polynomial_in, Root* n2_root_of_unity_table, Modulus* modulus, int small_npower, int n_power,
                                            int mod_count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x; //2048个线程中的哪一个
    int block_y = blockIdx.y;//32个4096维的NTT
    int block_z = blockIdx.z;

    int local_idx = threadIdx.x;

    __shared__ Data sharedmemorys[512];//把4096个系数分为8份

    int n_power__ = small_npower;
    int t_2 = 8; //TODO:为什么这里的8是一个确定的数字呢？
    int t = 1 << t_2; //t代表的是做CT蝴蝶操作的间隔

    int dividx = (block_y << n_power__);//第几个4096
    int divindex = block_z << n_power;//第几个batch

    int q_index = block_z % mod_count;
    Modulus q_thread = modulus[q_index];

    int address = dividx + ((idx >> t_2) << t_2) + idx; //对于blockIdx.x中的每一个线程,address相当于是blockIdx.x * 512

    int shrd_dixidx_t = (local_idx >> t_2) << t_2;//共享内存中每一组开始的起点
    int shrd_address = shrd_dixidx_t + local_idx; //每一个线程所指向的位置

    sharedmemorys[shrd_address] = polynomial_in[address + divindex];
    sharedmemorys[shrd_address + t] = polynomial_in[address + t + divindex]; //t:注意刚开始的间隔是256
    //以上:读的时候是将4096个系数分为8组,每组512个系数
#pragma unroll
    for(int loop_dep = 0; loop_dep < 3; loop_dep++)
    {
        CooleyTukeyUnit_(sharedmemorys[shrd_address], sharedmemorys[shrd_address + t],
                        n2_root_of_unity_table[(idx >> t_2)], q_thread);

        t = t >> 1;
        t_2 -= 1;

        shrd_dixidx_t = (local_idx >> t_2) << t_2;
        shrd_address = shrd_dixidx_t + local_idx;
        __syncthreads();
    }

#pragma unroll
    for(int loop_indep = 0; loop_indep < 5; loop_indep++)
    {
        CooleyTukeyUnit_(sharedmemorys[shrd_address], sharedmemorys[shrd_address + t],
                        n2_root_of_unity_table[(idx >> t_2)], q_thread);

        t = t >> 1;
        t_2 -= 1;

        shrd_dixidx_t = (local_idx >> t_2) << t_2;
        shrd_address = shrd_dixidx_t + local_idx;
    }

    address = dividx + ((idx >> t_2) << t_2) + idx;

    CooleyTukeyUnit_(sharedmemorys[shrd_address], sharedmemorys[shrd_address + t],
                    n2_root_of_unity_table[(idx >> t_2)], q_thread);
    polynomial_in[address + divindex] = sharedmemorys[shrd_address];
    polynomial_in[address + t + divindex] = sharedmemorys[shrd_address + t];
}

__global__ void FourStepPartialForwardCore2(Data* polynomial_in, Root* n2_root_of_unity_table,
                                            Modulus modulus, int small_npower, int n_power)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int block_y = blockIdx.y;
    int block_z = blockIdx.z;

    int local_idx = threadIdx.x;

    __shared__ Data sharedmemorys[512];

    int n_power__ = small_npower;
    int t_2 = 8;
    int t = 1 << t_2;

    int dividx = (block_y << n_power__);
    int divindex = block_z << n_power;

    Modulus q_thread = modulus;

    int address = dividx + ((idx >> t_2) << t_2) + idx;

    int shrd_dixidx_t = (local_idx >> t_2) << t_2;
    int shrd_address = shrd_dixidx_t + local_idx;

    sharedmemorys[shrd_address] = polynomial_in[address + divindex];
    sharedmemorys[shrd_address + t] = polynomial_in[address + t + divindex];

#pragma unroll
    for(int loop_dep = 0; loop_dep < 3; loop_dep++)
    {
        CooleyTukeyUnit_(sharedmemorys[shrd_address], sharedmemorys[shrd_address + t],
                        n2_root_of_unity_table[(idx >> t_2)], q_thread);

        t = t >> 1;
        t_2 -= 1;

        shrd_dixidx_t = (local_idx >> t_2) << t_2;
        shrd_address = shrd_dixidx_t + local_idx;
        __syncthreads();
    }

#pragma unroll
    for(int loop_indep = 0; loop_indep < 5; loop_indep++)
    {
        CooleyTukeyUnit_(sharedmemorys[shrd_address], sharedmemorys[shrd_address + t],
                        n2_root_of_unity_table[(idx >> t_2)], q_thread);

        t = t >> 1;
        t_2 -= 1;

        shrd_dixidx_t = (local_idx >> t_2) << t_2;
        shrd_address = shrd_dixidx_t + local_idx;
    }

    address = dividx + ((idx >> t_2) << t_2) + idx;

    CooleyTukeyUnit_(sharedmemorys[shrd_address], sharedmemorys[shrd_address + t],
                    n2_root_of_unity_table[(idx >> t_2)], q_thread);
    polynomial_in[address + divindex] = sharedmemorys[shrd_address];
    polynomial_in[address + t + divindex] = sharedmemorys[shrd_address + t];
}


/*2024-8-8:
32个128维的NTT
线程组织形式:dim3(32, batch_size), 64

数量统计:每一个batch,对应了2048个线程，每个线程完成1个修正（乘上W_root）

处理流程:需要完成32个64维的NTT，因此，每一个block对应一个64维的NTT，最终，输出n1*n2的比特翻转的NTT结果*/

//7, 6, 1, 12,
__global__ void FourStepPartialForwardCore(Data* polynomial_in, Root* n2_root_of_unity_table,
                                           Root* w_root_of_unity_table, Modulus* modulus,
                                           int small_npower, int T, int LOOP, int n_power,
                                           int mod_count)
{
    int local_idx = threadIdx.x;
    int block_x = blockIdx.x;
    int block_y = blockIdx.y;

    __shared__ Data sharedmemorys[512];//这里为什么不用128，不直接完成了就好

    int n_power__ = small_npower;
    int t_2 = T; //t_2的首次赋值为6
    int t = 1 << t_2; //t代表的是间隔

    int dividx = (block_x << n_power__); //block_x * (2^7)
    int divindex = block_y << n_power;

    //固定唯一模数
    int q_index = block_y % mod_count;
    Modulus q_thread = modulus[q_index];

    int address = dividx + ((local_idx >> t_2) << t_2) + local_idx; //对应该线程在4096个数中处理的位置

    int shrd_dixidx_t = (local_idx >> t_2) << t_2;
    int shrd_address = shrd_dixidx_t + local_idx; //对应线程在128个数中处理的位置

    Data mult_1 = polynomial_in[address + divindex];
    Data mult_2 = polynomial_in[address + t + divindex];

    mult_1 = VALUE_GPU::mult(mult_1, w_root_of_unity_table[address], q_thread);
    mult_2 = VALUE_GPU::mult(mult_2, w_root_of_unity_table[address + t], q_thread);

    sharedmemorys[shrd_address] = mult_1;
    sharedmemorys[shrd_address + t] = mult_2;

#pragma unroll
    for(int loop_dep = 0; loop_dep < LOOP; loop_dep++) //取值为1
    {
        CooleyTukeyUnit_(sharedmemorys[shrd_address], sharedmemorys[shrd_address + t],
                        n2_root_of_unity_table[(local_idx >> t_2)], q_thread);

        t = t >> 1;
        t_2 -= 1;

        shrd_dixidx_t = (local_idx >> t_2) << t_2;
        shrd_address = shrd_dixidx_t + local_idx; //二分，重新划分处理的位置
        __syncthreads();
    }

#pragma unroll
    //下面，每一组的长度已经是64，32个线程可以完全控制，因此不需要内存栅栏
    for(int loop_indep = 0; loop_indep < 5; loop_indep++)
    {
        CooleyTukeyUnit_(sharedmemorys[shrd_address], sharedmemorys[shrd_address + t],
                        n2_root_of_unity_table[(local_idx >> t_2)], q_thread);

        t = t >> 1;
        t_2 -= 1;
        shrd_dixidx_t = (local_idx >> t_2) << t_2;
        shrd_address = shrd_dixidx_t + local_idx;
    }

    address = dividx + ((local_idx >> t_2) << t_2) + local_idx; //等于shrd_address + dividx

    CooleyTukeyUnit_(sharedmemorys[shrd_address], sharedmemorys[shrd_address + t],
                    n2_root_of_unity_table[(local_idx >> t_2)], q_thread);
    polynomial_in[address + divindex] = sharedmemorys[shrd_address];
    polynomial_in[address + t + divindex] = sharedmemorys[shrd_address + t];
}

__global__ void FourStepPartialForwardCore(Data* polynomial_in, Root* n2_root_of_unity_table,
                                           Root* w_root_of_unity_table, Modulus modulus,
                                           int small_npower, int T, int LOOP, int n_power)
{
    int local_idx = threadIdx.x;
    int block_x = blockIdx.x;
    int block_y = blockIdx.y;

    __shared__ Data sharedmemorys[512];

    int n_power__ = small_npower;
    int t_2 = T;
    int t = 1 << t_2;

    int dividx = (block_x << n_power__);
    int divindex = block_y << n_power;

    Modulus q_thread = modulus;

    int address = dividx + ((local_idx >> t_2) << t_2) + local_idx;

    int shrd_dixidx_t = (local_idx >> t_2) << t_2;
    int shrd_address = shrd_dixidx_t + local_idx;

    Data mult_1 = polynomial_in[address + divindex];
    Data mult_2 = polynomial_in[address + t + divindex];

    mult_1 = VALUE_GPU::mult(mult_1, w_root_of_unity_table[address], q_thread);
    mult_2 = VALUE_GPU::mult(mult_2, w_root_of_unity_table[address + t], q_thread);

    sharedmemorys[shrd_address] = mult_1;
    sharedmemorys[shrd_address + t] = mult_2;

#pragma unroll
    for(int loop_dep = 0; loop_dep < LOOP; loop_dep++)
    {
        CooleyTukeyUnit_(sharedmemorys[shrd_address], sharedmemorys[shrd_address + t],
                        n2_root_of_unity_table[(local_idx >> t_2)], q_thread);

        t = t >> 1;
        t_2 -= 1;

        shrd_dixidx_t = (local_idx >> t_2) << t_2;
        shrd_address = shrd_dixidx_t + local_idx;
        __syncthreads();
    }

#pragma unroll
    for(int loop_indep = 0; loop_indep < 5; loop_indep++)
    {
        CooleyTukeyUnit_(sharedmemorys[shrd_address], sharedmemorys[shrd_address + t],
                        n2_root_of_unity_table[(local_idx >> t_2)], q_thread);

        t = t >> 1;
        t_2 -= 1;
        shrd_dixidx_t = (local_idx >> t_2) << t_2;
        shrd_address = shrd_dixidx_t + local_idx;
    }

    address = dividx + ((local_idx >> t_2) << t_2) + local_idx;

    CooleyTukeyUnit_(sharedmemorys[shrd_address], sharedmemorys[shrd_address + t],
                    n2_root_of_unity_table[(local_idx >> t_2)], q_thread);
    polynomial_in[address + divindex] = sharedmemorys[shrd_address];
    polynomial_in[address + t + divindex] = sharedmemorys[shrd_address + t];
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// INTT PART

/*2024-8-8:*/
__global__ void FourStepInverseCoreT1(Data* polynomial_in, Data* polynomial_out,
                                      Root* n1_root_of_unity_table, Modulus* modulus, int index1,
                                      int index2, int n_power, int mod_count)
{
    int idx_x = threadIdx.x;
    int idx_y = threadIdx.y;
    int block_x = blockIdx.x;
    int block_y = blockIdx.y;

    __shared__ Data sharedmemorys[32][32 + 1];

    int divindex = block_y << n_power;

    int q_index = block_y % mod_count;
    Modulus q_thread = modulus[q_index];

    int idx_index = idx_x + (idx_y << 5);
    int global_addresss = idx_index + (block_x << 10);

    // Load data from global & store to shared
    sharedmemorys[idx_y][idx_x] = polynomial_in[global_addresss + divindex];
    sharedmemorys[idx_y + 8][idx_x] = polynomial_in[global_addresss + 256 + divindex];
    sharedmemorys[idx_y + 16][idx_x] = polynomial_in[global_addresss + 512 + divindex];
    sharedmemorys[idx_y + 24][idx_x] = polynomial_in[global_addresss + 768 + divindex];
    __syncthreads();

    int global_index1 = idx_x >> 4;
    int global_index2 = idx_x % 16;

    int t_ = 0;
    int t = 1 << t_;
    int in_shared_address = ((global_index2 >> t_) << t_) + global_index2;

    GentlemanSandeUnit_(sharedmemorys[(idx_y << 1) + global_index1][in_shared_address],
                       sharedmemorys[(idx_y << 1) + global_index1][in_shared_address + t],
                       n1_root_of_unity_table[(global_index2 >> t_)], q_thread);
    GentlemanSandeUnit_(sharedmemorys[(idx_y << 1) + global_index1 + 16][in_shared_address],
                       sharedmemorys[(idx_y << 1) + global_index1 + 16][in_shared_address + t],
                       n1_root_of_unity_table[(global_index2 >> t_)], q_thread);
    __syncthreads();

    for(int i = 0; i < 4; i++)
    {
        t = t << 1;
        t_ += 1;

        in_shared_address = ((global_index2 >> t_) << t_) + global_index2;
        ;

        GentlemanSandeUnit_(sharedmemorys[(idx_y << 1) + global_index1][in_shared_address],
                           sharedmemorys[(idx_y << 1) + global_index1][in_shared_address + t],
                           n1_root_of_unity_table[(global_index2 >> t_)], q_thread);
        GentlemanSandeUnit_(sharedmemorys[(idx_y << 1) + global_index1 + 16][in_shared_address],
                           sharedmemorys[(idx_y << 1) + global_index1 + 16][in_shared_address + t],
                           n1_root_of_unity_table[(global_index2 >> t_)], q_thread);
        __syncthreads();
    }
    __syncthreads();

    polynomial_out[idx_x + (idx_y << index1) + (block_x << 5) + divindex] =
        sharedmemorys[idx_x][idx_y];
    polynomial_out[idx_x + (idx_y << index1) + (block_x << 5) + index2 + divindex] =
        sharedmemorys[idx_x][idx_y + 8];
    polynomial_out[idx_x + (idx_y << index1) + (block_x << 5) + (index2 * 2) + divindex] =
        sharedmemorys[idx_x][idx_y + 16];
    polynomial_out[idx_x + (idx_y << index1) + (block_x << 5) + (index2 * 3) + divindex] =
        sharedmemorys[idx_x][idx_y + 24];
}

__global__ void FourStepInverseCoreT1(Data* polynomial_in, Data* polynomial_out,
                                      Root* n1_root_of_unity_table, Modulus modulus, int index1,
                                      int index2, int n_power)
{
    int idx_x = threadIdx.x;
    int idx_y = threadIdx.y;
    int block_x = blockIdx.x;
    int block_y = blockIdx.y;

    __shared__ Data sharedmemorys[32][32 + 1];

    int divindex = block_y << n_power;

    Modulus q_thread = modulus;

    int idx_index = idx_x + (idx_y << 5);
    int global_addresss = idx_index + (block_x << 10);

    // Load data from global & store to shared
    sharedmemorys[idx_y][idx_x] = polynomial_in[global_addresss + divindex];
    sharedmemorys[idx_y + 8][idx_x] = polynomial_in[global_addresss + 256 + divindex];
    sharedmemorys[idx_y + 16][idx_x] = polynomial_in[global_addresss + 512 + divindex];
    sharedmemorys[idx_y + 24][idx_x] = polynomial_in[global_addresss + 768 + divindex];
    __syncthreads();

    int global_index1 = idx_x >> 4;
    int global_index2 = idx_x % 16;

    int t_ = 0;
    int t = 1 << t_;
    int in_shared_address = ((global_index2 >> t_) << t_) + global_index2;

    GentlemanSandeUnit_(sharedmemorys[(idx_y << 1) + global_index1][in_shared_address],
                       sharedmemorys[(idx_y << 1) + global_index1][in_shared_address + t],
                       n1_root_of_unity_table[(global_index2 >> t_)], q_thread);
    GentlemanSandeUnit_(sharedmemorys[(idx_y << 1) + global_index1 + 16][in_shared_address],
                       sharedmemorys[(idx_y << 1) + global_index1 + 16][in_shared_address + t],
                       n1_root_of_unity_table[(global_index2 >> t_)], q_thread);
    __syncthreads();

    for(int i = 0; i < 4; i++)
    {
        t = t << 1;
        t_ += 1;

        in_shared_address = ((global_index2 >> t_) << t_) + global_index2;
        ;

        GentlemanSandeUnit_(sharedmemorys[(idx_y << 1) + global_index1][in_shared_address],
                           sharedmemorys[(idx_y << 1) + global_index1][in_shared_address + t],
                           n1_root_of_unity_table[(global_index2 >> t_)], q_thread);
        GentlemanSandeUnit_(sharedmemorys[(idx_y << 1) + global_index1 + 16][in_shared_address],
                           sharedmemorys[(idx_y << 1) + global_index1 + 16][in_shared_address + t],
                           n1_root_of_unity_table[(global_index2 >> t_)], q_thread);
        __syncthreads();
    }
    __syncthreads();

    polynomial_out[idx_x + (idx_y << index1) + (block_x << 5) + divindex] =
        sharedmemorys[idx_x][idx_y];
    polynomial_out[idx_x + (idx_y << index1) + (block_x << 5) + index2 + divindex] =
        sharedmemorys[idx_x][idx_y + 8];
    polynomial_out[idx_x + (idx_y << index1) + (block_x << 5) + (index2 * 2) + divindex] =
        sharedmemorys[idx_x][idx_y + 16];
    polynomial_out[idx_x + (idx_y << index1) + (block_x << 5) + (index2 * 3) + divindex] =
        sharedmemorys[idx_x][idx_y + 24];
}

__global__ void FourStepInverseCoreT2(Data* polynomial_in, Data* polynomial_out,
                                      Root* n1_root_of_unity_table, Modulus* modulus, int index1,
                                      int index2, int index3, int n_power, int mod_count)
{
    int idx_x = threadIdx.x;
    int idx_y = threadIdx.y;
    int block_x = blockIdx.x;
    int block_y = blockIdx.y;

    __shared__ Data sharedmemorys[16][64 + 1];

    int divindex = block_y << n_power;

    int q_index = block_y % mod_count;
    Modulus q_thread = modulus[q_index];

    int idx_index = idx_x + (idx_y << 5);
    int global_addresss = idx_index + (block_x << 10);

    int shr_index1 = idx_y >> 1;
    int shr_index2 = idx_y % 2;

    // Load data from global & store to shared
    sharedmemorys[shr_index1][idx_x + (shr_index2 << 5)] =
        polynomial_in[global_addresss + divindex];
    sharedmemorys[shr_index1 + 4][idx_x + (shr_index2 << 5)] =
        polynomial_in[global_addresss + 256 + divindex];
    sharedmemorys[shr_index1 + 8][idx_x + (shr_index2 << 5)] =
        polynomial_in[global_addresss + 512 + divindex];
    sharedmemorys[shr_index1 + 12][idx_x + (shr_index2 << 5)] =
        polynomial_in[global_addresss + 768 + divindex];
    __syncthreads();

    int t_ = 0;
    int t = 1 << t_;
    int in_shared_address = ((idx_x >> t_) << t_) + idx_x;

    GentlemanSandeUnit_(sharedmemorys[idx_y][in_shared_address],
                       sharedmemorys[idx_y][in_shared_address + t],
                       n1_root_of_unity_table[(idx_x >> t_)], q_thread);
    GentlemanSandeUnit_(sharedmemorys[idx_y + 8][in_shared_address],
                       sharedmemorys[idx_y + 8][in_shared_address + t],
                       n1_root_of_unity_table[(idx_x >> t_)], q_thread);
    __syncthreads();

    for(int i = 0; i < 5; i++)
    {
        t = t << 1;
        t_ += 1;

        in_shared_address = ((idx_x >> t_) << t_) + idx_x;

        GentlemanSandeUnit_(sharedmemorys[idx_y][in_shared_address],
                           sharedmemorys[idx_y][in_shared_address + t],
                           n1_root_of_unity_table[(idx_x >> t_)], q_thread);
        GentlemanSandeUnit_(sharedmemorys[idx_y + 8][in_shared_address],
                           sharedmemorys[idx_y + 8][in_shared_address + t],
                           n1_root_of_unity_table[(idx_x >> t_)], q_thread);
        __syncthreads();
    }
    __syncthreads();

    int global_index1 = idx_x >> 4;
    int global_index2 = idx_x % 16;

    polynomial_out[global_index2 + (global_index1 << index1) + (idx_y << index2) + (block_x << 4) +
                   divindex] = sharedmemorys[global_index2][global_index1 + (idx_y << 1)];
    polynomial_out[global_index2 + (global_index1 << index1) + (idx_y << index2) + (block_x << 4) +
                   index3 + divindex] =
        sharedmemorys[global_index2][global_index1 + (idx_y << 1) + 16];
    polynomial_out[global_index2 + (global_index1 << index1) + (idx_y << index2) + (block_x << 4) +
                   (index3 * 2) + divindex] =
        sharedmemorys[global_index2][global_index1 + (idx_y << 1) + 32];
    polynomial_out[global_index2 + (global_index1 << index1) + (idx_y << index2) + (block_x << 4) +
                   (index3 * 3) + divindex] =
        sharedmemorys[global_index2][global_index1 + (idx_y << 1) + 48];
}

__global__ void FourStepInverseCoreT2(Data* polynomial_in, Data* polynomial_out,
                                      Root* n1_root_of_unity_table, Modulus modulus, int index1,
                                      int index2, int index3, int n_power)
{
    int idx_x = threadIdx.x;
    int idx_y = threadIdx.y;
    int block_x = blockIdx.x;
    int block_y = blockIdx.y;

    __shared__ Data sharedmemorys[16][64 + 1];

    int divindex = block_y << n_power;

    Modulus q_thread = modulus;

    int idx_index = idx_x + (idx_y << 5);
    int global_addresss = idx_index + (block_x << 10);

    int shr_index1 = idx_y >> 1;
    int shr_index2 = idx_y % 2;

    // Load data from global & store to shared
    sharedmemorys[shr_index1][idx_x + (shr_index2 << 5)] =
        polynomial_in[global_addresss + divindex];
    sharedmemorys[shr_index1 + 4][idx_x + (shr_index2 << 5)] =
        polynomial_in[global_addresss + 256 + divindex];
    sharedmemorys[shr_index1 + 8][idx_x + (shr_index2 << 5)] =
        polynomial_in[global_addresss + 512 + divindex];
    sharedmemorys[shr_index1 + 12][idx_x + (shr_index2 << 5)] =
        polynomial_in[global_addresss + 768 + divindex];
    __syncthreads();

    int t_ = 0;
    int t = 1 << t_;
    int in_shared_address = ((idx_x >> t_) << t_) + idx_x;

    GentlemanSandeUnit_(sharedmemorys[idx_y][in_shared_address],
                       sharedmemorys[idx_y][in_shared_address + t],
                       n1_root_of_unity_table[(idx_x >> t_)], q_thread);
    GentlemanSandeUnit_(sharedmemorys[idx_y + 8][in_shared_address],
                       sharedmemorys[idx_y + 8][in_shared_address + t],
                       n1_root_of_unity_table[(idx_x >> t_)], q_thread);
    __syncthreads();

    for(int i = 0; i < 5; i++)
    {
        t = t << 1;
        t_ += 1;

        in_shared_address = ((idx_x >> t_) << t_) + idx_x;

        GentlemanSandeUnit_(sharedmemorys[idx_y][in_shared_address],
                           sharedmemorys[idx_y][in_shared_address + t],
                           n1_root_of_unity_table[(idx_x >> t_)], q_thread);
        GentlemanSandeUnit_(sharedmemorys[idx_y + 8][in_shared_address],
                           sharedmemorys[idx_y + 8][in_shared_address + t],
                           n1_root_of_unity_table[(idx_x >> t_)], q_thread);
        __syncthreads();
    }
    __syncthreads();

    int global_index1 = idx_x >> 4;
    int global_index2 = idx_x % 16;

    polynomial_out[global_index2 + (global_index1 << index1) + (idx_y << index2) + (block_x << 4) +
                   divindex] = sharedmemorys[global_index2][global_index1 + (idx_y << 1)];
    polynomial_out[global_index2 + (global_index1 << index1) + (idx_y << index2) + (block_x << 4) +
                   index3 + divindex] =
        sharedmemorys[global_index2][global_index1 + (idx_y << 1) + 16];
    polynomial_out[global_index2 + (global_index1 << index1) + (idx_y << index2) + (block_x << 4) +
                   (index3 * 2) + divindex] =
        sharedmemorys[global_index2][global_index1 + (idx_y << 1) + 32];
    polynomial_out[global_index2 + (global_index1 << index1) + (idx_y << index2) + (block_x << 4) +
                   (index3 * 3) + divindex] =
        sharedmemorys[global_index2][global_index1 + (idx_y << 1) + 48];
}

__global__ void FourStepInverseCoreT3(Data* polynomial_in, Data* polynomial_out,
                                      Root* n1_root_of_unity_table, Modulus* modulus, int index1,
                                      int index2, int index3, int n_power, int mod_count)
{
    int idx_x = threadIdx.x;
    int idx_y = threadIdx.y;
    int block_x = blockIdx.x;
    int block_y = blockIdx.y;

    __shared__ Data sharedmemorys[8][128 + 1];

    int divindex = block_y << n_power;

    int q_index = block_y % mod_count;
    Modulus q_thread = modulus[q_index];

    int idx_index = idx_x + (idx_y << 5);
    int global_addresss = idx_index + (block_x << 10);

    int shr_index1 = idx_y >> 2;
    int shr_index2 = idx_y % 4;

    // Load data from global & store to shared
    sharedmemorys[shr_index1][idx_x + (shr_index2 << 5)] =
        polynomial_in[global_addresss + divindex];
    sharedmemorys[shr_index1 + 2][idx_x + (shr_index2 << 5)] =
        polynomial_in[global_addresss + 256 + divindex];
    sharedmemorys[shr_index1 + 4][idx_x + (shr_index2 << 5)] =
        polynomial_in[global_addresss + 512 + divindex];
    sharedmemorys[shr_index1 + 6][idx_x + (shr_index2 << 5)] =
        polynomial_in[global_addresss + 768 + divindex];
    __syncthreads();

    int shr_address = idx_x + ((idx_y % 2) << 5);
    int shr_in = idx_y >> 1;

    int t_ = 0;
    int t = 1 << t_;
    int in_shared_address = ((shr_address >> t_) << t_) + shr_address;

    GentlemanSandeUnit_(sharedmemorys[shr_in][in_shared_address],
                       sharedmemorys[shr_in][in_shared_address + t],
                       n1_root_of_unity_table[(shr_address >> t_)], q_thread);
    GentlemanSandeUnit_(sharedmemorys[shr_in + 4][in_shared_address],
                       sharedmemorys[shr_in + 4][in_shared_address + t],
                       n1_root_of_unity_table[(shr_address >> t_)], q_thread);
    __syncthreads();

    for(int i = 0; i < 6; i++)
    {
        t = t << 1;
        t_ += 1;

        in_shared_address = ((shr_address >> t_) << t_) + shr_address;

        GentlemanSandeUnit_(sharedmemorys[shr_in][in_shared_address],
                           sharedmemorys[shr_in][in_shared_address + t],
                           n1_root_of_unity_table[(shr_address >> t_)], q_thread);
        GentlemanSandeUnit_(sharedmemorys[shr_in + 4][in_shared_address],
                           sharedmemorys[shr_in + 4][in_shared_address + t],
                           n1_root_of_unity_table[(shr_address >> t_)], q_thread);
        __syncthreads();
    }
    __syncthreads();

    int global_index1 = idx_x >> 3;
    int global_index2 = idx_x % 8;

    polynomial_out[global_index2 + (global_index1 << index1) + (idx_y << index2) + (block_x << 3) +
                   divindex] = sharedmemorys[global_index2][global_index1 + (idx_y << 2)];
    polynomial_out[global_index2 + (global_index1 << index1) + (idx_y << index2) + (block_x << 3) +
                   index3 + divindex] =
        sharedmemorys[global_index2][global_index1 + (idx_y << 2) + 32];
    polynomial_out[global_index2 + (global_index1 << index1) + (idx_y << index2) + (block_x << 3) +
                   (index3 * 2) + divindex] =
        sharedmemorys[global_index2][global_index1 + (idx_y << 2) + 64];
    polynomial_out[global_index2 + (global_index1 << index1) + (idx_y << index2) + (block_x << 3) +
                   (index3 * 3) + divindex] =
        sharedmemorys[global_index2][global_index1 + (idx_y << 2) + 96];
}

__global__ void FourStepInverseCoreT3(Data* polynomial_in, Data* polynomial_out,
                                      Root* n1_root_of_unity_table, Modulus modulus, int index1,
                                      int index2, int index3, int n_power)
{
    int idx_x = threadIdx.x;
    int idx_y = threadIdx.y;
    int block_x = blockIdx.x;
    int block_y = blockIdx.y;

    __shared__ Data sharedmemorys[8][128 + 1];

    int divindex = block_y << n_power;

    Modulus q_thread = modulus;

    int idx_index = idx_x + (idx_y << 5);
    int global_addresss = idx_index + (block_x << 10);

    int shr_index1 = idx_y >> 2;
    int shr_index2 = idx_y % 4;

    // Load data from global & store to shared
    sharedmemorys[shr_index1][idx_x + (shr_index2 << 5)] =
        polynomial_in[global_addresss + divindex];
    sharedmemorys[shr_index1 + 2][idx_x + (shr_index2 << 5)] =
        polynomial_in[global_addresss + 256 + divindex];
    sharedmemorys[shr_index1 + 4][idx_x + (shr_index2 << 5)] =
        polynomial_in[global_addresss + 512 + divindex];
    sharedmemorys[shr_index1 + 6][idx_x + (shr_index2 << 5)] =
        polynomial_in[global_addresss + 768 + divindex];
    __syncthreads();

    int shr_address = idx_x + ((idx_y % 2) << 5);
    int shr_in = idx_y >> 1;

    int t_ = 0;
    int t = 1 << t_;
    int in_shared_address = ((shr_address >> t_) << t_) + shr_address;

    GentlemanSandeUnit_(sharedmemorys[shr_in][in_shared_address],
                       sharedmemorys[shr_in][in_shared_address + t],
                       n1_root_of_unity_table[(shr_address >> t_)], q_thread);
    GentlemanSandeUnit_(sharedmemorys[shr_in + 4][in_shared_address],
                       sharedmemorys[shr_in + 4][in_shared_address + t],
                       n1_root_of_unity_table[(shr_address >> t_)], q_thread);
    __syncthreads();

    for(int i = 0; i < 6; i++)
    {
        t = t << 1;
        t_ += 1;

        in_shared_address = ((shr_address >> t_) << t_) + shr_address;

        GentlemanSandeUnit_(sharedmemorys[shr_in][in_shared_address],
                           sharedmemorys[shr_in][in_shared_address + t],
                           n1_root_of_unity_table[(shr_address >> t_)], q_thread);
        GentlemanSandeUnit_(sharedmemorys[shr_in + 4][in_shared_address],
                           sharedmemorys[shr_in + 4][in_shared_address + t],
                           n1_root_of_unity_table[(shr_address >> t_)], q_thread);
        __syncthreads();
    }
    __syncthreads();

    int global_index1 = idx_x >> 3;
    int global_index2 = idx_x % 8;

    polynomial_out[global_index2 + (global_index1 << index1) + (idx_y << index2) + (block_x << 3) +
                   divindex] = sharedmemorys[global_index2][global_index1 + (idx_y << 2)];
    polynomial_out[global_index2 + (global_index1 << index1) + (idx_y << index2) + (block_x << 3) +
                   index3 + divindex] =
        sharedmemorys[global_index2][global_index1 + (idx_y << 2) + 32];
    polynomial_out[global_index2 + (global_index1 << index1) + (idx_y << index2) + (block_x << 3) +
                   (index3 * 2) + divindex] =
        sharedmemorys[global_index2][global_index1 + (idx_y << 2) + 64];
    polynomial_out[global_index2 + (global_index1 << index1) + (idx_y << index2) + (block_x << 3) +
                   (index3 * 3) + divindex] =
        sharedmemorys[global_index2][global_index1 + (idx_y << 2) + 96];
}

__global__ void FourStepInverseCoreT4(Data* polynomial_in, Data* polynomial_out,
                                      Root* n1_root_of_unity_table, Modulus* modulus, int index1,
                                      int index2, int index3, int n_power, int mod_count)
{
    int idx_x = threadIdx.x;
    int idx_y = threadIdx.y;
    int block_x = blockIdx.x;
    int block_y = blockIdx.y;

    __shared__ Data sharedmemorys[4][256 + 1];

    int divindex = block_y << n_power;

    int q_index = block_y % mod_count;
    Modulus q_thread = modulus[q_index];

    int idx_index = idx_x + (idx_y << 5);
    int global_addresss = idx_index + (block_x << 10);

    int shr_index1 = idx_y >> 3;
    int shr_index2 = idx_y % 8;

    // Load data from global & store to shared
    sharedmemorys[shr_index1][idx_x + (shr_index2 << 5)] =
        polynomial_in[global_addresss + divindex];
    sharedmemorys[shr_index1 + 1][idx_x + (shr_index2 << 5)] =
        polynomial_in[global_addresss + 256 + divindex];
    sharedmemorys[shr_index1 + 2][idx_x + (shr_index2 << 5)] =
        polynomial_in[global_addresss + 512 + divindex];
    sharedmemorys[shr_index1 + 3][idx_x + (shr_index2 << 5)] =
        polynomial_in[global_addresss + 768 + divindex];
    __syncthreads();

    int shr_address = idx_x + ((idx_y % 4) << 5);
    int shr_in = idx_y >> 2;

    int t_ = 0;
    int t = 1 << t_;
    int in_shared_address = ((shr_address >> t_) << t_) + shr_address;

    GentlemanSandeUnit_(sharedmemorys[shr_in][in_shared_address],
                       sharedmemorys[shr_in][in_shared_address + t],
                       n1_root_of_unity_table[(shr_address >> t_)], q_thread);
    GentlemanSandeUnit_(sharedmemorys[shr_in + 2][in_shared_address],
                       sharedmemorys[shr_in + 2][in_shared_address + t],
                       n1_root_of_unity_table[(shr_address >> t_)], q_thread);
    __syncthreads();

    for(int i = 0; i < 7; i++)
    {
        t = t << 1;
        t_ += 1;

        in_shared_address = ((shr_address >> t_) << t_) + shr_address;

        GentlemanSandeUnit_(sharedmemorys[shr_in][in_shared_address],
                           sharedmemorys[shr_in][in_shared_address + t],
                           n1_root_of_unity_table[(shr_address >> t_)], q_thread);
        GentlemanSandeUnit_(sharedmemorys[shr_in + 2][in_shared_address],
                           sharedmemorys[shr_in + 2][in_shared_address + t],
                           n1_root_of_unity_table[(shr_address >> t_)], q_thread);
        __syncthreads();
    }
    __syncthreads();

    int global_index1 = idx_x >> 2;
    int global_index2 = idx_x % 4;

    polynomial_out[global_index2 + (global_index1 << index1) + (idx_y << index2) + (block_x << 2) +
                   divindex] = sharedmemorys[global_index2][global_index1 + (idx_y << 3)];
    polynomial_out[global_index2 + (global_index1 << index1) + (idx_y << index2) + (block_x << 2) +
                   index3 + divindex] =
        sharedmemorys[global_index2][global_index1 + (idx_y << 3) + 64];
    polynomial_out[global_index2 + (global_index1 << index1) + (idx_y << index2) + (block_x << 2) +
                   (index3 * 2) + divindex] =
        sharedmemorys[global_index2][global_index1 + (idx_y << 3) + 128];
    polynomial_out[global_index2 + (global_index1 << index1) + (idx_y << index2) + (block_x << 2) +
                   (index3 * 3) + divindex] =
        sharedmemorys[global_index2][global_index1 + (idx_y << 3) + 192];
}

__global__ void FourStepInverseCoreT4(Data* polynomial_in, Data* polynomial_out,
                                      Root* n1_root_of_unity_table, Modulus modulus, int index1,
                                      int index2, int index3, int n_power)
{
    int idx_x = threadIdx.x;
    int idx_y = threadIdx.y;
    int block_x = blockIdx.x;
    int block_y = blockIdx.y;

    __shared__ Data sharedmemorys[4][256 + 1];

    int divindex = block_y << n_power;

    Modulus q_thread = modulus;

    int idx_index = idx_x + (idx_y << 5);
    int global_addresss = idx_index + (block_x << 10);

    int shr_index1 = idx_y >> 3;
    int shr_index2 = idx_y % 8;

    // Load data from global & store to shared
    sharedmemorys[shr_index1][idx_x + (shr_index2 << 5)] =
        polynomial_in[global_addresss + divindex];
    sharedmemorys[shr_index1 + 1][idx_x + (shr_index2 << 5)] =
        polynomial_in[global_addresss + 256 + divindex];
    sharedmemorys[shr_index1 + 2][idx_x + (shr_index2 << 5)] =
        polynomial_in[global_addresss + 512 + divindex];
    sharedmemorys[shr_index1 + 3][idx_x + (shr_index2 << 5)] =
        polynomial_in[global_addresss + 768 + divindex];
    __syncthreads();

    int shr_address = idx_x + ((idx_y % 4) << 5);
    int shr_in = idx_y >> 2;

    int t_ = 0;
    int t = 1 << t_;
    int in_shared_address = ((shr_address >> t_) << t_) + shr_address;

    GentlemanSandeUnit_(sharedmemorys[shr_in][in_shared_address],
                       sharedmemorys[shr_in][in_shared_address + t],
                       n1_root_of_unity_table[(shr_address >> t_)], q_thread);
    GentlemanSandeUnit_(sharedmemorys[shr_in + 2][in_shared_address],
                       sharedmemorys[shr_in + 2][in_shared_address + t],
                       n1_root_of_unity_table[(shr_address >> t_)], q_thread);
    __syncthreads();

    for(int i = 0; i < 7; i++)
    {
        t = t << 1;
        t_ += 1;

        in_shared_address = ((shr_address >> t_) << t_) + shr_address;

        GentlemanSandeUnit_(sharedmemorys[shr_in][in_shared_address],
                           sharedmemorys[shr_in][in_shared_address + t],
                           n1_root_of_unity_table[(shr_address >> t_)], q_thread);
        GentlemanSandeUnit_(sharedmemorys[shr_in + 2][in_shared_address],
                           sharedmemorys[shr_in + 2][in_shared_address + t],
                           n1_root_of_unity_table[(shr_address >> t_)], q_thread);
        __syncthreads();
    }
    __syncthreads();

    int global_index1 = idx_x >> 2;
    int global_index2 = idx_x % 4;

    polynomial_out[global_index2 + (global_index1 << index1) + (idx_y << index2) + (block_x << 2) +
                   divindex] = sharedmemorys[global_index2][global_index1 + (idx_y << 3)];
    polynomial_out[global_index2 + (global_index1 << index1) + (idx_y << index2) + (block_x << 2) +
                   index3 + divindex] =
        sharedmemorys[global_index2][global_index1 + (idx_y << 3) + 64];
    polynomial_out[global_index2 + (global_index1 << index1) + (idx_y << index2) + (block_x << 2) +
                   (index3 * 2) + divindex] =
        sharedmemorys[global_index2][global_index1 + (idx_y << 3) + 128];
    polynomial_out[global_index2 + (global_index1 << index1) + (idx_y << index2) + (block_x << 2) +
                   (index3 * 3) + divindex] =
        sharedmemorys[global_index2][global_index1 + (idx_y << 3) + 192];
}


/*2024-8-8:
参数: inverse:代表的是1/n mod q
     polynomial_in:n1*n2的排布
    polynomial_out:n2*n1的排布
线程组织形式:dim3(32, batch_size), 64

处理流程:（细节分析可参照FourStepPartialForwardCore）完成n1个n2维ntt，并乘上修正因子，最终完成转置操作。输入是n1*n2维的排布 ，输出是n2*n1维度的排布 [注意这里预先做了延迟处理]

7, 6, cfg.mod_inverse, 12, mod_count
*/
__global__ void FourStepPartialInverseCore(Data* polynomial_in, Data* polynomial_out,Root* n2_root_of_unity_table,
                                           Root* w_root_of_unity_table, Modulus* modulus,
                                           int small_npower, int LOOP, Ninverse* inverse,
                                           int poly_n_power, int mod_count)
{
    int local_idx = threadIdx.x;
    int block_x = blockIdx.x;
    int block_y = blockIdx.y;

    __shared__ Data sharedmemorys[512]; //TODO:感觉这里可以改为128

    int small_npower_ = small_npower;
    int t_2 = 0;
    int t = 1 << t_2;

    int dividx = (block_x << small_npower_);

    int divindex = block_y << poly_n_power;

    int q_index = block_y % mod_count;
    Modulus q_thread = modulus[q_index];

    int address = dividx + ((local_idx >> t_2) << t_2) + local_idx;

    int shrd_dixidx_t = (local_idx >> t_2) << t_2;
    int shrd_address = shrd_dixidx_t + local_idx;

    Data mult_1 = polynomial_in[address + divindex];
    Data mult_2 = polynomial_in[address + t + divindex];

    /*mult_1 = VALUE_GPU::mult(mult_1, w_root_of_unity_table[address], q_thread);
    mult_2 = VALUE_GPU::mult(mult_2, w_root_of_unity_table[address + t], q_thread); //这个需要放到后面*/

    sharedmemorys[shrd_address] = mult_1;
    sharedmemorys[shrd_address + t] = mult_2;

#pragma unroll
    for(int loop_dep = 0; loop_dep < LOOP; loop_dep++)
    {
        GentlemanSandeUnit_(sharedmemorys[shrd_address], sharedmemorys[shrd_address + t],
                           n2_root_of_unity_table[(local_idx >> t_2)], q_thread);

        t = t << 1;
        t_2 += 1;

        shrd_dixidx_t = (local_idx >> t_2) << t_2;
        shrd_address = shrd_dixidx_t + local_idx;
        __syncthreads();
    }

    address = dividx + ((local_idx >> t_2) << t_2) + local_idx;

    GentlemanSandeUnit_(sharedmemorys[shrd_address], sharedmemorys[shrd_address + t],
                       n2_root_of_unity_table[(local_idx >> t_2)], q_thread);
    __syncthreads();

    Data temp1 = VALUE_GPU::mult(sharedmemorys[shrd_address], inverse[0], q_thread); //除以n
    //polynomial_in[address + divindex] = temp1;
    //polynomial_in[address + divindex] = VALUE_GPU::mult(temp1 , w_root_of_unity_table[address + divindex], q_thread); //乘上修正因子

    polynomial_out[(address / (1 << small_npower)) + (address % (1 << small_npower)) * (1 << (poly_n_power - small_npower))+ divindex] = VALUE_GPU::mult(temp1 , w_root_of_unity_table[address + divindex], q_thread); //乘上修正因子

    Data temp2 = VALUE_GPU::mult(sharedmemorys[shrd_address + t], inverse[0], q_thread);
    //polynomial_in[address + t + divindex] = temp2;
    //polynomial_in[address + t + divindex] = VALUE_GPU::mult(temp2, w_root_of_unity_table[address + t + divindex], q_thread); //乘上修正因子
    polynomial_out[((address + t) / (1 << small_npower)) + ((address + t) % (1 << small_npower)) * (1 << (poly_n_power - small_npower)) + divindex] = VALUE_GPU::mult(temp2, w_root_of_unity_table[address + t + divindex], q_thread); //乘上修正因子
}




__global__ void FourStepPartialInverseCore(Data* polynomial_in, Root* n2_root_of_unity_table,
                                           Root* w_root_of_unity_table, Modulus modulus,
                                           int small_npower, int LOOP, Ninverse inverse,
                                           int poly_n_power)
{
    int local_idx = threadIdx.x;
    int block_x = blockIdx.x;
    int block_y = blockIdx.y;

    __shared__ Data sharedmemorys[512];

    int small_npower_ = small_npower;
    int t_2 = 0;
    int t = 1 << t_2;

    int dividx = (block_x << small_npower_);

    int divindex = block_y << poly_n_power;

    Modulus q_thread = modulus;

    int address = dividx + ((local_idx >> t_2) << t_2) + local_idx;

    int shrd_dixidx_t = (local_idx >> t_2) << t_2;
    int shrd_address = shrd_dixidx_t + local_idx;

    Data mult_1 = polynomial_in[address + divindex];
    Data mult_2 = polynomial_in[address + t + divindex];

    mult_1 = VALUE_GPU::mult(mult_1, w_root_of_unity_table[address], q_thread);
    mult_2 = VALUE_GPU::mult(mult_2, w_root_of_unity_table[address + t], q_thread);

    sharedmemorys[shrd_address] = mult_1;
    sharedmemorys[shrd_address + t] = mult_2;

#pragma unroll
    for(int loop_dep = 0; loop_dep < LOOP; loop_dep++)
    {
        GentlemanSandeUnit_(sharedmemorys[shrd_address], sharedmemorys[shrd_address + t],
                           n2_root_of_unity_table[(local_idx >> t_2)], q_thread);

        t = t << 1;
        t_2 += 1;

        shrd_dixidx_t = (local_idx >> t_2) << t_2;
        shrd_address = shrd_dixidx_t + local_idx;
        __syncthreads();
    }

    address = dividx + ((local_idx >> t_2) << t_2) + local_idx;

    GentlemanSandeUnit_(sharedmemorys[shrd_address], sharedmemorys[shrd_address + t],
                       n2_root_of_unity_table[(local_idx >> t_2)], q_thread);
    __syncthreads();

    Data temp1 = VALUE_GPU::mult(sharedmemorys[shrd_address], inverse, q_thread);
    polynomial_in[address + divindex] = temp1;

    Data temp2 = VALUE_GPU::mult(sharedmemorys[shrd_address + t], inverse, q_thread);
    polynomial_in[address + t + divindex] = temp2;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void FourStepPartialInverseCore1(Data* polynomial_in, Root* n2_root_of_unity_table,
                                            Root* w_root_of_unity_table, Modulus* modulus,
                                            int small_npower, int poly_n_power, int mod_count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int block_y = blockIdx.y;
    int block_z = blockIdx.z;

    int local_idx = threadIdx.x;

    __shared__ Data sharedmemorys[512];

    int small_npower_ = small_npower;
    int t_2 = 0;
    int t = 1 << t_2;

    int dividx = block_y << small_npower_;

    int divindex = block_z << poly_n_power;

    int q_index = block_z % mod_count;
    Modulus q_thread = modulus[q_index];

    int address = dividx + ((idx >> t_2) << t_2) + idx;

    int shrd_dixidx_t = (local_idx >> t_2) << t_2;
    int shrd_address = shrd_dixidx_t + local_idx;

    Data mult_1 = polynomial_in[address + divindex];
    Data mult_2 = polynomial_in[address + t + divindex];

    /*mult_1 = VALUE_GPU::mult(mult_1, w_root_of_unity_table[address], q_thread);
    mult_2 = VALUE_GPU::mult(mult_2, w_root_of_unity_table[address + t], q_thread);*/ //不需要修正这里

    sharedmemorys[shrd_address] = mult_1;
    sharedmemorys[shrd_address + t] = mult_2;

    for(int loop = 0; loop < 8; loop++)
    {
        GentlemanSandeUnit_(sharedmemorys[shrd_address], sharedmemorys[shrd_address + t],
                           n2_root_of_unity_table[(idx >> t_2)], q_thread);

        t = t << 1;
        t_2 += 1;

        shrd_dixidx_t = (local_idx >> t_2) << t_2;
        shrd_address = shrd_dixidx_t + local_idx;
        __syncthreads();
    }

    address = dividx + ((idx >> t_2) << t_2) + idx;

    GentlemanSandeUnit_(sharedmemorys[shrd_address], sharedmemorys[shrd_address + t],
                       n2_root_of_unity_table[(idx >> t_2)], q_thread);

    polynomial_in[address + divindex] = sharedmemorys[shrd_address];
    polynomial_in[address + t + divindex] = sharedmemorys[shrd_address + t];
}

__global__ void FourStepPartialInverseCore1(Data* polynomial_in, Root* n2_root_of_unity_table,
                                            Root* w_root_of_unity_table, Modulus modulus,
                                            int small_npower, int poly_n_power)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int block_y = blockIdx.y;
    int block_z = blockIdx.z;

    int local_idx = threadIdx.x;

    __shared__ Data sharedmemorys[512];

    int small_npower_ = small_npower;
    int t_2 = 0;
    int t = 1 << t_2;

    int dividx = block_y << small_npower_;

    int divindex = block_z << poly_n_power;

    Modulus q_thread = modulus;

    int address = dividx + ((idx >> t_2) << t_2) + idx;

    int shrd_dixidx_t = (local_idx >> t_2) << t_2;
    int shrd_address = shrd_dixidx_t + local_idx;

    Data mult_1 = polynomial_in[address + divindex];
    Data mult_2 = polynomial_in[address + t + divindex];

    mult_1 = VALUE_GPU::mult(mult_1, w_root_of_unity_table[address], q_thread);
    mult_2 = VALUE_GPU::mult(mult_2, w_root_of_unity_table[address + t], q_thread);

    sharedmemorys[shrd_address] = mult_1;
    sharedmemorys[shrd_address + t] = mult_2;

    for(int loop = 0; loop < 8; loop++)
    {
        GentlemanSandeUnit_(sharedmemorys[shrd_address], sharedmemorys[shrd_address + t],
                           n2_root_of_unity_table[(idx >> t_2)], q_thread);

        t = t << 1;
        t_2 += 1;

        shrd_dixidx_t = (local_idx >> t_2) << t_2;
        shrd_address = shrd_dixidx_t + local_idx;
        __syncthreads();
    }

    address = dividx + ((idx >> t_2) << t_2) + idx;

    GentlemanSandeUnit_(sharedmemorys[shrd_address], sharedmemorys[shrd_address + t],
                       n2_root_of_unity_table[(idx >> t_2)], q_thread);

    polynomial_in[address + divindex] = sharedmemorys[shrd_address];
    polynomial_in[address + t + divindex] = sharedmemorys[shrd_address + t];
}


/*
参数:polynonimal_in:n1*n2
    polynonimal_out:n2*n1
TODO:1.做转置 2.乘上修正因子*/
__global__ void FourStepPartialInverseCore2(Data* polynomial_in,Data* polynomial_out, Root* n2_root_of_unity_table, Root* w_root_of_unity_table,
                                            Modulus* modulus, int small_npower, int T, int loc1,
                                            int loc2, int loc3, int loop, Ninverse* inverse,
                                            int poly_n_power, int mod_count)
{
    int idx_x = threadIdx.x;
    int idx_y = threadIdx.y;
    int block_x = blockIdx.x;
    int block_y = blockIdx.y;
    int block_z = blockIdx.z;

    __shared__ Data sharedmemorys[512];

    int small_npower_ = small_npower;
    int t_2 = T;

    int divindex = block_z << poly_n_power;

    int q_index = block_z % mod_count;
    Modulus q_thread = modulus[q_index];

    int grid = block_y << small_npower_;

    int global_addresss = (idx_y << loc3) + idx_x + (block_x << loc1);
    int shared_addresss = (idx_x + (idx_y << loc1));

    int load_store_address = global_addresss + grid;

    // Load data from global & store to shared
    sharedmemorys[shared_addresss] = polynomial_in[load_store_address + divindex];
    sharedmemorys[shared_addresss + 256] = polynomial_in[load_store_address + loc2 + divindex];
    __syncthreads();

    int t_ = loc1;
    int t = 1 << t_;
    int in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;

    GentlemanSandeUnit_(sharedmemorys[in_shared_address], sharedmemorys[in_shared_address + t],
                       n2_root_of_unity_table[(global_addresss >> t_2)], q_thread);
    __syncthreads();

    for(int lp = 0; lp < loop; lp++)
    {
        t = t << 1;
        t_2 += 1;
        t_ += 1;

        in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;

        GentlemanSandeUnit_(sharedmemorys[in_shared_address], sharedmemorys[in_shared_address + t],
                           n2_root_of_unity_table[(global_addresss >> t_2)], q_thread);

        __syncthreads();
    }


    Data temp1 = VALUE_GPU::mult(sharedmemorys[shared_addresss], inverse[0], q_thread);
    polynomial_out[(load_store_address >> small_npower) + (load_store_address % (1 << small_npower) ) * (1 << (poly_n_power - small_npower)) + divindex] = VALUE_GPU::mult(temp1,w_root_of_unity_table[load_store_address + divindex], q_thread);//乘上修正因子

    Data temp2 = VALUE_GPU::mult(sharedmemorys[shared_addresss + 256], inverse[0], q_thread);
    polynomial_out[((load_store_address + loc2 ) >> small_npower) + ((load_store_address + loc2 ) % (1 << small_npower) ) * (1 << (poly_n_power - small_npower)) + divindex] = VALUE_GPU::mult(temp2,w_root_of_unity_table[load_store_address + loc2 + divindex], q_thread);
}

__global__ void FourStepPartialInverseCore2(Data* polynomial_in, Root* n2_root_of_unity_table,
                                            Modulus modulus, int small_npower, int T, int loc1,
                                            int loc2, int loc3, int loop, Ninverse inverse,
                                            int poly_n_power)
{
    int idx_x = threadIdx.x;
    int idx_y = threadIdx.y;
    int block_x = blockIdx.x;
    int block_y = blockIdx.y;
    int block_z = blockIdx.z;

    __shared__ Data sharedmemorys[512];

    int small_npower_ = small_npower;
    int t_2 = T;

    int divindex = block_z << poly_n_power;

    Modulus q_thread = modulus;

    int grid = block_y << small_npower_;

    int global_addresss = (idx_y << loc3) + idx_x + (block_x << loc1);
    int shared_addresss = (idx_x + (idx_y << loc1));

    int load_store_address = global_addresss + grid;

    // Load data from global & store to shared
    sharedmemorys[shared_addresss] = polynomial_in[load_store_address + divindex];
    sharedmemorys[shared_addresss + 256] = polynomial_in[load_store_address + loc2 + divindex];
    __syncthreads();

    int t_ = loc1;
    int t = 1 << t_;
    int in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;

    GentlemanSandeUnit_(sharedmemorys[in_shared_address], sharedmemorys[in_shared_address + t],
                       n2_root_of_unity_table[(global_addresss >> t_2)], q_thread);
    __syncthreads();

    for(int lp = 0; lp < loop; lp++)
    {
        t = t << 1;
        t_2 += 1;
        t_ += 1;

        in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;

        GentlemanSandeUnit_(sharedmemorys[in_shared_address], sharedmemorys[in_shared_address + t],
                           n2_root_of_unity_table[(global_addresss >> t_2)], q_thread);

        __syncthreads();
    }

    Data temp1 = VALUE_GPU::mult(sharedmemorys[shared_addresss], inverse, q_thread);
    polynomial_in[load_store_address + divindex] = temp1;

    Data temp2 = VALUE_GPU::mult(sharedmemorys[shared_addresss + 256], inverse, q_thread);
    polynomial_in[load_store_address + loc2 + divindex] = temp2;
}

/*2024-8-7:
参数:
    device_in: 逆向NTT-n1*n2维转置矩阵 正向NTT-n2*n1维转置矩阵
    device_out:逆向NTT-n2*n1维转置矩阵 正向NTT-n1*n2维转置矩阵
    mod_count:代表的是模数的个数

mod_count:传入的是1
*/

__host__ void GPU_4STEP_NTT(Data* device_in, Data* device_out, Root* n1_root_of_unity_table,
                            Root* n2_root_of_unity_table, Root* W_root_of_unity_table,
                            Modulus* modulus, ntt4step_rns_configuration cfg, int batch_size,
                            int mod_count)
{
    switch(cfg.ntt_type) //ntt_type指定了是做正向ntt还是逆向ntt
    {
        case FORWARD:
            switch(cfg.n_power)
            {
                case 12:
                    //n1=2^5,n2=2^7
                    //printf("hi");
                    FourStepForwardCoreT1<<<dim3(4, batch_size), dim3(32, 8)>>>(
                        device_in, device_out, n1_root_of_unity_table, modulus, 7, 1024, 12,
                        mod_count); //n2个n1维的NTT
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialForwardCore<<<dim3(32, batch_size), 64>>>(
                        device_out, n2_root_of_unity_table, W_root_of_unity_table, modulus, 7, 6, 1,
                        12, mod_count);//n1个n2维的NTT ,device_out为n1*n2维的矩阵
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 13:
                    // 5 + 8
                    FourStepForwardCoreT1<<<dim3(8, batch_size), dim3(32, 8)>>>(
                        device_in, device_out, n1_root_of_unity_table, modulus, 8, 2048, 13,
                        mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialForwardCore<<<dim3(32, batch_size), 128>>>(
                        device_out, n2_root_of_unity_table, W_root_of_unity_table, modulus, 8, 7, 2,
                        13, mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 14:
                    // 5 + 9
                    FourStepForwardCoreT1<<<dim3(16, batch_size), dim3(32, 8)>>>(
                        device_in, device_out, n1_root_of_unity_table, modulus, 9, 4096, 14,
                        mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialForwardCore<<<dim3(32, batch_size), 256>>>(
                        device_out, n2_root_of_unity_table, W_root_of_unity_table, modulus, 9, 8, 3,
                        14, mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 15:
                    // 6 + 9
                    FourStepForwardCoreT2<<<dim3(32, batch_size), dim3(32, 8)>>>(
                        device_in, device_out, n1_root_of_unity_table, modulus, 9, 10, 8192, 15,
                        mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialForwardCore<<<dim3(64, batch_size), 256>>>(
                        device_out, n2_root_of_unity_table, W_root_of_unity_table, modulus, 9, 8, 3,
                        15, mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 16:
                    // 7 + 9
                    FourStepForwardCoreT3<<<dim3(64, batch_size), dim3(32, 8)>>>(
                        device_in, device_out, n1_root_of_unity_table, modulus, 9, 11, 16384, 16,
                        mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialForwardCore<<<dim3(128, batch_size), 256>>>(
                        device_out, n2_root_of_unity_table, W_root_of_unity_table, modulus, 9, 8, 3,
                        16, mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 17:
                    //2^5 * 2^12
                    FourStepForwardCoreT1<<<dim3(128, batch_size), dim3(32, 8)>>>(
                        device_in, device_out, n1_root_of_unity_table, modulus, 12, 32768, 17,
                        mod_count);//device_in:n2*n1维 device_out:n1*n2维
                    THROW_IF_CUDA_ERROR(cudaGetLastError());

                    //下面两个kernel完成13层的NTT计算
                    FourStepPartialForwardCore1<<<dim3(8, 32, batch_size), dim3(64, 4)>>>(
                        device_out, n2_root_of_unity_table, W_root_of_unity_table,modulus, 12, 6,
                        2048, 3, 17, mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialForwardCore2<<<dim3(8, 32, batch_size), 256>>>(
                        device_out, n2_root_of_unity_table, modulus, 12, 17, mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());//做9层NTT

                    break;
                case 18:
                    // 5 + 13
                    FourStepForwardCoreT1<<<dim3(256, batch_size), dim3(32, 8)>>>(
                        device_in, device_out, n1_root_of_unity_table, modulus, 13, 65536, 18,
                        mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());

                    //下面两个kernel完成13层的NTT计算
                    FourStepPartialForwardCore1<<<dim3(16, 32, batch_size), dim3(32, 8)>>>(
                        device_out, n2_root_of_unity_table, W_root_of_unity_table, modulus, 13, 5,
                        4096, 4, 18, mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialForwardCore2<<<dim3(16, 32, batch_size), 256>>>(
                        device_out, n2_root_of_unity_table ,modulus, 13, 18, mod_count);//做9层NTT , W_root_of_unity_table
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 19:
                    // 5 + 14
                    FourStepForwardCoreT1<<<dim3(512, batch_size), dim3(32, 8)>>>(
                        device_in, device_out, n1_root_of_unity_table, modulus, 14, 131072, 19,
                        mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());


                    FourStepPartialForwardCore1<<<dim3(32, 32, batch_size), dim3(16, 16)>>>(
                        device_out, n2_root_of_unity_table, W_root_of_unity_table, modulus, 14, 4,
                        8192, 5, 19, mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialForwardCore2<<<dim3(32, 32, batch_size), 256>>>(
                        device_out, n2_root_of_unity_table, modulus, 14, 19, mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 20:
                    // 5 + 15
                    FourStepForwardCoreT1<<<dim3(1024, batch_size), dim3(32, 8)>>>(
                        device_in, device_out, n1_root_of_unity_table, modulus, 15, 262144, 20,
                        mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialForwardCore1<<<dim3(64, 32, batch_size), dim3(8, 32)>>>(
                        device_out, n2_root_of_unity_table, W_root_of_unity_table, modulus, 15, 3,
                        16384, 6, 20, mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialForwardCore2<<<dim3(64, 32, batch_size), 256>>>(
                        device_out, n2_root_of_unity_table, modulus, 15, 20, mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 21:
                    // 6 + 15
                    FourStepForwardCoreT2<<<dim3(2048, batch_size), dim3(32, 8)>>>(
                        device_in, device_out, n1_root_of_unity_table, modulus, 15, 16, 524288, 21,
                        mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialForwardCore1<<<dim3(64, 64, batch_size), dim3(8, 32)>>>(
                        device_out, n2_root_of_unity_table, W_root_of_unity_table, modulus, 15, 3,
                        16384, 6, 21, mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialForwardCore2<<<dim3(64, 64, batch_size), 256>>>(
                        device_out, n2_root_of_unity_table, modulus, 15, 21, mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 22:
                    // 7 + 15
                    FourStepForwardCoreT3<<<dim3(4096, batch_size), dim3(32, 8)>>>(
                        device_in, device_out, n1_root_of_unity_table, modulus, 15, 17, 1048576, 22,
                        mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialForwardCore1<<<dim3(64, 128, batch_size), dim3(8, 32)>>>(
                        device_out, n2_root_of_unity_table, W_root_of_unity_table, modulus, 15, 3,
                        16384, 6, 22, mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialForwardCore2<<<dim3(64, 128, batch_size), 256>>>(
                        device_out, n2_root_of_unity_table, modulus, 15, 22, mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 23:
                    // 7 + 16
                    FourStepForwardCoreT3<<<dim3(8192, batch_size), dim3(32, 8)>>>(
                        device_in, device_out, n1_root_of_unity_table, modulus, 16, 18, 2097152, 23,
                        mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialForwardCore1<<<dim3(128, 128, batch_size), dim3(4, 64)>>>(
                        device_out, n2_root_of_unity_table, W_root_of_unity_table, modulus, 16, 2,
                        32768, 7, 23, mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialForwardCore2<<<dim3(128, 128, batch_size), 256>>>(
                        device_out, n2_root_of_unity_table, modulus, 16, 23, mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 24:
                    // 8 + 16
                    FourStepForwardCoreT4<<<dim3(16384, batch_size), dim3(32, 8)>>>(
                        device_in, device_out, n1_root_of_unity_table, modulus, 16, 19, 4194304, 24,
                        mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialForwardCore1<<<dim3(128, 256, batch_size), dim3(4, 64)>>>(
                        device_out, n2_root_of_unity_table, W_root_of_unity_table, modulus, 16, 2,
                        32768, 7, 24, mod_count);//device_out是n1*n2维度的，
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialForwardCore2<<<dim3(128, 256, batch_size), 256>>>(
                        device_out, n2_root_of_unity_table, modulus, 16, 24, mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;

                default:
                    std::cout << "This ring size is not supported!" << std::endl;
                    break;
            }
            // ss
            break;
        case INVERSE:
            switch(cfg.n_power)
            {
                case 12:

                    /*FourStepInverseCoreT1<<<dim3(4, batch_size), dim3(32, 8)>>>(
                        device_in, device_out, n1_root_of_unity_table, modulus, 7, 1024, 12,
                        mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialInverseCore<<<dim3(32, batch_size), 64>>>(
                        device_out, n2_root_of_unity_table, W_root_of_unity_table, modulus, 7, 6,
                        cfg.mod_inverse, 12, mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());*/


                    FourStepPartialInverseCore<<<dim3(32, batch_size), 64>>>(
                        device_in,device_out, n2_root_of_unity_table, W_root_of_unity_table, modulus, 7, 6,
                        cfg.mod_inverse, 12, mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());

                    FourStepInverseCoreT1<<<dim3(4, batch_size), dim3(32, 8)>>>(
                        device_out, device_in, n1_root_of_unity_table, modulus, 7, 1024, 12,
                        mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 13:
                    FourStepPartialInverseCore<<<dim3(32, batch_size), 128>>>(
                        device_in, device_out, n2_root_of_unity_table, W_root_of_unity_table, modulus, 8, 7,
                        cfg.mod_inverse, 13, mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());

                    FourStepInverseCoreT1<<<dim3(8, batch_size), dim3(32, 8)>>>(
                        device_out, device_in, n1_root_of_unity_table, modulus, 8, 2048, 13,
                        mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 14:

                    FourStepPartialInverseCore<<<dim3(32, batch_size), 256>>>(
                        device_in,device_out, n2_root_of_unity_table, W_root_of_unity_table, modulus, 9, 8,
                        cfg.mod_inverse, 14, mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());

                    FourStepInverseCoreT1<<<dim3(16, batch_size), dim3(32, 8)>>>(
                        device_out, device_in, n1_root_of_unity_table, modulus, 9, 4096, 14,
                        mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 15:

                    FourStepPartialInverseCore<<<dim3(64, batch_size), 256>>>(
                        device_in, device_out, n2_root_of_unity_table, W_root_of_unity_table, modulus, 9, 8,
                        cfg.mod_inverse, 15, mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());

                    FourStepInverseCoreT2<<<dim3(32, batch_size), dim3(32, 8)>>>(
                        device_out, device_in, n1_root_of_unity_table, modulus, 9, 10, 8192, 15,
                        mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 16:

                    FourStepPartialInverseCore<<<dim3(128, batch_size), 256>>>(
                        device_in, device_out, n2_root_of_unity_table, W_root_of_unity_table, modulus, 9, 8,
                        cfg.mod_inverse, 16, mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());

                    FourStepInverseCoreT3<<<dim3(64, batch_size), dim3(32, 8)>>>(
                        device_out, device_in, n1_root_of_unity_table, modulus, 9, 11, 16384, 16,
                        mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 17:
                    //2^5 * 2^12 
                    FourStepPartialInverseCore1<<<dim3(8, 32, batch_size), 256>>>(
                        device_in, n2_root_of_unity_table, W_root_of_unity_table, modulus, 12, 17,
                        mod_count); //9层
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialInverseCore2<<<dim3(8, 32, batch_size), dim3(64, 4)>>>(
                        device_in, device_out, n2_root_of_unity_table, W_root_of_unity_table, modulus, 12, 9, 6, 2048, 9, 2,
                        cfg.mod_inverse, 17, mod_count); //3层
                    THROW_IF_CUDA_ERROR(cudaGetLastError());

                    FourStepInverseCoreT1<<<dim3(128, batch_size), dim3(32, 8)>>>(
                        device_out, device_in, n1_root_of_unity_table, modulus, 12, 32768, 17,
                        mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 18:

                    FourStepPartialInverseCore1<<<dim3(16, 32, batch_size), 256>>>(
                        device_in, n2_root_of_unity_table, W_root_of_unity_table, modulus, 13, 18,
                        mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialInverseCore2<<<dim3(16, 32, batch_size), dim3(32, 8)>>>(
                        device_in, device_out, n2_root_of_unity_table,  W_root_of_unity_table, modulus, 13, 9, 5, 4096, 9, 3,
                        cfg.mod_inverse, 18, mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());

                    FourStepInverseCoreT1<<<dim3(256, batch_size), dim3(32, 8)>>>(
                        device_out, device_in, n1_root_of_unity_table, modulus, 13, 65536, 18,
                        mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    
                    break;
                case 19:

                    FourStepPartialInverseCore1<<<dim3(32, 32, batch_size), 256>>>(
                        device_in, n2_root_of_unity_table, W_root_of_unity_table, modulus, 14, 19,
                        mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialInverseCore2<<<dim3(32, 32, batch_size), dim3(16, 16)>>>(
                        device_in, device_out, n2_root_of_unity_table, W_root_of_unity_table, modulus, 14, 9, 4, 8192, 9, 4,
                        cfg.mod_inverse, 19, mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());

                    FourStepInverseCoreT1<<<dim3(512, batch_size), dim3(32, 8)>>>(
                        device_out, device_in, n1_root_of_unity_table, modulus, 14, 131072, 19,
                        mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 20:

                    FourStepPartialInverseCore1<<<dim3(64, 32, batch_size), 256>>>(
                        device_in, n2_root_of_unity_table, W_root_of_unity_table, modulus, 15, 20,
                        mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialInverseCore2<<<dim3(64, 32, batch_size), dim3(8, 32)>>>(
                        device_in, device_out, n2_root_of_unity_table, W_root_of_unity_table, modulus, 15, 9, 3, 16384, 9, 5,
                        cfg.mod_inverse, 20, mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());

                    FourStepInverseCoreT1<<<dim3(1024, batch_size), dim3(32, 8)>>>(
                        device_out, device_in, n1_root_of_unity_table, modulus, 15, 262144, 20,
                        mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 21:

                    FourStepPartialInverseCore1<<<dim3(64, 64, batch_size), 256>>>(
                        device_in, n2_root_of_unity_table, W_root_of_unity_table, modulus, 15, 21,
                        mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialInverseCore2<<<dim3(64, 64, batch_size), dim3(8, 32)>>>(
                        device_in, device_out, n2_root_of_unity_table, W_root_of_unity_table,modulus, 15, 9, 3, 16384, 9, 5,
                        cfg.mod_inverse, 21, mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());

                    FourStepInverseCoreT2<<<dim3(2048, batch_size), dim3(32, 8)>>>(
                        device_out, device_in, n1_root_of_unity_table, modulus, 15, 16, 524288, 21,
                        mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 22:

                    FourStepPartialInverseCore1<<<dim3(64, 128, batch_size), 256>>>(
                        device_in, n2_root_of_unity_table, W_root_of_unity_table, modulus, 15, 22,
                        mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialInverseCore2<<<dim3(64, 128, batch_size), dim3(8, 32)>>>(
                        device_in, device_out, n2_root_of_unity_table, W_root_of_unity_table, modulus, 15, 9, 3, 16384, 9, 5,
                        cfg.mod_inverse, 22, mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());

                    FourStepInverseCoreT3<<<dim3(4096, batch_size), dim3(32, 8)>>>(
                        device_out, device_in, n1_root_of_unity_table, modulus, 15, 17, 1048576, 22,
                        mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 23:

                    FourStepPartialInverseCore1<<<dim3(128, 128, batch_size), 256>>>(
                        device_in, n2_root_of_unity_table, W_root_of_unity_table, modulus, 16, 23,
                        mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialInverseCore2<<<dim3(128, 128, batch_size), dim3(4, 64)>>>(
                        device_in, device_out, n2_root_of_unity_table, W_root_of_unity_table, modulus, 16, 9, 2, 32768, 9, 6,
                        cfg.mod_inverse, 23, mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());

                    FourStepInverseCoreT3<<<dim3(8192, batch_size), dim3(32, 8)>>>(
                        device_out, device_in, n1_root_of_unity_table, modulus, 16, 18, 2097152, 23,
                        mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 24:

                    FourStepPartialInverseCore1<<<dim3(128, 256, batch_size), 256>>>(
                        device_in, n2_root_of_unity_table, W_root_of_unity_table, modulus, 16, 24,
                        mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialInverseCore2<<<dim3(128, 256, batch_size), dim3(4, 64)>>>(
                        device_in, device_out, n2_root_of_unity_table, W_root_of_unity_table, modulus, 16, 9, 2, 32768, 9, 6,
                        cfg.mod_inverse, 24, mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());

                    FourStepInverseCoreT4<<<dim3(16384, batch_size), dim3(32, 8)>>>(
                        device_out, device_in, n1_root_of_unity_table, modulus, 16, 19, 4194304, 24,
                        mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;

                default:
                    std::cout << "This ring size is not supported!" << std::endl;
                    break;
            }

            break;

        default:
            break;
    }
}

/*2024-8-12:
dim3(8,32,batch_size),dim3(32,4)
8 * 32 * 4个线程完成4096个数组的转置*/
__global__ void small_transpose_4096(Data * device_in, Data * device_out, int n_power){
    int global_start = (blockIdx.z << n_power) + (blockIdx.y << 12); //定位到哪一个4096
    int local_start = (blockIdx.x & 1) * 32 + ((blockIdx.x >> 1) << 10);//(blockIdx.x & 1)代表列 ，(blockIdx.x >> 1)代表行，每行需要乘上16 * 64 = 1024
    __shared__ Data coes[16][33];

    //需要做两个cpu cycle
    coes[threadIdx.y][threadIdx.x] = device_in[global_start + local_start + (threadIdx.y << 6) + threadIdx.x]; //32个线程正好一起完成

    coes[threadIdx.y + 4][threadIdx.x] = device_in[global_start + local_start + ((threadIdx.y + 4)<< 6) + threadIdx.x];

    coes[threadIdx.y + 8][threadIdx.x] = device_in[global_start + local_start + ((threadIdx.y + 8)<< 6) + threadIdx.x];

    coes[threadIdx.y + 12][threadIdx.x] = device_in[global_start + local_start + ((threadIdx.y + 12)<< 6) + threadIdx.x];

    __syncthreads();

    //一定要横着填,转置后的分块矩阵位置为((blockIdx.x & 1),(blockIdx.x >> 1))
    local_start = ((blockIdx.x & 1) << 11) + ((blockIdx.x >> 1) << 4); //分块矩阵左上点的位置
    //在分块矩阵中的位置为(threadIdx.y * 2 + threadIdx.y % 2, threadIdx.x % 16S)
    device_out[global_start + local_start + (((threadIdx.y << 1) + (threadIdx.x >> 4) )<< 6) + (threadIdx.x & 15)] = coes[threadIdx.x & 15][(threadIdx.y << 1) + (threadIdx.x >> 4)];

    device_out[global_start + local_start + (((threadIdx.y << 1) + (threadIdx.x >> 4) + 8)<< 6) + (threadIdx.x & 15)] = coes[threadIdx.x & 15][(threadIdx.y << 1) + (threadIdx.x >> 4) + 8];

    device_out[global_start + local_start + (((threadIdx.y << 1) + (threadIdx.x >> 4) + 16)<< 6) + (threadIdx.x & 15)] = coes[threadIdx.x & 15][(threadIdx.y << 1) + (threadIdx.x >> 4) + 16]; //这条语句有问题

    device_out[global_start + local_start + (((threadIdx.y << 1) + (threadIdx.x >> 4) + 24)<< 6) + (threadIdx.x & 15)] = coes[threadIdx.x & 15][(threadIdx.y << 1) + (threadIdx.x >> 4) + 24];

}

/*2024-8-16:
在减少了存储体冲突的基础上,减少每个线程执行的存取操作数量，试图减少warp的stall
dim3(16,4096的个数,batch_size),dim3(16,16)*/
__global__ void small_transpose_4096_v2(Data * device_in, Data * device_out, int n_power){
    int global_start = (blockIdx.z << n_power) + (blockIdx.y << 12); //定位到哪一个4096
    int local_start = ((blockIdx.x >> 2) << 10) + ((blockIdx.x & 3) << 4 );//blockIdx.x对应的分块矩阵的下标为(blockIdx.x / 4, blockIdx.x % 4)，其对应在4096中的下标位置为(blockIdx.x / 4) * 16 * 64 + (blockIdx.x % 4) * 16
    __shared__ Data coes[16][18];

    //需要做两个cpu cycle
   

    coes[threadIdx.y][threadIdx.x] = device_in[global_start + local_start + (threadIdx.y << 6) + threadIdx.x];

    __syncthreads();

    //一定要横着填,转置后的分块矩阵位置为((blockIdx.x & 1),(blockIdx.x >> 1))
    local_start = ((blockIdx.x & 3) << 10) + ((blockIdx.x >> 2) << 4); //分块矩阵左上点的位置
    //在分块矩阵中的位置为(threadIdx.y * 2 + threadIdx.y % 2, threadIdx.x % 16S)
    device_out[global_start + local_start + (threadIdx.y << 6) + threadIdx.x] = coes[threadIdx.x][threadIdx.y];
}


/*2024-8-12:
dim3(8,32,batch_size),dim3(32,8)
8 * 32 * 4个线程完成4096个数组的转置*/
__global__ void small_transpose_4096_v3(Data * device_in, Data * device_out, int n_power){
    int global_start = (blockIdx.z << n_power) + (blockIdx.y << 12); //定位到哪一个4096
    int local_start = ((blockIdx.x & 1) << 5) + ((blockIdx.x >> 1) << 10);//(blockIdx.x & 1)代表列 ，(blockIdx.x >> 1)代表行，每行需要乘上16 * 64 = 1024
    __shared__ Data coes[16][33];

    //需要做两个cpu cycle
    coes[threadIdx.y][threadIdx.x] = device_in[global_start + local_start + (threadIdx.y << 6) + threadIdx.x]; //32个线程正好一起完成

    coes[threadIdx.y + 8][threadIdx.x] = device_in[global_start + local_start + ((threadIdx.y + 8)<< 6) + threadIdx.x];


    __syncthreads();

    //一定要横着填,转置后的分块矩阵位置为((blockIdx.x & 1),(blockIdx.x >> 1))
    local_start = ((blockIdx.x & 1) << 11) + ((blockIdx.x >> 1) << 4); //分块矩阵左上点的位置
    //在分块矩阵中的位置为(threadIdx.y * 2 + threadIdx.y % 2, threadIdx.x % 16S)
    device_out[global_start + local_start + (((threadIdx.y << 1) + (threadIdx.x >> 4) )<< 6) + (threadIdx.x & 15)] = coes[threadIdx.x & 15][(threadIdx.y << 1) + (threadIdx.x >> 4)];

    device_out[global_start + local_start + (((threadIdx.y << 1) + (threadIdx.x >> 4) + 16)<< 6) + (threadIdx.x & 15)] = coes[threadIdx.x & 15][(threadIdx.y << 1) + (threadIdx.x >> 4) + 16];


}

/*2024-8-12:
参数:length:代表以length为单位进行转置
    n_power:代表的是n值*/
void small_transpose(Data* device_in, Data* device_out,int length,int batch_size,int n_power){
    if(length == 4096){
        /*printf("small_transpose_4096 %d\n",(int)((1<<n_power) / length));
        small_transpose_4096<<<dim3(8,(int)((1<<n_power) / length),batch_size),dim3(32,4)>>>(device_in,device_out,n_power);
        printf("small_transpose_4096 DONE\n");*/

        /*printf("small_transpose_4096_v2 %d\n",(int)((1<<n_power) / length));
        small_transpose_4096_v2<<<dim3(16,(int)((1<<n_power) / length),batch_size),dim3(16,16)>>>(device_in,device_out,n_power);
        printf("small_transpose_4096_v2 DONE\n");*/

        printf("small_transpose_4096_v3 %d\n",(int)((1<<n_power) / length));
        small_transpose_4096_v3<<<dim3(8,(int)((1<<n_power) / length),batch_size),dim3(32,8)>>>(device_in,device_out,n_power);
        printf("small_transpose_4096_v3 DONE\n");

    }
}


__global__ void small_transpose_4096_W(Data * device_in, Data * device_out, int n_power, Root * W_big_root_table, Modulus* modulus,int mod_count){
    int global_start = (blockIdx.z << n_power) + (blockIdx.y << 12); //定位到哪一个4096
    int local_start = (blockIdx.x & 1) * 32 + ((blockIdx.x >> 1) << 10);//(blockIdx.x & 1)代表列 ，(blockIdx.x >> 1)代表行，每行需要乘上16 * 64 = 1024
    __shared__ Data coes[16][33];
    Modulus q_thread = modulus[blockIdx.z % mod_count];

    coes[threadIdx.y][threadIdx.x] = VALUE_GPU::mult(device_in[global_start + local_start + (threadIdx.y << 6) + threadIdx.x] ,W_big_root_table[(blockIdx.y << 12) + local_start + (threadIdx.y << 6) + threadIdx.x],q_thread); //32个线程正好一起完成

    coes[threadIdx.y + 4][threadIdx.x] = VALUE_GPU::mult(device_in[global_start + local_start + ((threadIdx.y + 4)<< 6) + threadIdx.x] ,W_big_root_table[(blockIdx.y << 12) + local_start + ((threadIdx.y + 4)<< 6) + threadIdx.x],q_thread);

    coes[threadIdx.y + 8][threadIdx.x] = VALUE_GPU::mult(device_in[global_start + local_start + ((threadIdx.y + 8)<< 6) + threadIdx.x],W_big_root_table[(blockIdx.y << 12) + local_start + ((threadIdx.y + 8)<< 6) + threadIdx.x],q_thread);

    coes[threadIdx.y + 12][threadIdx.x] = VALUE_GPU::mult(device_in[global_start + local_start + ((threadIdx.y + 12)<< 6) + threadIdx.x],W_big_root_table[(blockIdx.y << 12) + local_start + ((threadIdx.y + 12)<< 6) + threadIdx.x],q_thread);

    __syncthreads();

    
    //一定要横着填,转置后的分块矩阵位置为((blockIdx.x & 1),(blockIdx.x >> 1))
    local_start = ((blockIdx.x & 1) << 11) + ((blockIdx.x >> 1) << 4); //分块矩阵左上点的位置
    //在分块矩阵中的位置为(threadIdx.y * 2 + threadIdx.y % 2, threadIdx.x % 16S)
    device_out[global_start + local_start + (((threadIdx.y << 1) + (threadIdx.x >> 4) )<< 6) + (threadIdx.x & 15)] = coes[threadIdx.x & 15][(threadIdx.y << 1) + (threadIdx.x >> 4)];

    device_out[global_start + local_start + (((threadIdx.y << 1) + (threadIdx.x >> 4) + 8)<< 6) + (threadIdx.x & 15)] = coes[threadIdx.x & 15][(threadIdx.y << 1) + (threadIdx.x >> 4) + 8];

    device_out[global_start + local_start + (((threadIdx.y << 1) + (threadIdx.x >> 4) + 16)<< 6) + (threadIdx.x & 15)] = coes[threadIdx.x & 15][(threadIdx.y << 1) + (threadIdx.x >> 4) + 16]; //这条语句有问题

    device_out[global_start + local_start + (((threadIdx.y << 1) + (threadIdx.x >> 4) + 24)<< 6) + (threadIdx.x & 15)] = coes[threadIdx.x & 15][(threadIdx.y << 1) + (threadIdx.x >> 4) + 24];


}

/*2024-8-12:
参数:length:代表以length为单位进行转置
    n_power:代表的是n值
    
乘上n的修正因子*/
void small_transpose_mult_W(Data* device_in, Data* device_out,int length,int batch_size,int n_power,Root * W_big_root_table, Modulus *modulus,int mod_count){
    if(length == 4096){
        printf("small_transpose_4096 %d\n",(int)((1<<n_power) / length));
        small_transpose_4096_W<<<dim3(8,(int)((1<<n_power) / length),batch_size),dim3(32,4)>>>(device_in,device_out,n_power,W_big_root_table,modulus,mod_count);
        printf("small_transpose_4096 DONE\n");

    }
}

/*2024-8-12:
dim3(8,32,batch_size),dim3(32,4)
8 * 32 * 4个线程完成4096个数组的转置*/
__global__ void small_transpose_mult_4096(Data * device_in, Data * device_out, int n_power, Root * W_small_root_table, Modulus* modulus,int mod_count){
    int global_start = (blockIdx.z << n_power) + (blockIdx.y << 12); //定位到哪一个4096
    int local_start = (blockIdx.x & 1) * 32 + ((blockIdx.x >> 1) << 10);//(blockIdx.x & 1)代表列 ，(blockIdx.x >> 1)代表行，每行需要乘上16 * 64 = 1024
    __shared__ Data coes[16][33];
    Modulus q_thread = modulus[blockIdx.z % mod_count];
    coes[threadIdx.y][threadIdx.x] = device_in[global_start + local_start + (threadIdx.y << 6) + threadIdx.x]; //32个线程正好一起完成

    coes[threadIdx.y + 4][threadIdx.x] = device_in[global_start + local_start + ((threadIdx.y + 4)<< 6) + threadIdx.x];

    coes[threadIdx.y + 8][threadIdx.x] = device_in[global_start + local_start + ((threadIdx.y + 8)<< 6) + threadIdx.x];

    coes[threadIdx.y + 12][threadIdx.x] = device_in[global_start + local_start + ((threadIdx.y + 12)<< 6) + threadIdx.x];

    __syncthreads();
    
    //一定要横着填,转置后的分块矩阵位置为((blockIdx.x & 1),(blockIdx.x >> 1))
    local_start = ((blockIdx.x & 1) << 11) + ((blockIdx.x >> 1) << 4); //分块矩阵左上点的位置
    //在分块矩阵中的位置为(threadIdx.y * 2 + threadIdx.y % 2, threadIdx.x % 16S)
    device_out[global_start + local_start + (((threadIdx.y << 1) + (threadIdx.x >> 4) )<< 6) + (threadIdx.x & 15)] = VALUE_GPU::mult(coes[threadIdx.x & 15][(threadIdx.y << 1) + (threadIdx.x >> 4)],W_small_root_table[local_start + (((threadIdx.y << 1) + (threadIdx.x >> 4) )<< 6) + (threadIdx.x & 15)],q_thread);

    device_out[global_start + local_start + (((threadIdx.y << 1) + (threadIdx.x >> 4) + 8)<< 6) + (threadIdx.x & 15)] = VALUE_GPU::mult(coes[threadIdx.x & 15][(threadIdx.y << 1) + (threadIdx.x >> 4) + 8],W_small_root_table[local_start + (((threadIdx.y << 1) + (threadIdx.x >> 4) + 8)<< 6) + (threadIdx.x & 15)],q_thread);

    device_out[global_start + local_start + (((threadIdx.y << 1) + (threadIdx.x >> 4) + 16)<< 6) + (threadIdx.x & 15)] = VALUE_GPU::mult(coes[threadIdx.x & 15][(threadIdx.y << 1) + (threadIdx.x >> 4) + 16],W_small_root_table[local_start + (((threadIdx.y << 1) + (threadIdx.x >> 4) + 16)<< 6) + (threadIdx.x & 15)],q_thread); //这条语句有问题

    device_out[global_start + local_start + (((threadIdx.y << 1) + (threadIdx.x >> 4) + 24)<< 6) + (threadIdx.x & 15)] = VALUE_GPU::mult(coes[threadIdx.x & 15][(threadIdx.y << 1) + (threadIdx.x >> 4) + 24],W_small_root_table[local_start + (((threadIdx.y << 1) + (threadIdx.x >> 4) + 24)<< 6) + (threadIdx.x & 15)],q_thread);
}

/*2024-8-12:
参数:length:代表以length为单位进行转置
    n_power:代表的是n值
    
乘上4096的修正因子*/
void small_transpose_mult(Data* device_in, Data* device_out,int length,int batch_size,int n_power,Root * W_small_root_table, Modulus *modulus,int mod_count){
    if(length == 4096){
        printf("small_transpose_4096\n");
        small_transpose_mult_4096<<<dim3(8,(int)((1<<n_power) / length),batch_size),dim3(32,4)>>>(device_in,device_out,n_power,W_small_root_table,modulus,mod_count);
        printf("small_transpose_4096 DONE\n");

    }
}

/*2024-8-24:
用于测试对于64维度旋转因子的常量内存的访问*/
__global__ void constant_m(){
    printf("[in constant_m]%lld\n",Csitable64[1]); //这里应该如何来实现呢
    //printf("1\n");
}
/*2024-8-9:*/
__host__ void GPU_4STEP_NTT_hxw(Data* device_in, Data* device_out, Root* n1_root_of_unity_table, Root* n2_root_of_unity_table, Root* W_root_of_unity_table, Root * n64_root_of_unity_table, Root * n64_W_root_of_unity_table, Modulus* modulus, ntt4step_rns_configuration cfg, int batch_size, int mod_count){
    switch(cfg.ntt_type) //ntt_type指定了是做正向ntt还是逆向ntt
    {
        case FORWARD:
            switch(cfg.n_power){
                    
                case 12:
                    //测试small_transpose()函数
                    /*small_transpose(device_in, device_out,4096,batch_size,12);
                    cudaDeviceSynchronize();
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;*/
                    /*printf("v1\n");
                    FourStepFowardCoreFull<<<dim3(1,batch_size),dim3(32,8)>>>(device_out,n64_root_of_unity_table,n64_W_root_of_unity_table, W_root_of_unity_table,mod_count,12,modulus);//out:n1*n2
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;*/
                    //DEBUG:x轴优先填满32个线程为一个warp
                    printf("v2\n");
                    FourStepFowardCore12_1<<<dim3(16,batch_size),dim3(32,4)>>>(device_in, device_out,n64_root_of_unity_table,n64_W_root_of_unity_table, W_root_of_unity_table, mod_count ,12, modulus);

                    FourStepFowardCore12_2<<<dim3(16,batch_size),dim3(32,4)>>>(device_out,n64_root_of_unity_table,n64_W_root_of_unity_table, W_root_of_unity_table, mod_count ,12, modulus);
                    
                    break;
                case 16:
                    //2^8 * 2^8
                    printf("test for 16\n");
                    FourStepFowardCore256<<<dim3(128, batch_size), dim3(32, 2)>>>(device_in,n1_root_of_unity_table, mod_count ,16, modulus,device_out);//device_in:n2*n1维的
                    //Transpose_Batch<<<dim3(16, 16, batch_size), dim3(16, 16)>>>(device_in, device_out, 256, 256, 16);//n1*n2维的
                    FourStepFowardCore256<<<dim3(128, batch_size), dim3(32, 2)>>>(device_out,n2_root_of_unity_table, W_root_of_unity_table, mod_count ,16, modulus);

                    break;
                case 17:
                    
                    /*printf("small transpose\n");
                    small_transpose(device_out,device_in,4096,1,17);
                    small_transpose(device_in,device_out,4096,1,17);
                    break;*/
                    //5 + 12
                    printf("17 test for 4096 5555\n");
                    FourStepForwardCoreT1<<<dim3(128, batch_size), dim3(32, 8)>>>(
                        device_in, device_out, n1_root_of_unity_table, modulus, 12, 32768, 17,
                        mod_count);//device_in:n2*n1维 device_out:n1*n2维
                    THROW_IF_CUDA_ERROR(cudaGetLastError());

                    /*printf("v1\n");
                    FourStepFowardCoreFull<<<dim3(32,batch_size),dim3(32,8)>>>(device_out,n64_root_of_unity_table,n64_W_root_of_unity_table, W_root_of_unity_table,mod_count,17,modulus);

                    THROW_IF_CUDA_ERROR(cudaGetLastError());*/
                    
                    printf("v2\n");
                    //small_transpose(device_out,device_in,4096,1,17);
                    small_transpose_mult_W(device_out,device_in,4096,1,17,W_root_of_unity_table,modulus,mod_count);
                    //FourStepFowardCore12_2<<<dim3(512,batch_size),dim3(32,4)>>>(device_in,n64_root_of_unity_table,n64_W_root_of_unity_table, W_root_of_unity_table, mod_count ,17, modulus); //n2乘上n1维度

                    FourStepFowardCore12_2<<<dim3(512,batch_size),dim3(32,4)>>>(device_in,n64_root_of_unity_table,n64_W_root_of_unity_table, W_root_of_unity_table, mod_count ,17, modulus); //n2乘上n1维度
                    small_transpose_mult(device_in,device_out,4096,1,17,n64_W_root_of_unity_table,modulus,mod_count);
                    //FourStepFowardCore12_2<<<dim3(512,batch_size),dim3(32,4)>>>(device_out,n64_root_of_unity_table,n64_W_root_of_unity_table, W_root_of_unity_table, mod_count ,1, modulus); //n2乘上n1维度
                    FourStepFowardCore12_2<<<dim3(512,batch_size),dim3(32,4)>>>(device_out,n64_root_of_unity_table,n64_W_root_of_unity_table, W_root_of_unity_table, mod_count ,1, modulus); //n2乘上n1维度

                    break;

                case 24:
                    //2^12 * 2^12
                    //step1:n1 * n2 转置维n2 * n1，且n1转置为n11* n12
                    small_transpose(device_in,device_out,4096,batch_size,24);
                    FourStepFowardCore12_2<<<dim3(65536,batch_size),dim3(32,4)>>>(device_out,n64_root_of_unity_table,n64_W_root_of_unity_table, W_root_of_unity_table, mod_count ,24, modulus);
                    small_transpose_mult(device_out,device_in,4096,batch_size,24,n64_W_root_of_unity_table,modulus,mod_count);
                    FourStepFowardCore12_2<<<dim3(65536,batch_size),dim3(32,4)>>>(device_in,n64_root_of_unity_table,n64_W_root_of_unity_table, W_root_of_unity_table, mod_count ,24, modulus);//n1*n2维度

                    GPU_Transpose(device_in, device_out, 4096, 4096, 24, batch_size);

                    small_transpose_mult_W(device_out,device_in,4096,batch_size,24,W_root_of_unity_table,modulus,mod_count);
                    FourStepFowardCore12_2<<<dim3(65536,batch_size),dim3(32,4)>>>(device_in,n64_root_of_unity_table,n64_W_root_of_unity_table, W_root_of_unity_table, mod_count ,24, modulus);
                    small_transpose_mult(device_in,device_out,4096,batch_size,24,n64_W_root_of_unity_table,modulus,mod_count);
                    FourStepFowardCore12_2<<<dim3(65536,batch_size),dim3(32,4)>>>(device_out,n64_root_of_unity_table,n64_W_root_of_unity_table, W_root_of_unity_table, mod_count ,24, modulus);//n1*n2维度

                    THROW_IF_CUDA_ERROR(cudaGetLastError());

                    break; 
                
                default:
                    printf("hxw not allowed n_power\n");
            }
            break;

        case INVERSE:

            break;
        
        default:
            break;
    }
}

__host__ void GPU_4STEP_NTT(Data* device_in, Data* device_out, Root* n1_root_of_unity_table,
                            Root* n2_root_of_unity_table, Root* W_root_of_unity_table,
                            Modulus modulus, ntt4step_configuration cfg, int batch_size)
{
    switch(cfg.ntt_type)
    {
        case FORWARD:
            switch(cfg.n_power)
            {
                case 12:

                    FourStepForwardCoreT1<<<dim3(4, batch_size), dim3(32, 8)>>>(
                        device_in, device_out, n1_root_of_unity_table, modulus, 7, 1024, 12);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialForwardCore<<<dim3(32, batch_size), 64>>>(
                        device_out, n2_root_of_unity_table, W_root_of_unity_table, modulus, 7, 6, 1,
                        12);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 13:

                    FourStepForwardCoreT1<<<dim3(8, batch_size), dim3(32, 8)>>>(
                        device_in, device_out, n1_root_of_unity_table, modulus, 8, 2048, 13);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialForwardCore<<<dim3(32, batch_size), 128>>>(
                        device_out, n2_root_of_unity_table, W_root_of_unity_table, modulus, 8, 7, 2,
                        13);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 14:

                    FourStepForwardCoreT1<<<dim3(16, batch_size), dim3(32, 8)>>>(
                        device_in, device_out, n1_root_of_unity_table, modulus, 9, 4096, 14);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialForwardCore<<<dim3(32, batch_size), 256>>>(
                        device_out, n2_root_of_unity_table, W_root_of_unity_table, modulus, 9, 8, 3,
                        14);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 15:

                    FourStepForwardCoreT2<<<dim3(32, batch_size), dim3(32, 8)>>>(
                        device_in, device_out, n1_root_of_unity_table, modulus, 9, 10, 8192, 15);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialForwardCore<<<dim3(64, batch_size), 256>>>(
                        device_out, n2_root_of_unity_table, W_root_of_unity_table, modulus, 9, 8, 3,
                        15);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 16:

                    FourStepForwardCoreT3<<<dim3(64, batch_size), dim3(32, 8)>>>(
                        device_in, device_out, n1_root_of_unity_table, modulus, 9, 11, 16384, 16);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialForwardCore<<<dim3(128, batch_size), 256>>>(
                        device_out, n2_root_of_unity_table, W_root_of_unity_table, modulus, 9, 8, 3,
                        16);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 17:
                    // 5 + 12
                    FourStepForwardCoreT1<<<dim3(128, batch_size), dim3(32, 8)>>>(
                        device_in, device_out, n1_root_of_unity_table, modulus, 12, 32768, 17);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialForwardCore1<<<dim3(8, 32, batch_size), dim3(64, 4)>>>(
                        device_out, n2_root_of_unity_table, W_root_of_unity_table, modulus, 12, 6,
                        2048, 3, 17);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialForwardCore2<<<dim3(8, 32, batch_size), 256>>>(
                        device_out, n2_root_of_unity_table, modulus, 12, 17);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 18:
                    // 5 + 13
                    FourStepForwardCoreT1<<<dim3(256, batch_size), dim3(32, 8)>>>(
                        device_in, device_out, n1_root_of_unity_table, modulus, 13, 65536, 18);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialForwardCore1<<<dim3(16, 32, batch_size), dim3(32, 8)>>>(
                        device_out, n2_root_of_unity_table, W_root_of_unity_table, modulus, 13, 5,
                        4096, 4, 18);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialForwardCore2<<<dim3(16, 32, batch_size), 256>>>(
                        device_out, n2_root_of_unity_table, modulus, 13, 18);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 19:

                    FourStepForwardCoreT1<<<dim3(512, batch_size), dim3(32, 8)>>>(
                        device_in, device_out, n1_root_of_unity_table, modulus, 14, 131072, 19);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialForwardCore1<<<dim3(32, 32, batch_size), dim3(16, 16)>>>(
                        device_out, n2_root_of_unity_table, W_root_of_unity_table, modulus, 14, 4,
                        8192, 5, 19);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialForwardCore2<<<dim3(32, 32, batch_size), 256>>>(
                        device_out, n2_root_of_unity_table, modulus, 14, 19);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 20:

                    FourStepForwardCoreT1<<<dim3(1024, batch_size), dim3(32, 8)>>>(
                        device_in, device_out, n1_root_of_unity_table, modulus, 15, 262144, 20);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialForwardCore1<<<dim3(64, 32, batch_size), dim3(8, 32)>>>(
                        device_out, n2_root_of_unity_table, W_root_of_unity_table, modulus, 15, 3,
                        16384, 6, 20);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialForwardCore2<<<dim3(64, 32, batch_size), 256>>>(
                        device_out, n2_root_of_unity_table, modulus, 15, 20);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 21:

                    FourStepForwardCoreT2<<<dim3(2048, batch_size), dim3(32, 8)>>>(
                        device_in, device_out, n1_root_of_unity_table, modulus, 15, 16, 524288, 21);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialForwardCore1<<<dim3(64, 64, batch_size), dim3(8, 32)>>>(
                        device_out, n2_root_of_unity_table, W_root_of_unity_table, modulus, 15, 3,
                        16384, 6, 21);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialForwardCore2<<<dim3(64, 64, batch_size), 256>>>(
                        device_out, n2_root_of_unity_table, modulus, 15, 21);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 22:

                    FourStepForwardCoreT3<<<dim3(4096, batch_size), dim3(32, 8)>>>(
                        device_in, device_out, n1_root_of_unity_table, modulus, 15, 17, 1048576, 22);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialForwardCore1<<<dim3(64, 128, batch_size), dim3(8, 32)>>>(
                        device_out, n2_root_of_unity_table, W_root_of_unity_table, modulus, 15, 3,
                        16384, 6, 22);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialForwardCore2<<<dim3(64, 128, batch_size), 256>>>(
                        device_out, n2_root_of_unity_table, modulus, 15, 22);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 23:

                    FourStepForwardCoreT3<<<dim3(8192, batch_size), dim3(32, 8)>>>(
                        device_in, device_out, n1_root_of_unity_table, modulus, 16, 18, 2097152, 23);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialForwardCore1<<<dim3(128, 128, batch_size), dim3(4, 64)>>>(
                        device_out, n2_root_of_unity_table, W_root_of_unity_table, modulus, 16, 2,
                        32768, 7, 23);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialForwardCore2<<<dim3(128, 128, batch_size), 256>>>(
                        device_out, n2_root_of_unity_table, modulus, 16, 23);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 24:

                    FourStepForwardCoreT4<<<dim3(16384, batch_size), dim3(32, 8)>>>(
                        device_in, device_out, n1_root_of_unity_table, modulus, 16, 19, 4194304, 24);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialForwardCore1<<<dim3(128, 256, batch_size), dim3(4, 64)>>>(
                        device_out, n2_root_of_unity_table, W_root_of_unity_table, modulus, 16, 2,
                        32768, 7, 24);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialForwardCore2<<<dim3(128, 256, batch_size), 256>>>(
                        device_out, n2_root_of_unity_table, modulus, 16, 24);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;

                default:
                    std::cout << "This ring size is not supported!" << std::endl;
                    break;
            }
            // ss
            break;
        case INVERSE:
            switch(cfg.n_power)
            {
                case 12:

                    FourStepInverseCoreT1<<<dim3(4, batch_size), dim3(32, 8)>>>(
                        device_in, device_out, n1_root_of_unity_table, modulus, 7, 1024, 12);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialInverseCore<<<dim3(32, batch_size), 64>>>(
                        device_out, n2_root_of_unity_table, W_root_of_unity_table, modulus, 7, 6,
                        cfg.mod_inverse, 12);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 13:

                    FourStepInverseCoreT1<<<dim3(8, batch_size), dim3(32, 8)>>>(
                        device_in, device_out, n1_root_of_unity_table, modulus, 8, 2048, 13);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialInverseCore<<<dim3(32, batch_size), 128>>>(
                        device_out, n2_root_of_unity_table, W_root_of_unity_table, modulus, 8, 7,
                        cfg.mod_inverse, 13);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 14:

                    FourStepInverseCoreT1<<<dim3(16, batch_size), dim3(32, 8)>>>(
                        device_in, device_out, n1_root_of_unity_table, modulus, 9, 4096, 14);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialInverseCore<<<dim3(32, batch_size), 256>>>(
                        device_out, n2_root_of_unity_table, W_root_of_unity_table, modulus, 9, 8,
                        cfg.mod_inverse, 14);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 15:

                    FourStepInverseCoreT2<<<dim3(32, batch_size), dim3(32, 8)>>>(
                        device_in, device_out, n1_root_of_unity_table, modulus, 9, 10, 8192, 15);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialInverseCore<<<dim3(64, batch_size), 256>>>(
                        device_out, n2_root_of_unity_table, W_root_of_unity_table, modulus, 9, 8,
                        cfg.mod_inverse, 15);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 16:

                    FourStepInverseCoreT3<<<dim3(64, batch_size), dim3(32, 8)>>>(
                        device_in, device_out, n1_root_of_unity_table, modulus, 9, 11, 16384, 16);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialInverseCore<<<dim3(128, batch_size), 256>>>(
                        device_out, n2_root_of_unity_table, W_root_of_unity_table, modulus, 9, 8,
                        cfg.mod_inverse, 16);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 17:

                    FourStepInverseCoreT1<<<dim3(128, batch_size), dim3(32, 8)>>>(
                        device_in, device_out, n1_root_of_unity_table, modulus, 12, 32768, 17);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialInverseCore1<<<dim3(8, 32, batch_size), 256>>>(
                        device_out, n2_root_of_unity_table, W_root_of_unity_table, modulus, 12, 17);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialInverseCore2<<<dim3(8, 32, batch_size), dim3(64, 4)>>>(
                        device_out, n2_root_of_unity_table, modulus, 12, 9, 6, 2048, 9, 2,
                        cfg.mod_inverse, 17);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 18:

                    FourStepInverseCoreT1<<<dim3(256, batch_size), dim3(32, 8)>>>(
                        device_in, device_out, n1_root_of_unity_table, modulus, 13, 65536, 18);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialInverseCore1<<<dim3(16, 32, batch_size), 256>>>(
                        device_out, n2_root_of_unity_table, W_root_of_unity_table, modulus, 13, 18);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialInverseCore2<<<dim3(16, 32, batch_size), dim3(32, 8)>>>(
                        device_out, n2_root_of_unity_table, modulus, 13, 9, 5, 4096, 9, 3,
                        cfg.mod_inverse, 18);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 19:

                    FourStepInverseCoreT1<<<dim3(512, batch_size), dim3(32, 8)>>>(
                        device_in, device_out, n1_root_of_unity_table, modulus, 14, 131072, 19);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialInverseCore1<<<dim3(32, 32, batch_size), 256>>>(
                        device_out, n2_root_of_unity_table, W_root_of_unity_table, modulus, 14, 19);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialInverseCore2<<<dim3(32, 32, batch_size), dim3(16, 16)>>>(
                        device_out, n2_root_of_unity_table, modulus, 14, 9, 4, 8192, 9, 4,
                        cfg.mod_inverse, 19);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 20:

                    FourStepInverseCoreT1<<<dim3(1024, batch_size), dim3(32, 8)>>>(
                        device_in, device_out, n1_root_of_unity_table, modulus, 15, 262144, 20);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialInverseCore1<<<dim3(64, 32, batch_size), 256>>>(
                        device_out, n2_root_of_unity_table, W_root_of_unity_table, modulus, 15, 20);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialInverseCore2<<<dim3(64, 32, batch_size), dim3(8, 32)>>>(
                        device_out, n2_root_of_unity_table, modulus, 15, 9, 3, 16384, 9, 5,
                        cfg.mod_inverse, 20);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 21:

                    FourStepInverseCoreT2<<<dim3(2048, batch_size), dim3(32, 8)>>>(
                        device_in, device_out, n1_root_of_unity_table, modulus, 15, 16, 524288, 21);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialInverseCore1<<<dim3(64, 64, batch_size), 256>>>(
                        device_out, n2_root_of_unity_table, W_root_of_unity_table, modulus, 15, 21);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialInverseCore2<<<dim3(64, 64, batch_size), dim3(8, 32)>>>(
                        device_out, n2_root_of_unity_table, modulus, 15, 9, 3, 16384, 9, 5,
                        cfg.mod_inverse, 21);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 22:

                    FourStepInverseCoreT3<<<dim3(4096, batch_size), dim3(32, 8)>>>(
                        device_in, device_out, n1_root_of_unity_table, modulus, 15, 17, 1048576, 22);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialInverseCore1<<<dim3(64, 128, batch_size), 256>>>(
                        device_out, n2_root_of_unity_table, W_root_of_unity_table, modulus, 15, 22);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialInverseCore2<<<dim3(64, 128, batch_size), dim3(8, 32)>>>(
                        device_out, n2_root_of_unity_table, modulus, 15, 9, 3, 16384, 9, 5,
                        cfg.mod_inverse, 22);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 23:

                    FourStepInverseCoreT3<<<dim3(8192, batch_size), dim3(32, 8)>>>(
                        device_in, device_out, n1_root_of_unity_table, modulus, 16, 18, 2097152, 23);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialInverseCore1<<<dim3(128, 128, batch_size), 256>>>(
                        device_out, n2_root_of_unity_table, W_root_of_unity_table, modulus, 16, 23);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialInverseCore2<<<dim3(128, 128, batch_size), dim3(4, 64)>>>(
                        device_out, n2_root_of_unity_table, modulus, 16, 9, 2, 32768, 9, 6,
                        cfg.mod_inverse, 23);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 24:

                    FourStepInverseCoreT4<<<dim3(16384, batch_size), dim3(32, 8)>>>(
                        device_in, device_out, n1_root_of_unity_table, modulus, 16, 19, 4194304, 24);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialInverseCore1<<<dim3(128, 256, batch_size), 256>>>(
                        device_out, n2_root_of_unity_table, W_root_of_unity_table, modulus, 16, 24);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialInverseCore2<<<dim3(128, 256, batch_size), dim3(4, 64)>>>(
                        device_out, n2_root_of_unity_table, modulus, 16, 9, 2, 32768, 9, 6,
                        cfg.mod_inverse, 24);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;

                default:
                    std::cout << "This ring size is not supported!" << std::endl;
                    break;
            }

            break;

        default:
            break;
    }
}