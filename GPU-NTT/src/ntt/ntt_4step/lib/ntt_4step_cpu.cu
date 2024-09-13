// --------------------- //
// Authors: Alisah Ozcan
// --------------------- //

#include "ntt_4step_cpu.cuh"

NTT_4STEP_CPU::NTT_4STEP_CPU(NTTParameters4Step parameters_) { parameters = parameters_; }

std::vector<Data> NTT_4STEP_CPU::mult(std::vector<Data>& input1, std::vector<Data>& input2)
{
    std::vector<Data> output;
    for(int i = 0; i < parameters.n; i++)
    {
        output.push_back(VALUE::mult(input1[i], input2[i], parameters.modulus));
    }

    return output;
}

/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////

/*2024-8-4*/
std::vector<Data> NTT_4STEP_CPU::ntt(std::vector<Data>& input)
{   
    //printf("[hxw debug] in NTT_4STEP_CPU::ntt\n");
    /*for(int i=0;i<input.size();i++){
        printf("%d,",input[i]);
    }
    printf("\n");*/
    std::vector<std::vector<Data>> matrix = vector_to_matrix(input, parameters.n1, parameters.n2); //step1 转变为n1*n2的矩阵
    std::vector<std::vector<Data>> transposed_matrix = transpose_matrix(matrix); //转置NTT操作

    /*printf("n1:%d ,n2:%d, n1 root num:%d, n2 root num:%d\n\n",parameters.n1,parameters.n2,parameters.n1_based_inverse_root_of_unity_table.size(),parameters.n2_based_root_of_unity_table.size());
    for(int i=0;i<parameters.n1_based_root_of_unity_table.size();i++){
        printf("%d,",parameters.n1_based_root_of_unity_table[i]);
    }
    printf("\n");*/

    for(int i = 0; i < parameters.n2; i++)
    {
        core_ntt(transposed_matrix[i], parameters.n1_based_root_of_unity_table,
                 int(log2(parameters.n1))); //最后计算的顺序是比特翻转的顺序
    }

    std::vector<std::vector<Data>> transposed_matrix2 = transpose_matrix(transposed_matrix);
    std::vector<Data> vector_ = matrix_to_vector(transposed_matrix2);

    product(vector_, parameters.W_root_of_unity_table, parameters.logn);

    std::vector<std::vector<Data>> matrix3 =
        vector_to_matrix(vector_, parameters.n1, parameters.n2);

    for(int i = 0; i < parameters.n1; i++)
    {
        core_ntt(matrix3[i], parameters.n2_based_root_of_unity_table,
                 int(log2(parameters.n2)));
    }

    //transposed_matrix2 = transpose_matrix(matrix3);
    std::vector<Data> result = matrix_to_vector(matrix3);

    return result;
}

/*2024-8-4:*/
std::vector<Data> NTT_4STEP_CPU::intt(std::vector<Data>& input)
{
    /*std::vector<std::vector<Data>> transposed_matrix =
        vector_to_matrix_intt(input, parameters.n1, parameters.n2);

    for(int i = 0; i < parameters.n2; i++)
    {
        core_intt(transposed_matrix[i], parameters.n1_based_inverse_root_of_unity_table,
                  int(log2(parameters.n1))); //n2个n1维的NTT
    }

    std::vector<std::vector<Data>> transposed_matrix2 = transpose_matrix(transposed_matrix); //N1*N2矩阵
    std::vector<Data> vector_ = matrix_to_vector(transposed_matrix2);

    product(vector_, parameters.W_inverse_root_of_unity_table, parameters.logn);

    std::vector<std::vector<Data>> transposed_matrix3 =
        vector_to_matrix(vector_, parameters.n1, parameters.n2); //n1*n2矩阵

    for(int i = 0; i < parameters.n1; i++)
    {
        core_intt(transposed_matrix3[i], parameters.n2_based_inverse_root_of_unity_table,
                  int(log2(parameters.n2))); //n1个n2维NTT
    }

    transposed_matrix2 = transpose_matrix(transposed_matrix3);//n2*n1矩阵

    std::vector<Data> result = matrix_to_vector(transposed_matrix2);

    for(int i = 0; i < parameters.n; i++)
    {
        result[i] = VALUE::mult(result[i], parameters.n_inv, parameters.modulus);
    }

    return result;*/
    //向量转变为矩阵
    std::vector<std::vector<Data>> matrix = vector_to_matrix(input, parameters.n1, parameters.n2);

    for(int i = 0; i < parameters.n1; i++)
    {
        core_intt(matrix[i], parameters.n2_based_inverse_root_of_unity_table,
                  int(log2(parameters.n2))); //n1个n2维NTT
    }

    std::vector<Data> vector_ = matrix_to_vector(matrix);

    product(vector_, parameters.W_inverse_root_of_unity_table, parameters.logn);

    std::vector<std::vector<Data>> matrix1 = vector_to_matrix(vector_, parameters.n1, parameters.n2); //n1*n2矩阵

    std::vector<std::vector<Data>> transposed_matrix = transpose_matrix(matrix1);

    for(int i = 0; i < parameters.n2; i++)
    {
        core_intt(transposed_matrix[i], parameters.n1_based_inverse_root_of_unity_table,
                  int(log2(parameters.n1))); //n2个n1维的NTT
    }

    matrix1 = transpose_matrix(transposed_matrix);

    std::vector<Data> result = matrix_to_vector(matrix1);

    for(int i = 0; i < parameters.n; i++)
    {
        result[i] = VALUE::mult(result[i], parameters.n_inv, parameters.modulus);
    }

    return result;
}

std::vector<Data> NTT_4STEP_CPU::negative_ntt(std::vector<Data>& input)
{   
    
    std::vector<std::vector<Data>> matrix = vector_to_matrix(input, parameters.n1, parameters.n2); //step1 转变为n1*n2的矩阵
    std::vector<std::vector<Data>> transposed_matrix = transpose_matrix(matrix); //转置NTT操作


    for(int i = 0; i < parameters.n2; i++)
    {
        negative_core_ntt(transposed_matrix[i], parameters.negative_2n1_based_root_of_unity_table,
                 int(log2(parameters.n1))); 
    }

    std::vector<std::vector<Data>> transposed_matrix2 = transpose_matrix(transposed_matrix);
    std::vector<Data> vector_ = matrix_to_vector(transposed_matrix2);
    
    product(vector_, parameters.negative_W_root_of_unity_table, parameters.logn);

    std::vector<std::vector<Data>> matrix3 =
        vector_to_matrix(vector_, parameters.n1, parameters.n2);

    for(int i = 0; i < parameters.n1; i++)
    {
        core_ntt(matrix3[i], parameters.negative_n2_based_root_of_unity_table,
                 int(log2(parameters.n2)));
    }

    //transposed_matrix2 = transpose_matrix(matrix3);
    std::vector<Data> result = matrix_to_vector(matrix3);

    return result;
}

/*2024-9-8:*/
std::vector<Data> NTT_4STEP_CPU::negative_intt(std::vector<Data>& input)
{

    //向量转变为矩阵
    std::vector<std::vector<Data>> matrix = vector_to_matrix(input, parameters.n1, parameters.n2);

    for(int i = 0; i < parameters.n1; i++)
    {
        core_intt(matrix[i], parameters.negative_n2_based_inverse_root_of_unity_table,
                  int(log2(parameters.n2))); //n1个n2维NTT
    }

    std::vector<Data> vector_ = matrix_to_vector(matrix);

    product(vector_, parameters.negative_W_inverse_root_of_unity_table, parameters.logn);

    std::vector<std::vector<Data>> matrix1 = vector_to_matrix(vector_, parameters.n1, parameters.n2); //n1*n2矩阵

    std::vector<std::vector<Data>> transposed_matrix = transpose_matrix(matrix1);

    for(int i = 0; i < parameters.n2; i++)
    {
        negative_core_intt(transposed_matrix[i], parameters.negative_2n1_based_inverse_root_of_unity_table,
                  int(log2(parameters.n1))); //n2个n1维的NTT
    }

    matrix1 = transpose_matrix(transposed_matrix);

    std::vector<Data> result = matrix_to_vector(matrix1);

    for(int i = 0; i < parameters.n; i++)
    {
        result[i] = VALUE::mult(result[i], parameters.n_inv, parameters.modulus);
    }

    return result;
}
/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////
/*2024-8-1:
root_table:代表的是n次本原单位根的0到n/2次的值
使用经典迭代算法,得到循环卷积NTT的比特翻转结果*/
void NTT_4STEP_CPU::core_ntt(std::vector<Data>& input, std::vector<Data> root_table, int log_size)
{
    // Merged NTT with pre-processing (optimized) (iterative)
    // This is not NTT, this is pre-processing + NTT
    // (see: https://eprint.iacr.org/2016/504.pdf)

    int n_ = 1 << log_size;
    int t = n_; //组长度
    int m = 1; //组数

    while(m < n_)
    {
        t = t >> 1; //步长

        for(int i = 0; i < m; i++)
        {
            int j1 = 2 * i * t; //2*t是组的长度,j1是每个组开始的位置
            int j2 = j1 + t - 1; //j2是每个组下半部分开始的位置

            int index = bitreverse(i, log_size - 1); //m的长度为log_size-1 

            Data S = root_table[index];

            for(int j = j1; j < (j2 + 1); j++)
            {
                Data U = input[j];
                Data V = VALUE::mult(input[j + t], S, parameters.modulus);

                input[j] = VALUE::add(U, V, parameters.modulus);
                input[j + t] = VALUE::sub(U, V, parameters.modulus);
            }
        }

        m = m << 1;
    }
}

void NTT_4STEP_CPU::core_intt(std::vector<Data>& input, std::vector<Data> root_table, int log_size)
{
    // Merged INTT with post-processing (optimized) (iterative)
    // This is not NTT, this is pre-processing + NTT
    // (see: https://eprint.iacr.org/2016/504.pdf)

    int n_ = 1 << log_size;
    int t = 1; //间隔
    int m = n_; //组数乘以2

    while(m > 1)
    {
        int j1 = 0; //第一个元素的位置
        int h = m >> 1; //h代表组数
        for(int i = 0; i < h; i++) //i代表的是第几组
        {
            int j2 = j1 + t - 1; //每组的上半部分的元素下标

            int index = bitreverse(i, log_size - 1);

            Data S = root_table[index];

            for(int j = j1; j < (j2 + 1); j++)
            {
                Data U = input[j];
                Data V = input[j + t];

                input[j] = VALUE::add(U, V, parameters.modulus);
                input[j + t] = VALUE::sub(U, V, parameters.modulus);
                input[j + t] = VALUE::mult(input[j + t], S, parameters.modulus);
            }

            j1 = j1 + (t << 1);
        }

        t = t << 1;
        m = m >> 1;
    }
}

/*2024-9-8:
负循环卷积NTT的正向实现
*/
void NTT_4STEP_CPU::negative_core_ntt(std::vector<Data>& input, std::vector<Data> root_table, int log_size)
{
    // Merged NTT with pre-processing (optimized) (iterative)
    // This is not NTT, this is pre-processing + NTT
    // (see: https://eprint.iacr.org/2016/504.pdf)

    int n_ = 1 << log_size;
    int t = n_; //组长度
    int m = 1; //组数
    while(m < n_)
    {
        t = t >> 1; //步长

        for(int i = 0; i < m; i++)
        {
            int j1 = 2 * i * t; //2*t是组的长度,j1是每个组开始的位置
            int j2 = j1 + t - 1; //j2是每个组下半部分开始的位置

            int index = bitreverse(m + i, log_size); //m的长度为log_size-1 

            Data S = root_table[index];

            for(int j = j1; j < (j2 + 1); j++)
            {
                Data U = input[j];
                Data V = VALUE::mult(input[j + t], S, parameters.modulus);

                input[j] = VALUE::add(U, V, parameters.modulus);
                input[j + t] = VALUE::sub(U, V, parameters.modulus);
            }
        }
        m = m << 1;
    }
}

/*2024-9-8:
负循环卷积NTT的逆向实现*/
void NTT_4STEP_CPU::negative_core_intt(std::vector<Data>& input, std::vector<Data> root_table, int log_size)
{
    // Merged INTT with post-processing (optimized) (iterative)
    // This is not NTT, this is pre-processing + NTT
    // (see: https://eprint.iacr.org/2016/504.pdf)

    int n_ = 1 << log_size;
    int t = 1; //间隔
    int m = n_; //组数乘以2

    while(m > 1)
    {
        int j1 = 0; //第一个元素的位置
        int h = m >> 1; //h代表组数
        for(int i = 0; i < h; i++) //i代表的是第几组
        {
            int j2 = j1 + t - 1; //每组的上半部分的元素下标

            int index = bitreverse(h + i, log_size);

            Data S = root_table[index];

            for(int j = j1; j < (j2 + 1); j++)
            {
                Data U = input[j];
                Data V = input[j + t];

                input[j] = VALUE::add(U, V, parameters.modulus);
                input[j + t] = VALUE::sub(U, V, parameters.modulus);
                input[j + t] = VALUE::mult(input[j + t], S, parameters.modulus);
            }

            j1 = j1 + (t << 1);
        }

        t = t << 1;
        m = m >> 1;
    }
}

/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////

void NTT_4STEP_CPU::product(std::vector<Data>& input, std::vector<Data> root_table, int log_size)
{
    int n_ = 1 << log_size;
    for(int i = 0; i < n_; i++)
    {
        input[i] = VALUE::mult(input[i], root_table[i], parameters.modulus);
    }
}

/*按行展开array*/
std::vector<std::vector<Data>> NTT_4STEP_CPU::vector_to_matrix(const std::vector<Data>& array,
                                                               int rows, int cols)
{
    std::vector<std::vector<Data>> matrix(rows, std::vector<Data>(cols));

    for(int i = 0; i < rows; ++i)
    {
        for(int j = 0; j < cols; ++j)
        {
            matrix[i][j] = array[(i * cols) + j];
        }
    }

    return matrix;
}

/*2024-8-2:
返回n2*n1的矩阵*/
std::vector<std::vector<Data>> NTT_4STEP_CPU::vector_to_matrix_intt(const std::vector<Data>& array,
                                                                    int rows, int cols)
{
    std::vector<std::vector<Data>> matrix(cols);

    for(int i = 0; i < rows; ++i)
    {
        for(int j = 0; j < cols; ++j)
        {
            matrix[int(((i * cols) + j) / rows)].push_back(array[i + (j * rows)]);
        }
    }

    std::vector<std::vector<Data>> matrix2(rows, std::vector<Data>(cols));
    std::vector<std::vector<Data>> matrix3(cols);
    for(int i = 0; i < rows; ++i)
    {
        for(int j = 0; j < cols; ++j)
        {
            matrix2[i][j] = array[j * rows + i];
        }
    }

    for(int i = 0; i < rows; ++i)
    {
        for(int j = 0; j < cols; ++j)
        {
            matrix3[int(((i * cols) + j) / rows)].push_back(matrix2[i][j]) ;
        }
    }
    printf("vector_to_matrix_intt done %d %d %d %d\n",matrix.size(),matrix[0].size(),matrix2.size(),matrix2[0].size());
    for(int i=0;i<matrix.size();i++){
        for(int j=0;j<matrix[0].size();j++){
            if(matrix[i][j] != matrix3[i][j]){
                printf("wrong\n");
                return matrix;
            }
        }
    }
    return matrix3;
    //return matrix;
}

/*2024-8-1：
按照行主序转化为向量*/
std::vector<Data> NTT_4STEP_CPU::matrix_to_vector(
    const std::vector<std::vector<Data>>& originalMatrix)
{
    int rows = originalMatrix.size();
    int cols = originalMatrix[0].size();

    std::vector<Data> result;

    for(int i = 0; i < rows; ++i)
    {
        for(int j = 0; j < cols; ++j)
        {
            result.push_back(originalMatrix[i][j]);
        }
    }

    return result;
}

/*2024-8-1;
完成普通的矩阵转置操作*/
std::vector<std::vector<Data>> NTT_4STEP_CPU::transpose_matrix(
    const std::vector<std::vector<Data>>& originalMatrix)
{
    int rows = originalMatrix.size();
    int cols = originalMatrix[0].size();

    std::vector<std::vector<Data>> transpose(cols, std::vector<Data>(rows));

    for(int i = 0; i < rows; ++i)
    {
        for(int j = 0; j < cols; ++j)
        {
            transpose[j][i] = originalMatrix[i][j];
        }
    }

    return transpose;
}

std::vector<Data> NTT_4STEP_CPU::intt_first_transpose(const std::vector<Data>& input)
{
    std::vector<std::vector<Data>> transposed_matrix =
        vector_to_matrix_intt(input, parameters.n1, parameters.n2);

    std::vector<Data> result = matrix_to_vector(transposed_matrix);

    return result;
}