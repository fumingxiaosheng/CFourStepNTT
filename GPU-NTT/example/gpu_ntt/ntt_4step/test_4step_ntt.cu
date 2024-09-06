#include <cstdlib>
#include <random>

#include "ntt.cuh"
#include "ntt_4step.cuh"
#include "ntt_4step_cpu.cuh"
//#include "hxw_constant.cuh"

#define DEFAULT_MODULUS

using namespace std;

int LOGN;
int BATCH;
int N;


extern __constant__ Root Csitable64[32];

int main(int argc, char* argv[])
{
    printf("test_4_step_ntt hxw12\n");
    
    CudaDevice();

    //根据传入的参数指定LOGN和BATCH
    if(argc < 3)
    {
        LOGN = 24;
        BATCH = 1;
    }
    else
    {
        LOGN = atoi(argv[1]);
        BATCH = atoi(argv[2]);
    }

//定义约减的方式
#ifdef BARRETT_64
    ModularReductionType modular_reduction_type = ModularReductionType::BARRET;
#elif defined(GOLDILOCKS_64)
    ModularReductionType modular_reduction_type = ModularReductionType::GOLDILOCK;
#elif defined(PLANTARD_64)
    ModularReductionType modular_reduction_type = ModularReductionType::PLANTARD;
#else
#error "Please define reduction type."
#endif

    // Current 4step NTT implementation only works for ReductionPolynomial::X_N_minus!
    NTTParameters4Step parameters(LOGN, modular_reduction_type, ReductionPolynomial::X_N_minus);
    
    // NTT generator with certain modulus and root of unity
    NTT_4STEP_CPU generator(parameters);

    std::random_device rd;
    std::mt19937 gen(rd());
    unsigned long long minNumber = 0;
    unsigned long long maxNumber = parameters.modulus.value - 1;
    std::uniform_int_distribution<unsigned long long> dis(minNumber, maxNumber);

    // Random data generation for polynomials
    //调用随机数生成器，生成相应的多项式
    vector<vector<Data>> input1(BATCH);
    for(int j = 0; j < BATCH; j++)
    {
        for(int i = 0; i < parameters.n; i++)
        {
            input1[j].push_back(dis(gen));
            //input1[j].push_back(i % 4096);
            //printf("%d,",input1[j][i]);
        }
        //printf("\n");
    }

    // Performing CPU NTT
    vector<vector<Data>> ntt_result(BATCH);
    for(int i = 0; i < BATCH; i++)
    {
        ntt_result[i] = generator.ntt(input1[i]);
    }
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////

    Data* Input_Datas;

    THROW_IF_CUDA_ERROR(cudaMalloc(&Input_Datas, BATCH * parameters.n * sizeof(Data)));

    Data* Output_Datas;
    THROW_IF_CUDA_ERROR(cudaMalloc(&Output_Datas, BATCH * parameters.n * sizeof(Data)));

    Data* Input_Datas1;

    THROW_IF_CUDA_ERROR(cudaMalloc(&Input_Datas1, BATCH * parameters.n * sizeof(Data)));

    Data* Output_Datas1;
    THROW_IF_CUDA_ERROR(cudaMalloc(&Output_Datas1, BATCH * parameters.n * sizeof(Data)));
    for(int j = 0; j < BATCH; j++)
    {
        //input1[j].data()代表指向vector的第一个数据的指针
        THROW_IF_CUDA_ERROR(cudaMemcpy(Input_Datas + (parameters.n * j), input1[j].data(),
                                       parameters.n * sizeof(Data), cudaMemcpyHostToDevice));

        THROW_IF_CUDA_ERROR(cudaMemcpy(Input_Datas1 + (parameters.n * j), input1[j].data(),
                                       parameters.n * sizeof(Data), cudaMemcpyHostToDevice));
    }

    //////////////////////////////////////////////////////////////////////////

    //传输相应的旋转因子表
    vector<Root_> psitable1 =
        parameters.gpu_root_of_unity_table_generator(parameters.n1_based_root_of_unity_table);//ROOT_是根据约减类型而定的数据类型，注意在param宏宏,root是data类型的
    Root* psitable_device1; //Root是根据具体的约减类型确定的,其和Root_是相同的
    THROW_IF_CUDA_ERROR(cudaMalloc(&psitable_device1, (parameters.n1 >> 1) * sizeof(Root)));
    THROW_IF_CUDA_ERROR(cudaMemcpy(psitable_device1, psitable1.data(),
                                   (parameters.n1 >> 1) * sizeof(Root), cudaMemcpyHostToDevice));

    vector<Root_> psitable2 =
        parameters.gpu_root_of_unity_table_generator(parameters.n2_based_root_of_unity_table);
    Root* psitable_device2;
    THROW_IF_CUDA_ERROR(cudaMalloc(&psitable_device2, (parameters.n2 >> 1) * sizeof(Root)));
    THROW_IF_CUDA_ERROR(cudaMemcpy(psitable_device2, psitable2.data(),
                                   (parameters.n2 >> 1) * sizeof(Root), cudaMemcpyHostToDevice));

    Root* W_Table_device;
    THROW_IF_CUDA_ERROR(cudaMalloc(&W_Table_device, parameters.n * sizeof(Root)));
    THROW_IF_CUDA_ERROR(cudaMemcpy(W_Table_device, parameters.W_root_of_unity_table.data(),
                                   parameters.n * sizeof(Root), cudaMemcpyHostToDevice));

    //新增加4096的传输
    vector<Root_> psitable64 =
        parameters.gpu_root_of_unity_table_generator(parameters.n64_root_of_unity_table);//ROOT_是根据约减类型而定的数据类型，注意在param宏宏,root是data类型的
    Root* psitable_device64; //Root是根据具体的约减类型确定的,其和Root_是相同的

    THROW_IF_CUDA_ERROR(cudaMalloc(&psitable_device64, (UNITY_SIZE1 >> 1) * sizeof(Root)));
    THROW_IF_CUDA_ERROR(cudaMemcpy(psitable_device64, psitable64.data(),
                                   (UNITY_SIZE1 >> 1) * sizeof(Root), cudaMemcpyHostToDevice));

    Root* n64_W_Table_device;
    THROW_IF_CUDA_ERROR(cudaMalloc(&n64_W_Table_device, UNITY_SIZE1 * UNITY_SIZE2 * sizeof(Root)));
    THROW_IF_CUDA_ERROR(cudaMemcpy(n64_W_Table_device, parameters.n64_W_root_of_unity_table.data(),
                                   UNITY_SIZE1 * UNITY_SIZE2 * sizeof(Root), cudaMemcpyHostToDevice));

    //////////////////////////////////////////////////////////////////////////

    
    THROW_IF_CUDA_ERROR(cudaMemcpyToSymbol(Csitable64, psitable64.data(), 32 * sizeof(Root))); //将64维的旋转因子拷贝到常量内存中

    //////////////////////////////////////////////////////////////////////////

    Modulus* test_modulus;//用于存放模数
    THROW_IF_CUDA_ERROR(cudaMalloc(&test_modulus, sizeof(Modulus)));

    Modulus test_modulus_[1] = {parameters.modulus};

    THROW_IF_CUDA_ERROR(
        cudaMemcpy(test_modulus, test_modulus_, sizeof(Modulus), cudaMemcpyHostToDevice));

    Ninverse* test_ninverse;//n^-1在Zq上的值,device变量指针
    THROW_IF_CUDA_ERROR(cudaMalloc(&test_ninverse, sizeof(Ninverse)));

    Ninverse test_ninverse_[1] = {parameters.n_inv};

    THROW_IF_CUDA_ERROR(
        cudaMemcpy(test_ninverse, test_ninverse_, sizeof(Ninverse), cudaMemcpyHostToDevice));

    ntt4step_rns_configuration cfg_intt = {.n_power = LOGN,
                                      .ntt_type = FORWARD,
                                      .mod_inverse = test_ninverse,
                                      .stream = 0};

    //////////////////////////////////////////////////////////////////////////
    GPU_Transpose(Input_Datas, Output_Datas, parameters.n1, parameters.n2, parameters.logn, BATCH);//d,d,h,h,h,h Input_Datas是n1*n2维,Output_Datas是n2*n1维度 //在12的第一版中是不需要进行转置的


    GPU_4STEP_NTT(Output_Datas, Input_Datas, psitable_device1, psitable_device2, W_Table_device, test_modulus, cfg_intt, BATCH, 1);

    //根据自己定义的划分进行求值
    if(parameters.n == (1 << 12)){
        printf("n=4096\n");
        parameters.n1 = 64;
        parameters.n2 = 64;
    }

    if(parameters.n == (1 << 16)){
        printf("n=2^{16}\n");
        parameters.n1 = 256;
        parameters.n2 = 256;
    }
    
    GPU_Transpose(Input_Datas1, Output_Datas1, parameters.n1, parameters.n2, parameters.logn, BATCH);//d,d,h,h,h,h Input_Datas是n1*n2维,Output_Datas是n2*n1维度

    GPU_4STEP_NTT_hxw(Output_Datas1, Input_Datas1, psitable_device1, psitable_device2, W_Table_device, psitable_device64, n64_W_Table_device ,test_modulus, cfg_intt, BATCH, 1);
    //GPU_4STEP_NTT_hxw(Output_Datas1, Input_Datas1, psitable_device1, psitable_device2, W_Table_device, Csitable64, n64_W_Table_device ,test_modulus, cfg_intt, BATCH, 1);

    //GPU_Transpose(Input_Datas, Output_Datas, parameters.n1, parameters.n2, parameters.logn, BATCH);

    vector<Data> Output_Host(parameters.n * BATCH);
    cudaMemcpy(Output_Host.data(), Input_Datas, parameters.n * BATCH * sizeof(Data),
               cudaMemcpyDeviceToHost);

    vector<Data> Output_Host1(parameters.n * BATCH);
    cudaMemcpy(Output_Host1.data(), Input_Datas1, parameters.n * BATCH * sizeof(Data),
               cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    // Comparing GPU NTT results and CPU NTT results
    //读出相应的数
    bool check = true;
    /*printf("main look\n");
    for(int i=0;i<BATCH;i++){
        for(int j=0;j<parameters.n;j++){
            printf("%lld,",Output_Host1[(i * parameters.n) + j]);
            if((j + 1)% 64 == 0) printf("\n");
            if((j + 1)% parameters.n2 == 0) printf("\n");
        }
        printf("\n\n");
    }*/

    for(int i = 0; i < BATCH; i++)
    {
        check = check_result(Output_Host.data() + (i * parameters.n), ntt_result[i].data(),
                             parameters.n);

        if(!check)
        {
            cout << "(in " << i << ". Poly.)" << endl;
            break;
        }

        if((i == (BATCH - 1)) && check)
        {
            cout << "origaninal All Correct." << endl;
        }
    }

    for(int i = 0; i < BATCH; i++)
    {
        check = check_result(Output_Host1.data() + (i * parameters.n), ntt_result[i].data(),
                             parameters.n);

        if(!check)
        {
            cout << "(in " << i << ". Poly.)" << endl;
            break;
        }

        if((i == (BATCH - 1)) && check)
        {
            cout << "hxw All Correct." << endl;
        }
    }

    return EXIT_SUCCESS;
}