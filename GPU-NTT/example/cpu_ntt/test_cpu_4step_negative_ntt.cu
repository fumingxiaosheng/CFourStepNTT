#include <cstdlib>  // For atoi or atof functions
#include <fstream>
#include <random>

#include "ntt.cuh"
#include "ntt_4step_cpu.cuh"

#define DEFAULT_MODULUS

using namespace std;

int LOGN;
int BATCH;
int N;

int main(int argc, char* argv[])
{
    printf("hxw negative ntt test 2\n");
    CudaDevice();

    if(argc < 3)
    {
        LOGN = 12;
        BATCH = 1;
    }
    else
    {
        LOGN = atoi(argv[1]);
        BATCH = atoi(argv[2]);
    }

#ifdef BARRETT_64
    ModularReductionType modular_reduction_type = ModularReductionType::BARRET;
#elif defined(GOLDILOCKS_64)
    ModularReductionType modular_reduction_type = ModularReductionType::GOLDILOCK;
#elif defined(PLANTARD_64)
    ModularReductionType modular_reduction_type = ModularReductionType::PLANTARD;
#else
#error "Please define reduction type."
#endif

    NTTParameters4Step parameters(LOGN, modular_reduction_type, ReductionPolynomial::X_N_minus);

    printf("moduls %lld\n",parameters.modulus);

    printf("[out]n1:%d ,n2:%d, n1 root num:%d, n2 root num:%d\n\n",parameters.n1,parameters.n2,parameters.n1_based_inverse_root_of_unity_table.size(),parameters.n2_based_root_of_unity_table.size());
    /*for(int i=0;i<parameters.n1_based_root_of_unity_table.size();i++){
        printf("%d,",parameters.n1_based_root_of_unity_table[i]);
    }
    printf("\n");*/


    // NTT generator with certain modulus and root of unity
    NTT_4STEP_CPU generator(parameters);

    std::random_device rd;
    std::mt19937 gen(rd());
    unsigned long long minNumber = 0;
    unsigned long long maxNumber = parameters.modulus.value - 1;
    std::uniform_int_distribution<unsigned long long> dis(minNumber, maxNumber);

    // Random data generation for polynomials
    vector<Data> input1;
    vector<Data> input2;
    for(int j = 0; j < BATCH; j++)
    {
        for(int i = 0; i < parameters.n; i++)
        {
            input1.push_back(dis(gen));
            input2.push_back(dis(gen));
        }
    }

    // Performing CPU NTT
    
    vector<Data> ntt_input1 = generator.negative_ntt(input1);

    vector<Data> input1_hat = generator.negative_intt(ntt_input1);

    //printf("%d %d\n",input1.size(),input1_hat.size());
    printf("负折叠卷积NTT的计算结果%d\n",check_result(input1.data(), input1_hat.data(), parameters.n));
    //return 0;
    //printf("ntt_input1 = generator.ntt(input1); done\n");
    vector<Data> ntt_input2 = generator.negative_ntt(input2);
    //printf("ntt_input2 = generator.ntt(input2); done\n");

    vector<Data> input2_hat = generator.negative_intt(ntt_input2);
    printf("负折叠卷积NTT的计算结果%d\n",check_result(input2.data(), input2_hat.data(), parameters.n));

    //printf("%d %d\n",ntt_input1.size(),ntt_input2.size());
    vector<Data> output = generator.mult(ntt_input1, ntt_input2);
    //printf("output = generator.mult(ntt_input1, ntt_input2)\n");
    vector<Data> ntt_mult_result = generator.negative_intt(output);
    //printf("ntt_mult_result = generator.intt(output);\n");

    //vector<Data> ntt3 = generator.intt(ntt_input2);

    // Comparing CPU NTT multiplication results and schoolbook multiplication
    // results
    bool check = true;
    std::vector<Data> schoolbook_result = schoolbook_poly_multiplication(
        input1, input2, parameters.modulus, ReductionPolynomial::X_N_plus);

    check = check_result(ntt_mult_result.data(), schoolbook_result.data(), parameters.n);

    if(check)
    {
        cout << "All Correct." << endl;
    }

    return EXIT_SUCCESS;
}

/*

cmake . -B./cmake-build
cmake . -D MODULAR_REDUCTION_TYPE=0 -B./cmake-build

cmake --build ./cmake-build/ --parallel

./cmake-build/bin/cpu_ntt_examples

*/

//