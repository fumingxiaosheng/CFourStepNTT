// --------------------- //
// Authors: Alisah Ozcan
// --------------------- //

#include "nttparameters.cuh"

/*2024-8-1:
将index视为n_power长度的数字，对其进行比特翻转
处理流程:每次取出indx的最后一个bit,然后把他放到res_1中*/
int bitreverse(int index, int n_power)
{
    int res_1 = 0;
    for (int i = 0; i < n_power; i++)
    {
        res_1 <<= 1;
        res_1 = (index & 1) | res_1;
        index >>= 1; //每次取出最后一个比特
    }
    return res_1;
}

NTTParameters::NTTParameters(int LOGN,
                             ModularReductionType modular_reduction_type,
                             ReductionPolynomial poly_reduce_type)
{
    logn = LOGN;
    n = 1 << logn;

    poly_reduction = poly_reduce_type;
    modular_reduction = modular_reduction_type;

    modulus = modulus_pool();

#ifdef PLANTARD_64
    R = R_pool();  // for Plantard Reduction
#endif

    omega = omega_pool();
    psi = psi_pool();

    // n_inv = VALUE::modinv(n, modulus);

    root_of_unity =
        (poly_reduce_type == ReductionPolynomial::X_N_minus) ? omega : psi;
    inverse_root_of_unity = VALUE::modinv(root_of_unity, modulus);

    root_of_unity_size = (poly_reduce_type == ReductionPolynomial::X_N_minus)
                             ? (1 << (logn - 1))
                             : (1 << logn);

    forward_root_of_unity_table_generator();
    inverse_root_of_unity_table_generator();

    n_inverse_generator();
}

NTTParameters::NTTParameters(int LOGN, NTTFactors ntt_factors,
                             ReductionPolynomial poly_reduce_type)
{
    logn = LOGN;
    n = 1 << logn;

    poly_reduction = poly_reduce_type;

    modulus = ntt_factors.modulus;

    omega = ntt_factors.omega;
    psi = ntt_factors.psi;

    n_inverse_generator();

    root_of_unity =
        (poly_reduce_type == ReductionPolynomial::X_N_minus) ? omega : psi;
    inverse_root_of_unity = VALUE::modinv(root_of_unity, modulus);

    root_of_unity_size = (poly_reduce_type == ReductionPolynomial::X_N_minus)
                             ? (1 << (logn - 1))
                             : (1 << logn);

    forward_root_of_unity_table_generator();
    inverse_root_of_unity_table_generator();

    n_inverse_generator();
}

NTTParameters::NTTParameters() {}

Modulus NTTParameters::modulus_pool()
{
    customAssert(12 <= logn <= 24, "LOGN should be in range 12 to 24.");
    if ((modular_reduction == ModularReductionType::BARRET) ||
        (modular_reduction == ModularReductionType::PLANTARD))
    {
        static Data primes[] = {
            576460752303415297, 576460752303439873, 576460752304439297,
            576460752308273153, 576460752308273153, 576460752315482113,
            576460752315482113, 576460752340123649, 576460752364240897,
            576460752475389953, 576460752597024769, 576460753024843777,
            576460753175838721, 288230377292562433, 288230383802122241,
            288230385815388161, 288230385815388161};

        Modulus prime(primes[logn - 12]);
        return prime;
    }
    else if ((modular_reduction == ModularReductionType::GOLDILOCK))
    {
        static Data primes[] = {
            18446744069414584321, 18446744069414584321, 18446744069414584321,
            18446744069414584321, 18446744069414584321, 18446744069414584321,
            18446744069414584321, 18446744069414584321, 18446744069414584321,
            18446744069414584321, 18446744069414584321, 18446744069414584321,
            18446744069414584321, 18446744069414584321, 18446744069414584321,
            18446744069414584321, 18446744069414584321};

        Modulus prime(primes[logn - 12]);
        return prime;
    }
    else
    {
        throw std::runtime_error("Reduction type is not supported!");
    }
}

Data NTTParameters::omega_pool()
{
    customAssert(12 <= logn <= 24, "LOGN should be in range 12 to 24.");
    if ((modular_reduction == ModularReductionType::BARRET) ||
        (modular_reduction == ModularReductionType::PLANTARD))
    {
        static Data W[] = {
            288482366111684746, 37048445140799662,  459782973201979845,
            64800917766465203,  425015386842055933, 18734847765732801,
            119109113519742895, 227584740857897520, 477282059544659462,
            570131728462077067, 433594414095420776, 219263994987749328,
            189790554094222112, 96649110792683523, 250648942594717784,
            279172744045218282, 225865349704673648};

        return W[logn - 12];
    }
    else if ((modular_reduction == ModularReductionType::GOLDILOCK))
    {
        static Data W[] = {
            10246984546936836946, 13835541155533501494, 4150195341322954475,
            967374280115824382,   6097901191693030237,  11209893866417677111,
            3132005779464972240,  5277653236910267777,  14100038135461308321,
            6945674003198401120,  8888549356274591460,  17066151606537038542,
            4420008141159053259, 9092277472170284592, 2759361949145500591,
            14068871798966024971, 13445489538489244109};

        return W[logn - 12];
    }
    else
    {
        throw std::runtime_error("Reduction type is not supported!");
    }
}

Data NTTParameters::psi_pool()
{
    customAssert(12 <= logn <= 24, "LOGN should be in range 12 to 24.");
    if ((modular_reduction == ModularReductionType::BARRET) ||
        (modular_reduction == ModularReductionType::PLANTARD))
    {
        static unsigned long long PSI[] = {
            238394956950829, 54612008597396, 8242615629351, 16141297350887,
            3760097055997,   11571974431275, 328867687796,  2298846063117,
            731868219707,    409596963254,   189266227206,  31864818375,
            92067739764, 5214432335, 734084005, 3351406780, 717004697};

        return PSI[logn - 12];
    }
    else if ((modular_reduction == ModularReductionType::GOLDILOCK))
    {
        static unsigned long long PSI[] = {
            3864827780506026, 5467143286794452, 563364698871119,
            633449661190857,  352105042511453,  309187323659538,
            21950690503823,   19333215661852,   3041397215123,
            22687970875277,   4798578965421,    204312508762,
            6369339824961, 36819894005, 31311927402,
            217045171522, 32062695042};

        return PSI[logn - 12];
    }
    else
    {
        throw std::runtime_error("Reduction type is not supported!");
    }
}

void NTTParameters::forward_root_of_unity_table_generator()
{
    forward_root_of_unity_table.push_back(1);

    for (int i = 1; i < root_of_unity_size; i++)
    {
        Data exp = VALUE::mult(forward_root_of_unity_table[i - 1],
                               root_of_unity, modulus);
        forward_root_of_unity_table.push_back(exp);
    }
}

void NTTParameters::inverse_root_of_unity_table_generator()
{
    inverse_root_of_unity_table.push_back(1);

    for (int i = 1; i < root_of_unity_size; i++)
    {
        Data exp = VALUE::mult(inverse_root_of_unity_table[i - 1],
                               inverse_root_of_unity, modulus);
        inverse_root_of_unity_table.push_back(exp);
    }
}

void NTTParameters::n_inverse_generator()
{
#if defined(BARRETT_64) || defined(GOLDILOCKS_64)
    n_inv = VALUE::modinv(n, modulus);
#elif defined(PLANTARD_64)
    Data n_inv_ = VALUE::modinv(n, modulus);
    Data base_ = 2;
    Data expo_ = 128;
    Data correction = modulus.value - VALUE::exp(base_, expo_, modulus);

    Data mult = VALUE::mult(n_inv_, correction, modulus);

    __uint128_t mult_ = (__uint128_t)mult * (__uint128_t)R;

    n_inv.x = (mult_ & UINT64_MAX);
    n_inv.y = (mult_ >> 64u);
#else
#error "Please define reduction type."
#endif
}

#ifdef PLANTARD_64
__uint128_t NTTParameters::R_pool()
{
    customAssert(12 <= logn <= 24, "LOGN should be in range 12 to 24.");

    if ((modular_reduction == ModularReductionType::BARRET) ||
        (modular_reduction == ModularReductionType::GOLDILOCK))
    {
        std::runtime_error("Only Plantard reduction can be used!");
    }
    else if ((modular_reduction == ModularReductionType::PLANTARD))
    {
        static __uint128_t R_lo[] = {
            17874787470856429569, 17942336517665964033, 17975028228979458049,
            14490355320300634113, 14490355320300634113, 16861622415275851777,
            16861622415275851777, 5189493672438136833,  8650610041606373377,
            17899855785974824961, 17956485032730165249, 18390730552622710785,
            184647583849775105};

        static __uint128_t R_hi[] = {
            11555533987311009281, 13842095462251758527, 9154506088165204343,
            18092274698773475081, 18092274698773475081, 5085365105665609632,
            5085365105665609632,  15936220441903101320, 2182956421460119645,
            10044172439809674011, 11784997034778628255, 4955880097643588863,
            18424807717242322944};

        __uint128_t R = ((__uint128_t)R_hi[logn - 12] << (__uint128_t)64) +
                        ((__uint128_t)R_lo[logn - 12]);

        return R;
    }
    else
    {
        throw std::runtime_error("Reduction type is not supported!");
    }
}
#endif

std::vector<Root_> NTTParameters::gpu_root_of_unity_table_generator(
    std::vector<Data> table)
{
    // Taking Bitreverse order of root of unity table

    std::vector<Data> new_table;
    int lg = log2(root_of_unity_size);
    for (int i = 0; i < root_of_unity_size; i++)
    {
        new_table.push_back(table[bitreverse(i, lg)]);
    }

#if defined(BARRETT_64) || defined(GOLDILOCKS_64)
    return new_table;
#elif defined(PLANTARD_64)
    Data base_ = 2;
    Data expo_ = 128;
    Data correction = modulus.value - VALUE::exp(base_, expo_, modulus);

    std::vector<__uint128_t> plantard_table;
    for (int i = 0; i < root_of_unity_size; i++)
    {
        Data temp = VALUE::mult(new_table[i], correction, modulus);

        __uint128_t mult_ = (__uint128_t)temp * (__uint128_t)R;

        plantard_table.push_back(mult_);
    }
    return plantard_table;
#else
#error "Please define reduction type."
#endif
}















































NTTParameters4Step::NTTParameters4Step(int LOGN,
                             ModularReductionType modular_reduction_type,
                             ReductionPolynomial poly_reduce_type)
{

    logn = LOGN;
    n = 1 << logn;

    poly_reduction = poly_reduce_type;
    modular_reduction = modular_reduction_type;

    modulus = modulus_pool();
    
#ifdef PLANTARD_64
    R = R_pool();  // for Plantard Reduction
#endif

    omega = omega_pool();
    psi = psi_pool();

    //n_inv = VALUE::modinv(n, modulus);

    root_of_unity =
        (poly_reduce_type == ReductionPolynomial::X_N_minus) ? omega : psi;
    inverse_root_of_unity = VALUE::modinv(root_of_unity, modulus); //本原单位根

    root_of_unity_size = (poly_reduce_type == ReductionPolynomial::X_N_minus)
                             ? (1 << (logn - 1))
                             : (1 << logn);

    std::vector<int> dimention = matrix_dimention();
    n1 = dimention[0];
    n2 = dimention[1];

    small_forward_root_of_unity_table_generator();
    small_inverse_root_of_unity_table_generator();

    n64_root_of_unity_table_generator(); //HXW add

    TW_forward_table_generator();
    TW_inverse_table_generator();

    n64_W_root_of_unity_table_generator(); //HXW add

    n_inverse_generator();

}

NTTParameters4Step::NTTParameters4Step() {}



Modulus NTTParameters4Step::modulus_pool()
{
    customAssert(12 <= logn <= 24, "LOGN should be in range 12 to 24.");
    if ((modular_reduction == ModularReductionType::BARRET) ||
        (modular_reduction == ModularReductionType::PLANTARD))
    {
        static Data primes[] = {
            576460752303415297, 576460752303439873, 576460752304439297,
            576460752308273153, 576460752308273153, 576460752315482113,
            576460752315482113, 576460752340123649, 576460752364240897,
            576460752475389953, 576460752597024769, 576460753024843777,
            576460753175838721, 288230377292562433, 288230383802122241,
            288230385815388161, 288230385815388161};

        Modulus prime(primes[logn - 12]);
        return prime;
    }
    else if ((modular_reduction == ModularReductionType::GOLDILOCK))
    {
        static Data primes[] = {
            18446744069414584321, 18446744069414584321, 18446744069414584321,
            18446744069414584321, 18446744069414584321, 18446744069414584321,
            18446744069414584321, 18446744069414584321, 18446744069414584321,
            18446744069414584321, 18446744069414584321, 18446744069414584321,
            18446744069414584321, 18446744069414584321, 18446744069414584321,
            18446744069414584321, 18446744069414584321};

        Modulus prime(primes[logn - 12]);
        return prime;
    }
    else
    {
        throw std::runtime_error("Reduction type is not supported!");
    }
}

Data NTTParameters4Step::omega_pool()
{
    customAssert(12 <= logn <= 24, "LOGN should be in range 12 to 24.");
    if ((modular_reduction == ModularReductionType::BARRET) ||
        (modular_reduction == ModularReductionType::PLANTARD))
    {
        static Data W[] = {
            288482366111684746, 37048445140799662,  459782973201979845,
            64800917766465203,  425015386842055933, 18734847765732801,
            119109113519742895, 227584740857897520, 477282059544659462,
            570131728462077067, 433594414095420776, 219263994987749328,
            189790554094222112, 96649110792683523, 250648942594717784,
            279172744045218282, 225865349704673648};

        return W[logn - 12];
    }
    else if ((modular_reduction == ModularReductionType::GOLDILOCK))
    {
        static Data W[] = {
            10246984546936836946, 13835541155533501494, 4150195341322954475,
            967374280115824382,   6097901191693030237,  11209893866417677111,
            3132005779464972240,  5277653236910267777,  14100038135461308321,
            6945674003198401120,  8888549356274591460,  17066151606537038542,
            4420008141159053259, 9092277472170284592, 2759361949145500591,
            14068871798966024971, 13445489538489244109};

        return W[logn - 12];
    }
    else
    {
        throw std::runtime_error("Reduction type is not supported!");
    }
}

Data NTTParameters4Step::psi_pool()
{
    customAssert(12 <= logn <= 24, "LOGN should be in range 12 to 24.");
    if ((modular_reduction == ModularReductionType::BARRET) ||
        (modular_reduction == ModularReductionType::PLANTARD))
    {
        static unsigned long long PSI[] = {
            238394956950829, 54612008597396, 8242615629351, 16141297350887,
            3760097055997,   11571974431275, 328867687796,  2298846063117,
            731868219707,    409596963254,   189266227206,  31864818375,
            92067739764, 5214432335, 734084005, 3351406780, 717004697};

        return PSI[logn - 12];
    }
    else if ((modular_reduction == ModularReductionType::GOLDILOCK))
    {
        static unsigned long long PSI[] = {
            3864827780506026, 5467143286794452, 563364698871119,
            633449661190857,  352105042511453,  309187323659538,
            21950690503823,   19333215661852,   3041397215123,
            22687970875277,   4798578965421,    204312508762,
            6369339824961, 36819894005, 31311927402,
            217045171522, 32062695042};

        return PSI[logn - 12];
    }
    else
    {
        throw std::runtime_error("Reduction type is not supported!");
    }
}

std::vector<int>  NTTParameters4Step::matrix_dimention() {
		customAssert(12 <= logn <= 24, "LOGN should be in range 12 to 24.");
		std::vector<int> shape;
		switch (logn) {
        case 12:
			shape = { 32, 128 };
            return shape;
        case 13:
            shape = { 32, 256 };
            return shape;
        case 14:
            shape = { 32, 512 };
            return shape;
		case 15:
            shape = { 64, 512 };
            return shape;
		case 16:
            //shape = { 128, 512 };
            shape = {256,256};
            return shape;
		case 17:
            shape = { 32, 4096 };
            return shape;
		case 18:
            shape = { 32, 8192 };
            return shape;
		case 19:
            shape = { 32, 16384 };
            return shape;
		case 20:
            shape = { 32, 32768 };
            return shape;
		case 21:
            shape = { 64, 32768 };
            return shape;
		case 22:
            shape = { 128, 32768 };
            return shape;
		case 23:
            shape = { 128, 65536 };
            return shape;
		case 24:
            //shape = { 256, 65536 };
            shape = {4096,4096};
            return shape;
        default:
            throw std::runtime_error("Invalid choice.\n");
    	}

	}

/*2024-8-1:
处理逻辑:依据n次本原单位根，计算得到n1次本原单位根,n1_based_root_of_unity_table用于存储n1次本原单位根的0至n1/2次的值*/
void NTTParameters4Step::small_forward_root_of_unity_table_generator(){

    Data exp_n1 =  int(n / n1);
    Data small_root_of_unity_n1 = VALUE::exp(root_of_unity, exp_n1, modulus); //本原单位根的n2次方就是n1次本原单位根
    n1_based_root_of_unity_table.push_back(1);
    for (int i = 1; i < (n1 >> 1); i++)
    {
        Data exp = VALUE::mult(n1_based_root_of_unity_table[i - 1],
                               small_root_of_unity_n1, modulus); //前一个的值乘上 n1次本原单位根
        n1_based_root_of_unity_table.push_back(exp);
    }

    Data exp_n2 =  int(n / n2);
    Data small_root_of_unity_n2 = VALUE::exp(root_of_unity, exp_n2, modulus); 
    n2_based_root_of_unity_table.push_back(1);
    for (int i = 1; i < (n2 >> 1); i++)
    {
        Data exp = VALUE::mult(n2_based_root_of_unity_table[i - 1],
                               small_root_of_unity_n2, modulus);
        n2_based_root_of_unity_table.push_back(exp);
    }

}

/*2024-8-9:
处理逻辑，计算64阶本单位根后,计算其0-31次方的值
*/
void NTTParameters4Step::n64_root_of_unity_table_generator(){
    Data exp_n64 =  int(n / UNITY_SIZE1);
    Data small_root_of_unity_n64 = VALUE::exp(root_of_unity, exp_n64, modulus);
    //printf("small_root_of_unity_n64:%lld root_of_unity:%lld",small_root_of_unity_n64,root_of_unity);
    n64_root_of_unity_table.push_back(1);
    for(int i=1; i < (UNITY_SIZE1 >> 1); i++){
        Data exp = VALUE::mult(n64_root_of_unity_table[i - 1],
                               small_root_of_unity_n64, modulus);
        n64_root_of_unity_table.push_back(exp);
        //printf("%lld,",exp);
    }
    //printf("\n");
}

/*2024-8-4:
生成修改因子*/
void NTTParameters4Step::TW_forward_table_generator(){

    int lg = log2(n1);
    for(int i = 0; i < n1; i++){
        for(int j = 0; j < n2; j++){
            Data index = bitreverse(i, lg);
            index = index * j;
            W_root_of_unity_table.push_back(VALUE::exp(root_of_unity, index, modulus));
        }
    }

}

void NTTParameters4Step::small_inverse_root_of_unity_table_generator(){

    Data exp_n1 =  int(n / n1);
    Data small_root_of_unity_n1 = VALUE::exp(root_of_unity, exp_n1, modulus); 
    small_root_of_unity_n1 = VALUE::modinv(small_root_of_unity_n1, modulus);
    n1_based_inverse_root_of_unity_table.push_back(1);
    for (int i = 1; i < (n1 >> 1); i++)
    {
        Data exp = VALUE::mult(n1_based_inverse_root_of_unity_table[i - 1],
                               small_root_of_unity_n1, modulus);
        n1_based_inverse_root_of_unity_table.push_back(exp);
    }

    Data exp_n2 =  int(n / n2);
    Data small_root_of_unity_n2 = VALUE::exp(root_of_unity, exp_n2, modulus); 
    small_root_of_unity_n2 = VALUE::modinv(small_root_of_unity_n2, modulus);
    n2_based_inverse_root_of_unity_table.push_back(1);
    for (int i = 1; i < (n2 >> 1); i++)
    {
        Data exp = VALUE::mult(n2_based_inverse_root_of_unity_table[i - 1],
                               small_root_of_unity_n2, modulus);
        n2_based_inverse_root_of_unity_table.push_back(exp);
    }

}

/*2024-8-4:
更改修正因子为zeta^{bitrev(i)*j}*/
void NTTParameters4Step::TW_inverse_table_generator(){

    /*int lg = log2(n2);
    for(int i = 0; i < n1; i++){
        for(int j = 0; j < n2; j++){
            Data index = bitreverse(j, lg);
            index = index * i;
            W_inverse_root_of_unity_table.push_back(VALUE::exp(inverse_root_of_unity, index, modulus));
        }
    }*/

   int lg = log2(n1);
    for(int i = 0; i < n1; i++){
        for(int j = 0; j < n2; j++){
            Data index = bitreverse(i, lg);
            index = index * j;
            W_inverse_root_of_unity_table.push_back(VALUE::exp(inverse_root_of_unity, index, modulus));
        }
    }
}

/*2024-8-9:
*/
void NTTParameters4Step::n64_W_root_of_unity_table_generator(){
    Data exp_n4096 = int(n / (UNITY_SIZE1 * UNITY_SIZE2)) ;//DEBUG:这里一定要单独把exp_n64写出来
    Data n4096_root_of_unity = VALUE::exp(root_of_unity, exp_n4096, modulus);
    //4096 = 64 * 64
    int lg = log2(UNITY_SIZE1);
    for(int i = 0; i < UNITY_SIZE1; i++){
        for(int j = 0; j < UNITY_SIZE2; j++){
            Data index = bitreverse(i, lg);
            index = index * j;
            n64_W_root_of_unity_table.push_back(VALUE::exp(n4096_root_of_unity, index, modulus));
        }
    }
   //printf("n4096_root_of_unity:%lld\n",n4096_root_of_unity);
}

void NTTParameters4Step::n_inverse_generator(){
    n_inv = VALUE::modinv(n, modulus);
}

void NTTParameters4Step::n_inverse_generator_gpu(){
#if defined(BARRETT_64) || defined(GOLDILOCKS_64)
    n_inv_gpu = VALUE::modinv(n, modulus);
#elif defined(PLANTARD_64)
    Data n_inv_ = VALUE::modinv(n, modulus);
    Data base_ = 2;
    Data expo_ = 128;
    Data correction = modulus.value - VALUE::exp(base_, expo_, modulus);

    Data mult = VALUE::mult(n_inv_, correction, modulus);

    __uint128_t mult_ = (__uint128_t)mult * (__uint128_t)R;

    n_inv_gpu.x = (mult_ & UINT64_MAX);
    n_inv_gpu.y = (mult_ >> 64u);
#else
#error "Please define reduction type."
#endif
}



#ifdef PLANTARD_64
__uint128_t NTTParameters4Step::R_pool()
{
    customAssert(12 <= logn <= 24, "LOGN should be in range 12 to 24.");

    if ((modular_reduction == ModularReductionType::BARRET) ||
        (modular_reduction == ModularReductionType::GOLDILOCK))
    {
        std::runtime_error("Only Plantard reduction can be used!");
    }
    else if ((modular_reduction == ModularReductionType::PLANTARD))
    {
        static __uint128_t R_lo[] = {
            17874787470856429569, 17942336517665964033, 17975028228979458049,
            14490355320300634113, 14490355320300634113, 16861622415275851777,
            16861622415275851777, 5189493672438136833,  8650610041606373377,
            17899855785974824961, 17956485032730165249, 18390730552622710785,
            184647583849775105};

        static __uint128_t R_hi[] = {
            11555533987311009281, 13842095462251758527, 9154506088165204343,
            18092274698773475081, 18092274698773475081, 5085365105665609632,
            5085365105665609632,  15936220441903101320, 2182956421460119645,
            10044172439809674011, 11784997034778628255, 4955880097643588863,
            18424807717242322944};

        __uint128_t R = ((__uint128_t)R_hi[logn - 12] << (__uint128_t)64) +
                        ((__uint128_t)R_lo[logn - 12]);

        return R;
    }
    else
    {
        throw std::runtime_error("Reduction type is not supported!");
    }
}
#endif

/*2024-8-7:
将本原单位根的次方表,转化为比特翻转的顺序*/
std::vector<Root_> NTTParameters4Step::gpu_root_of_unity_table_generator(
    std::vector<Data> table)
{
    // Taking Bitreverse order of root of unity table
    //printf("[hxw debug] %d\n",table.size());
    std::vector<Data> new_table;
    int lg = log2(table.size());
    for (int i = 0; i < table.size(); i++)
    {
        new_table.push_back(table[bitreverse(i, lg)]);
    }

#if defined(BARRETT_64) || defined(GOLDILOCKS_64)
    return new_table;
#elif defined(PLANTARD_64)
    Data base_ = 2;
    Data expo_ = 128;
    Data correction = modulus.value - VALUE::exp(base_, expo_, modulus);

    std::vector<__uint128_t> plantard_table;
    for (int i = 0; i < table.size(); i++)
    {
        Data temp = VALUE::mult(new_table[i], correction, modulus);

        __uint128_t mult_ = (__uint128_t)temp * (__uint128_t)R;

        plantard_table.push_back(mult_);
    }
    return plantard_table;
#else
#error "Please define reduction type."
#endif
}