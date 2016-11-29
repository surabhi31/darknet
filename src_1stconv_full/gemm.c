#include "gemm.h"
#include "utils.h"
#include "cuda.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define MIN(X,Y) (((X) < (Y)) ? (X) : (Y))

#define array_size(M,N) M*N

void gemm_nn_hw(int M, int N, int K, float ALPHA, 
        int lda, 
        float B[array_size(27,600)], int ldb,
        float C[array_size(16,600)], int ldc,
        float D[array_size(16,600)], int N_K, int NN);
void gemm_nn_sw(int M, int N, int K, float ALPHA, 
        int lda, 
        float B[array_size(27,200704)], int ldb,
        float C[array_size(16,200704)], int ldc,
        float D[array_size(16,200704)], int N_K, int NN);

void init_weights(float Weights[432]);

void gemm_bin(int M, int N, int K, float ALPHA, 
        char  *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            char A_PART = A[i*lda+k];
            if(A_PART){
                for(j = 0; j < N; ++j){
                    C[i*ldc+j] += B[k*ldb+j];
                }
            } else {
                for(j = 0; j < N; ++j){
                    C[i*ldc+j] -= B[k*ldb+j];
                }
            }
        }
    }
}

float *random_matrix(int rows, int cols)
{
    int i;
    float *m = calloc(rows*cols, sizeof(float));
    for(i = 0; i < rows*cols; ++i){
        m[i] = (float)rand()/RAND_MAX;
    }
    return m;
}

void time_random_matrix(int TA, int TB, int m, int k, int n)
{
    float *a;
    if(!TA) a = random_matrix(m,k);
    else a = random_matrix(k,m);
    int lda = (!TA)?k:m;
    float *b;
    if(!TB) b = random_matrix(k,n);
    else b = random_matrix(n,k);
    int ldb = (!TB)?n:k;

    float *c = random_matrix(m,n);
    int i;
    clock_t start = clock(), end;
    for(i = 0; i<10; ++i){
        gemm_cpu(TA,TB,m,n,k,1,a,lda,b,ldb,1,c,n);
    }
    end = clock();
    printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %lf ms\n",m,k,k,n, TA, TB, (float)(end-start)/CLOCKS_PER_SEC);
    free(a);
    free(b);
    free(c);
}


void gemm(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc)
{
    gemm_cpu( TA,  TB,  M, N, K, ALPHA,A,lda, B, ldb,BETA,C,ldc);
}

void gemm_nn(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    printf ("GEMM NN SW\n");
    int i,j,k;
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            register float A_PART = ALPHA*A[i*lda+k];
            for(j = 0; j < N; ++j){
                C[i*ldc+j] += A_PART*B[k*ldb+j];
            }
        }
    }
}

void gemm_nn_hw(int M, int N, int K, float ALPHA, 
        int lda, 
        float B[array_size(27,600)], int ldb,
        float C[array_size(16,600)], int ldc,
        float D[array_size(16,600)], int N_K, int NN)
{
    int i,j,k;
    float Weights[432];
    float tmp[array_size(16,600)];

    init_weights(Weights);

    // init temp
    for(i = 0; i < M; ++i){
      for(j = 0; j < ldc; ++j){
        tmp[i*ldc+j] = D[i*ldc+j];
      }
    }

    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            register float A_PART = ALPHA*Weights[i*lda+k];
            for(j = 0; j < N; ++j){
                tmp[i*ldc+j] += A_PART*B[k*ldb+j];
            }
        }
    }

    // Update C
    for(i = 0; i < M; ++i){
      for(j = 0; j < ldc; ++j){
        C[i*ldc+j] = tmp[i*ldc+j];
      }
    }

}

void gemm_nn_sw(int M, int N, int K, float ALPHA, 
        int lda, 
        float B[array_size(27,200704)], int ldb,
        float C[array_size(16,200704)], int ldc,
        float D[array_size(16,200704)], int N_K, int NN)
{
    int i,j,k;
    float Weights[432];
    init_weights(Weights);
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            register float A_PART = ALPHA*Weights[i*lda+k];
            for(j = N_K; j < MIN(N_K+N, NN); ++j){
                D[i*ldc+j] = C[i*ldc+j] + A_PART*B[k*ldb+j];
            }
        }
    }
}

void gemm_nt(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            register float sum = 0;
            for(k = 0; k < K; ++k){
                sum += ALPHA*A[i*lda+k]*B[j*ldb + k];
            }
            C[i*ldc+j] += sum;
        }
    }
}

void gemm_tn(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            register float A_PART = ALPHA*A[k*lda+i];
            for(j = 0; j < N; ++j){
                C[i*ldc+j] += A_PART*B[k*ldb+j];
            }
        }
    }
}

void gemm_tt(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            register float sum = 0;
            for(k = 0; k < K; ++k){
                sum += ALPHA*A[i+k*lda]*B[k+j*ldb];
            }
            C[i*ldc+j] += sum;
        }
    }
}


void gemm_cpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc)
{
    //printf("cpu: %d %d %d %d %d %f %d %d %f %d\n",TA, TB, M, N, K, ALPHA, lda, ldb, BETA, ldc);
    int i, j;
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            C[i*ldc + j] *= BETA;
        }
    }
    if(!TA && !TB)
        gemm_nn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else if(TA && !TB)
        gemm_tn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else if(!TA && TB)
        gemm_nt(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else
        gemm_tt(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
}

void gemm_cpu_hw(int TA, int TB, int M, int N, int K, float ALPHA, 
        float A[FIXED_SIZE], int lda, 
        float B[FIXED_SIZE], int ldb,
        float BETA,
        float C[FIXED_SIZE], int ldc)
{
    int i, j;
    int P = 600;
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            C[i*ldc + j] *= BETA;
        }
    }


    printf("Allocating memory! \n");
    float *IFMAP;
    IFMAP=(float *) malloc(K*N*sizeof(float));

    float *OFMAP;
    OFMAP=(float *) malloc(M*N*sizeof(float));

    float *OFMAP_TMP;
    OFMAP_TMP=(float *) malloc(M*N*sizeof(float));

    if (!IFMAP || !OFMAP || !OFMAP_TMP ) {
       if (IFMAP) free(IFMAP);
       if (OFMAP) free(OFMAP);
       if (OFMAP_TMP) free(OFMAP_TMP);
       printf("Mem alloc failed");
    } 

//    printf("Main \n");
//    //gemm_nn_hw(M, N, K, ALPHA,lda, IFMAP, ldb,OFMAP_TMP,ldc,OFMAP,0,N);
//    gemm_nn_hw(M, N, K, ALPHA,lda, IFMAP, ldb,OFMAP,ldc,OFMAP_TMP,0,N);
//
//    for(i = 0; i<M; ++i ) {
//      for(j = 0; j<N; ++j ) {
//        C[i*ldc+j] = OFMAP[i*ldc+j];
//      }
//    }
//    C = D;

    int x;
    for(x = 0; x < N; x = x+P) {
      // Init arrays
      for(i = 0; i<K; ++i ) {
        for(j = x; j< MIN(P+x,N); ++j ) {
          IFMAP[i*P+(j-x)] = B[i*ldb+j]; 
        }
      }
      for(i = 0; i<M; ++i ) {
        for(j = x; j<MIN(P+x,N); ++j ) {
          OFMAP[i*P+(j-x)] = C[i*ldc+j]; 
        }
      }

      for(i = 0; i<M; ++i ) {
        for(j = x; j<MIN(P+x,N); ++j ) {
          OFMAP_TMP[i*P+(j-x)] = C[i*ldc+j]; 
        }
      }
      gemm_nn_hw(M, P, K, ALPHA,lda, IFMAP, P,OFMAP,P,OFMAP_TMP,x,N);

      for(i = 0; i<M; ++i ) {
        for(j = x; j<MIN(P+x,N); ++j ) {
          C[i*ldc+j] = OFMAP[i*P+(j-x)];
        }
      }
    }

    free(IFMAP);
    free(OFMAP);
    free(OFMAP_TMP);
//    float *D = C;
//    printf("Entering compute in sw mode! \n");
//    for(j = 300*P; j < N; j=j+P){
//      gemm_nn_sw(M, P, K, ALPHA,lda, B, ldb,C,ldc,D,j,N);
//    }
//    C = D;
}

#ifdef GPU

#include <math.h>

void gemm_ongpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A_gpu, int lda, 
        float *B_gpu, int ldb,
        float BETA,
        float *C_gpu, int ldc)
{
    cublasHandle_t handle = blas_handle();
    cudaError_t status = cublasSgemm(handle, (TB ? CUBLAS_OP_T : CUBLAS_OP_N), 
            (TA ? CUBLAS_OP_T : CUBLAS_OP_N), N, M, K, &ALPHA, B_gpu, ldb, A_gpu, lda, &BETA, C_gpu, ldc);
    check_error(status);
}

void gemm_gpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc)
{
    float *A_gpu = cuda_make_array(A, (TA ? lda*K:lda*M));
    float *B_gpu = cuda_make_array(B, (TB ? ldb*N : ldb*K));
    float *C_gpu = cuda_make_array(C, ldc*M);

    gemm_ongpu(TA, TB, M, N, K, ALPHA, A_gpu, lda, B_gpu, ldb, BETA, C_gpu, ldc);

    cuda_pull_array(C_gpu, C, ldc*M);
    cuda_free(A_gpu);
    cuda_free(B_gpu);
    cuda_free(C_gpu);
}

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void time_gpu_random_matrix(int TA, int TB, int m, int k, int n)
{
    float *a;
    if(!TA) a = random_matrix(m,k);
    else a = random_matrix(k,m);
    int lda = (!TA)?k:m;
    float *b;
    if(!TB) b = random_matrix(k,n);
    else b = random_matrix(n,k);
    int ldb = (!TB)?n:k;

    float *c = random_matrix(m,n);
    int i;
    clock_t start = clock(), end;
    for(i = 0; i<32; ++i){
        gemm_gpu(TA,TB,m,n,k,1,a,lda,b,ldb,1,c,n);
    }
    end = clock();
    printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %lf s\n",m,k,k,n, TA, TB, (float)(end-start)/CLOCKS_PER_SEC);
    free(a);
    free(b);
    free(c);
}

void time_ongpu(int TA, int TB, int m, int k, int n)
{
    int iter = 10;
    float *a = random_matrix(m,k);
    float *b = random_matrix(k,n);

    int lda = (!TA)?k:m;
    int ldb = (!TB)?n:k;

    float *c = random_matrix(m,n);

    float *a_cl = cuda_make_array(a, m*k);
    float *b_cl = cuda_make_array(b, k*n);
    float *c_cl = cuda_make_array(c, m*n);

    int i;
    clock_t start = clock(), end;
    for(i = 0; i<iter; ++i){
        gemm_ongpu(TA,TB,m,n,k,1,a_cl,lda,b_cl,ldb,1,c_cl,n);
        cudaThreadSynchronize();
    }
    double flop = ((double)m)*n*(2.*k + 2.)*iter;
    double gflop = flop/pow(10., 9);
    end = clock();
    double seconds = sec(end-start);
    printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %lf s, %lf GFLOPS\n",m,k,k,n, TA, TB, seconds, gflop/seconds);
    cuda_free(a_cl);
    cuda_free(b_cl);
    cuda_free(c_cl);
    free(a);
    free(b);
    free(c);
}


void test_gpu_accuracy(int TA, int TB, int m, int k, int n)
{
    srand(0);
    float *a;
    if(!TA) a = random_matrix(m,k);
    else a = random_matrix(k,m);
    int lda = (!TA)?k:m;
    float *b;
    if(!TB) b = random_matrix(k,n);
    else b = random_matrix(n,k);
    int ldb = (!TB)?n:k;

    float *c = random_matrix(m,n);
    float *c_gpu = random_matrix(m,n);
    memset(c, 0, m*n*sizeof(float));
    memset(c_gpu, 0, m*n*sizeof(float));
    int i;
    //pm(m,k,b);
    gemm_gpu(TA,TB,m,n,k,1,a,lda,b,ldb,1,c_gpu,n);
    //printf("GPU\n");
    //pm(m, n, c_gpu);

    gemm_cpu(TA,TB,m,n,k,1,a,lda,b,ldb,1,c,n);
    //printf("\n\nCPU\n");
    //pm(m, n, c);
    double sse = 0;
    for(i = 0; i < m*n; ++i) {
        //printf("%f %f\n", c[i], c_gpu[i]);
        sse += pow(c[i]-c_gpu[i], 2);
    }
    printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %g SSE\n",m,k,k,n, TA, TB, sse/(m*n));
    free(a);
    free(b);
    free(c);
    free(c_gpu);
}

int test_gpu_blas()
{
    /*
       test_gpu_accuracy(0,0,10,576,75); 

       test_gpu_accuracy(0,0,17,10,10); 
       test_gpu_accuracy(1,0,17,10,10); 
       test_gpu_accuracy(0,1,17,10,10); 
       test_gpu_accuracy(1,1,17,10,10); 

       test_gpu_accuracy(0,0,1000,10,100); 
       test_gpu_accuracy(1,0,1000,10,100); 
       test_gpu_accuracy(0,1,1000,10,100); 
       test_gpu_accuracy(1,1,1000,10,100); 

       test_gpu_accuracy(0,0,10,10,10); 

       time_ongpu(0,0,64,2916,363); 
       time_ongpu(0,0,64,2916,363); 
       time_ongpu(0,0,64,2916,363); 
       time_ongpu(0,0,192,729,1600); 
       time_ongpu(0,0,384,196,1728); 
       time_ongpu(0,0,256,196,3456); 
       time_ongpu(0,0,256,196,2304); 
       time_ongpu(0,0,128,4096,12544); 
       time_ongpu(0,0,128,4096,4096); 
     */
    time_ongpu(0,0,64,75,12544); 
    time_ongpu(0,0,64,75,12544); 
    time_ongpu(0,0,64,75,12544); 
    time_ongpu(0,0,64,576,12544); 
    time_ongpu(0,0,256,2304,784); 
    time_ongpu(1,1,2304,256,784); 
    time_ongpu(0,0,512,4608,196); 
    time_ongpu(1,1,4608,512,196); 

    return 0;
}
#endif

void init_weights(float Weights[432]) {
  Weights[0]   = -0.064049;
  Weights[1]   = -0.045396;
  Weights[2]   = -0.037942;
  Weights[3]   = -0.110669;
  Weights[4]   = -0.142887;
  Weights[5]   = -0.096666;
  Weights[6]   = -0.059662;
  Weights[7]   = -0.140883;
  Weights[8]   = -0.077999;
  Weights[9]   = -0.029043;
  Weights[10]  = -0.015262;
  Weights[11]  = 0.016828;
  Weights[12]  = -0.086994;
  Weights[13]  = -0.098943;
  Weights[14]  = -0.057030;
  Weights[15]  = -0.095970;
  Weights[16]  = -0.142695;
  Weights[17]  = -0.085212;
  Weights[18]  = 0.035949;
  Weights[19]  = 0.021758;
  Weights[20]  = 0.069699;
  Weights[21]  = -0.016599;
  Weights[22]  = -0.039375;
  Weights[23]  = 0.016566;
  Weights[24]  = 0.009513;
  Weights[25]  = -0.060279;
  Weights[26]  = 0.011305;
  Weights[27]  = -0.367249;
  Weights[28]  = -0.539303;
  Weights[29]  = -0.322591;
  Weights[30]  = 0.060934;
  Weights[31]  = -0.069708;
  Weights[32]  = 0.020492;
  Weights[33]  = 0.354283;
  Weights[34]  = 0.484615;
  Weights[35]  = 0.411248;
  Weights[36]  = -0.386754;
  Weights[37]  = -0.575147;
  Weights[38]  = -0.350489;
  Weights[39]  = 0.077713;
  Weights[40]  = -0.037349;
  Weights[41]  = 0.011587;
  Weights[42]  = 0.329123;
  Weights[43]  = 0.488097;
  Weights[44]  = 0.373668;
  Weights[45]  = -0.274666;
  Weights[46]  = -0.386286;
  Weights[47]  = -0.252450;
  Weights[48]  = 0.081470;
  Weights[49]  = -0.000253;
  Weights[50]  = 0.027131;
  Weights[51]  = 0.244731;
  Weights[52]  = 0.345035;
  Weights[53]  = 0.271151;
  Weights[54]  = 0.048910;
  Weights[55]  = 0.044255;
  Weights[56]  = 0.013175;
  Weights[57]  = 0.072052;
  Weights[58]  = 0.100945;
  Weights[59]  = 0.040982;
  Weights[60]  = 0.061299;
  Weights[61]  = 0.056611;
  Weights[62]  = 0.007536;
  Weights[63]  = 0.065697;
  Weights[64]  = 0.074062;
  Weights[65]  = 0.066363;
  Weights[66]  = 0.096432;
  Weights[67]  = 0.133825;
  Weights[68]  = 0.110367;
  Weights[69]  = 0.079579;
  Weights[70]  = 0.077771;
  Weights[71]  = 0.051433;
  Weights[72]  = -0.079734;
  Weights[73]  = -0.112314;
  Weights[74]  = -0.098023;
  Weights[75]  = -0.100942;
  Weights[76]  = -0.130901;
  Weights[77]  = -0.110407;
  Weights[78]  = -0.144683;
  Weights[79]  = -0.186492;
  Weights[80]  = -0.160715;
  Weights[81]  = 0.020491;
  Weights[82]  = 0.018107;
  Weights[83]  = 0.017737;
  Weights[84]  = 0.021935;
  Weights[85]  = 0.012123;
  Weights[86]  = 0.030554;
  Weights[87]  = 0.043495;
  Weights[88]  = 0.037545;
  Weights[89]  = 0.052784;
  Weights[90]  = 0.066645;
  Weights[91]  = 0.057874;
  Weights[92]  = 0.076482;
  Weights[93]  = 0.047373;
  Weights[94]  = 0.017640;
  Weights[95]  = 0.060144;
  Weights[96]  = 0.086085;
  Weights[97]  = 0.044519;
  Weights[98]  = 0.092652;
  Weights[99]  = 0.075497;
  Weights[100] = 0.081775;
  Weights[101] = 0.085064;
  Weights[102] = 0.074896;
  Weights[103] = 0.056466;
  Weights[104] = 0.068583;
  Weights[105] = 0.086938;
  Weights[106] = 0.055582;
  Weights[107] = 0.087979;
  Weights[108] = -0.285478;
  Weights[109] = -0.019653;
  Weights[110] = 0.345278;
  Weights[111] = -0.438799;
  Weights[112] = -0.037588;
  Weights[113] = 0.436483;
  Weights[114] = -0.383369;
  Weights[115] = 0.051548;
  Weights[116] = 0.366270;
  Weights[117] = -0.329909;
  Weights[118] = -0.026821;
  Weights[119] = 0.334411;
  Weights[120] = -0.514832;
  Weights[121] = -0.039002;
  Weights[122] = 0.460419;
  Weights[123] = -0.402291;
  Weights[124] = 0.058740;
  Weights[125] = 0.363699;
  Weights[126] = -0.206056;
  Weights[127] = -0.005059;
  Weights[128] = 0.224419;
  Weights[129] = -0.321306;
  Weights[130] = -0.010215;
  Weights[131] = 0.298698;
  Weights[132] = -0.279343;
  Weights[133] = 0.070275;
  Weights[134] = 0.261606;
  Weights[135] = 0.242569;
  Weights[136] = 0.116060;
  Weights[137] = -0.013415;
  Weights[138] = -0.419785;
  Weights[139] = -0.326175;
  Weights[140] = -0.022160;
  Weights[141] = 0.160009;
  Weights[142] = 0.198714;
  Weights[143] = 0.038484;
  Weights[144] = 0.294787;
  Weights[145] = 0.137222;
  Weights[146] = -0.010351;
  Weights[147] = -0.501659;
  Weights[148] = -0.383250;
  Weights[149] = -0.023648;
  Weights[150] = 0.214822;
  Weights[151] = 0.259429;
  Weights[152] = 0.045957;
  Weights[153] = 0.189860;
  Weights[154] = 0.076730;
  Weights[155] = -0.005162;
  Weights[156] = -0.331359;
  Weights[157] = -0.249885;
  Weights[158] = -0.014541;
  Weights[159] = 0.131527;
  Weights[160] = 0.152464;
  Weights[161] = 0.013625;
  Weights[162] = -0.203777;
  Weights[163] = -0.108901;
  Weights[164] = -0.045687;
  Weights[165] = -0.096644;
  Weights[166] = 0.152983;
  Weights[167] = 0.159629;
  Weights[168] = -0.121730;
  Weights[169] = 0.062618;
  Weights[170] = 0.197030;
  Weights[171] = -0.240978;
  Weights[172] = -0.091014;
  Weights[173] = -0.010481;
  Weights[174] = -0.146597;
  Weights[175] = 0.182564;
  Weights[176] = 0.181673;
  Weights[177] = -0.146679;
  Weights[178] = 0.096083;
  Weights[179] = 0.218333;
  Weights[180] = -0.139365;
  Weights[181] = -0.060262;
  Weights[182] = -0.010551;
  Weights[183] = -0.104560;
  Weights[184] = 0.111869;
  Weights[185] = 0.103572;
  Weights[186] = -0.085335;
  Weights[187] = 0.025382;
  Weights[188] = 0.116080;
  Weights[189] = 0.097371;
  Weights[190] = 0.167597;
  Weights[191] = 0.110650;
  Weights[192] = -0.036876;
  Weights[193] = 0.017842;
  Weights[194] = 0.057932;
  Weights[195] = -0.150641;
  Weights[196] = -0.139692;
  Weights[197] = -0.046319;
  Weights[198] = 0.077428;
  Weights[199] = 0.060498;
  Weights[200] = 0.038227;
  Weights[201] = 0.105800;
  Weights[202] = 0.107418;
  Weights[203] = 0.083196;
  Weights[204] = 0.109122;
  Weights[205] = 0.110351;
  Weights[206] = 0.086962;
  Weights[207] = -0.098519;
  Weights[208] = -0.133139;
  Weights[209] = -0.061195;
  Weights[210] = 0.035622;
  Weights[211] = 0.010370;
  Weights[212] = -0.015952;
  Weights[213] = 0.099417;
  Weights[214] = 0.131931;
  Weights[215] = 0.025812;
  Weights[216] = 0.156707;
  Weights[217] = -0.356221;
  Weights[218] = 0.210613;
  Weights[219] = 0.246392;
  Weights[220] = -0.442155;
  Weights[221] = 0.194008;
  Weights[222] = 0.179819;
  Weights[223] = -0.257459;
  Weights[224] = 0.072929;
  Weights[225] = 0.171351;
  Weights[226] = -0.465199;
  Weights[227] = 0.268176;
  Weights[228] = 0.299898;
  Weights[229] = -0.559181;
  Weights[230] = 0.245532;
  Weights[231] = 0.214966;
  Weights[232] = -0.278101;
  Weights[233] = 0.091597;
  Weights[234] = 0.110905;
  Weights[235] = -0.271801;
  Weights[236] = 0.148567;
  Weights[237] = 0.172845;
  Weights[238] = -0.315583;
  Weights[239] = 0.144872;
  Weights[240] = 0.114207;
  Weights[241] = -0.175049;
  Weights[242] = 0.057914;
  Weights[243] = 0.031307;
  Weights[244] = 0.058963;
  Weights[245] = -0.007255;
  Weights[246] = 0.051301;
  Weights[247] = 0.100871;
  Weights[248] = 0.062653;
  Weights[249] = 0.035226;
  Weights[250] = 0.045940;
  Weights[251] = 0.023455;
  Weights[252] = -0.077770;
  Weights[253] = -0.054609;
  Weights[254] = -0.116076;
  Weights[255] = -0.141094;
  Weights[256] = -0.115833;
  Weights[257] = -0.117307;
  Weights[258] = -0.201489;
  Weights[259] = -0.226906;
  Weights[260] = -0.198016;
  Weights[261] = 0.119785;
  Weights[262] = 0.153772;
  Weights[263] = 0.078824;
  Weights[264] = 0.119333;
  Weights[265] = 0.160290;
  Weights[266] = 0.140711;
  Weights[267] = 0.087031;
  Weights[268] = 0.081911;
  Weights[269] = 0.076539;
  Weights[270] = 0.326242;
  Weights[271] = 0.040075;
  Weights[272] = -0.365727;
  Weights[273] = 0.450917;
  Weights[274] = -0.042975;
  Weights[275] = -0.464367;
  Weights[276] = 0.403586;
  Weights[277] = 0.024881;
  Weights[278] = -0.331582;
  Weights[279] = 0.347123;
  Weights[280] = 0.043263;
  Weights[281] = -0.422495;
  Weights[282] = 0.487444;
  Weights[283] = -0.046382;
  Weights[284] = -0.547478;
  Weights[285] = 0.385702;
  Weights[286] = 0.016303;
  Weights[287] = -0.349749;
  Weights[288] = 0.201352;
  Weights[289] = 0.042346;
  Weights[290] = -0.256398;
  Weights[291] = 0.289306;
  Weights[292] = -0.025776;
  Weights[293] = -0.322066;
  Weights[294] = 0.259447;
  Weights[295] = 0.036159;
  Weights[296] = -0.194817;
  Weights[297] = 0.060476;
  Weights[298] = -0.026369;
  Weights[299] = 0.094858;
  Weights[300] = -0.101062;
  Weights[301] = -0.200735;
  Weights[302] = 0.013529;
  Weights[303] = -0.054191;
  Weights[304] = -0.082036;
  Weights[305] = 0.157404;
  Weights[306] = 0.050235;
  Weights[307] = 0.007907;
  Weights[308] = 0.062751;
  Weights[309] = -0.059916;
  Weights[310] = -0.125095;
  Weights[311] = 0.017297;
  Weights[312] = -0.072896;
  Weights[313] = -0.077088;
  Weights[314] = 0.071014;
  Weights[315] = -0.018860;
  Weights[316] = -0.001331;
  Weights[317] = -0.013083;
  Weights[318] = -0.068675;
  Weights[319] = -0.065929;
  Weights[320] = -0.019279;
  Weights[321] = -0.097682;
  Weights[322] = -0.049194;
  Weights[323] = 0.012213;
  Weights[324] = -0.154086;
  Weights[325] = -0.151793;
  Weights[326] = -0.128748;
  Weights[327] = -0.159260;
  Weights[328] = -0.148493;
  Weights[329] = -0.094127;
  Weights[330] = -0.231925;
  Weights[331] = -0.180055;
  Weights[332] = -0.097392;
  Weights[333] = 0.149652;
  Weights[334] = 0.141412;
  Weights[335] = 0.113743;
  Weights[336] = 0.158771;
  Weights[337] = 0.143767;
  Weights[338] = 0.133898;
  Weights[339] = 0.089024;
  Weights[340] = 0.091712;
  Weights[341] = 0.092802;
  Weights[342] = 0.078120;
  Weights[343] = 0.068321;
  Weights[344] = -0.026740;
  Weights[345] = 0.106323;
  Weights[346] = 0.081434;
  Weights[347] = -0.012973;
  Weights[348] = 0.057619;
  Weights[349] = 0.042367;
  Weights[350] = -0.067043;
  Weights[351] = 0.380687;
  Weights[352] = 0.505032;
  Weights[353] = 0.367969;
  Weights[354] = -0.011850;
  Weights[355] = -0.056445;
  Weights[356] = 0.057177;
  Weights[357] = -0.352487;
  Weights[358] = -0.534827;
  Weights[359] = -0.354795;
  Weights[360] = 0.404564;
  Weights[361] = 0.551341;
  Weights[362] = 0.387640;
  Weights[363] = -0.003815;
  Weights[364] = -0.058395;
  Weights[365] = 0.061942;
  Weights[366] = -0.372085;
  Weights[367] = -0.604489;
  Weights[368] = -0.381647;
  Weights[369] = 0.275405;
  Weights[370] = 0.337972;
  Weights[371] = 0.239537;
  Weights[372] = 0.013591;
  Weights[373] = -0.053855;
  Weights[374] = 0.037433;
  Weights[375] = -0.231390;
  Weights[376] = -0.390020;
  Weights[377] = -0.257715;
  Weights[378] = 0.070536;
  Weights[379] = 0.060445;
  Weights[380] = 0.090112;
  Weights[381] = 0.033577;
  Weights[382] = 0.002713;
  Weights[383] = 0.080030;
  Weights[384] = 0.012857;
  Weights[385] = 0.036756;
  Weights[386] = 0.093639;
  Weights[387] = -0.076698;
  Weights[388] = -0.191530;
  Weights[389] = -0.169015;
  Weights[390] = -0.179441;
  Weights[391] = -0.306897;
  Weights[392] = -0.229474;
  Weights[393] = -0.160911;
  Weights[394] = -0.204887;
  Weights[395] = -0.154613;
  Weights[396] = 0.167764;
  Weights[397] = 0.104697;
  Weights[398] = 0.109872;
  Weights[399] = 0.112747;
  Weights[400] = 0.030831;
  Weights[401] = 0.090656;
  Weights[402] = 0.136328;
  Weights[403] = 0.122764;
  Weights[404] = 0.158473;
  Weights[405] = 0.181205;
  Weights[406] = 0.223681;
  Weights[407] = 0.170409;
  Weights[408] = 0.217128;
  Weights[409] = 0.264190;
  Weights[410] = 0.178473;
  Weights[411] = 0.176698;
  Weights[412] = 0.208407;
  Weights[413] = 0.133560;
  Weights[414] = -0.162913;
  Weights[415] = -0.188261;
  Weights[416] = -0.164639;
  Weights[417] = -0.188550;
  Weights[418] = -0.219027;
  Weights[419] = -0.176282;
  Weights[420] = -0.197434;
  Weights[421] = -0.204081;
  Weights[422] = -0.173185;
  Weights[423] = 0.003420;
  Weights[424] = -0.033567;
  Weights[425] = 0.003733;
  Weights[426] = -0.004463;
  Weights[427] = -0.050087;
  Weights[428] = 0.017009;
  Weights[429] = 0.001786;
  Weights[430] = -0.013393;
  Weights[431] = 0.030971;
}
