//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//
//#include <stdio.h>
//#include "Neuron.h"
//#include "kernel.cuh"
//
//
//#define cudaCheckErrors(msg) \
//    do { \
//        cudaError_t __err = cudaGetLastError(); \
//        if (__err != cudaSuccess) { \
//            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
//                msg, cudaGetErrorString(__err), \
//                __FILE__, __LINE__); \
//            fprintf(stderr, "*** FAILED - ABORTING\n"); \
//            exit(1); \
//        } \
//    } while (0)
//
////cudaCheckErrors("Elayer allocation error");
//__global__ void Sqs_X(float* X)
//{
//    if (threadIdx.x < 784)
//    {
//        X[threadIdx.x] = (1.0f / (1.0f + powf(2.71828f, -(X[threadIdx.x]))));
//
//    }
//}
//__global__ void Dot_w_X(float* refw, float* w, float* X, int w_row, int w_col)
//{
//    if (blockIdx.x < w_row && threadIdx.x < w_col)
//    {
//        refw[blockIdx.x * w_col + threadIdx.x] = (w[blockIdx.x * w_col + threadIdx.x] * X[threadIdx.x]);
//
//    }
//}
//
//
//int Kernal_alloc_mem(E_Layer& devEL)
//{
//
//    /*-----memeory allocation ---------*/
//    cudaMalloc((void**)&devEL.LF.X, 784 * 5 * sizeof(float));
//    cudaCheckErrors("Elayer allocation error");
//    cudaMalloc((void**)&devEL.LF.w, 784 * 20 * sizeof(float));
//    cudaMalloc((void**)&devEL.LF.refw, 784 * 20 * sizeof(float));
//    cudaMalloc((void**)&devEL.LF.b, 20 * sizeof(float));
//    cudaMalloc((void**)&devEL.LF.Y, 20 * sizeof(float));
//
//    // devHF[0] mem allocation
//    //cudaMalloc((void**)&devLF.X, 784 * sizeof(float));
//    cudaMalloc((void**)&devEL.LH[0].w, 20 * 20 * sizeof(float));
//    cudaCheckErrors("Elayer allocation error");
//    cudaMalloc((void**)&devEL.LH[0].refw, 20 * 20 * sizeof(float));
//    cudaMalloc((void**)&devEL.LH[0].b, 20 * sizeof(float));
//    cudaMalloc((void**)&devEL.LH[0].Y, 20 * sizeof(float));
//
//    // devHF[0] mem allocation
//    //cudaMalloc((void**)&devLF.X, 784 * sizeof(float));
//    cudaMalloc((void**)&devEL.LH[1].w, 20 * 20 * sizeof(float));
//    cudaCheckErrors("Elayer allocation error");
//    cudaMalloc((void**)&devEL.LH[1].refw, 20 * 20 * sizeof(float));
//    cudaMalloc((void**)&devEL.LH[1].b, 20 * sizeof(float));
//    cudaMalloc((void**)&devEL.LH[1].Y, 20 * sizeof(float));
//
//    //devLL mem allocation
//    cudaMalloc((void**)&devEL.LL.w, 20 * 10 * sizeof(float));
//    cudaCheckErrors("Elayer allocation error");
//    cudaMalloc((void**)&devEL.LL.refw, 20 * 10 * sizeof(float));
//    cudaMalloc((void**)&devEL.LL.b, 10 * sizeof(float));
//    cudaMalloc((void**)&devEL.LL.Y, 10 * sizeof(float));
//
//}
//
//
//int Kernal_free_mem(E_Layer& devEL)
//{
//    cudaFree(devEL.LF.X);
//    cudaFree(devEL.LF.b);
//    //cudaFree(&devLF.db);
//    cudaFree(devEL.LF.w);
//    cudaFree(devEL.LF.refw);
//    //cudaFree(&devLF.dw);
//    cudaFree(devEL.LF.Y);
//
//    cudaFree(devEL.LH[0].b);
//    //cudaFree(&devLH[0].db);
//    cudaFree(devEL.LH[0].w);
//    cudaFree(devEL.LH[0].refw);
//    //cudaFree(&devLH[0].dw);
//    cudaFree(devEL.LH[0].Y);
//
//    cudaFree(devEL.LH[1].b);
//    //cudaFree(&devLH[1].db);
//    cudaFree(devEL.LH[1].w);
//    cudaFree(devEL.LH[1].refw);
//    //cudaFree(&devLH[1].dw);
//    cudaFree(devEL.LH[1].Y);
//
//    cudaFree(devEL.LL.b);
//    //cudaFree(&devLL.db);
//    cudaFree(devEL.LL.w);
//    cudaFree(devEL.LL.refw);
//    //cudaFree(&devLL.dw);
//    cudaFree(devEL.LL.Y);
//    cudaCheckErrors("free mem failed");
//
//    printf("-- cuda freed memory --\n");
//}
//
//
//
//int Kernal_cpy_host(E_Layer& devEL, E_Layer& EL)
//{
//    cudaMemcpy(devEL.LF.X, EL.io.X, 784 * 5 * sizeof(float), cudaMemcpyHostToDevice);
//    cudaCheckErrors("copy failed from io failed");
//
//    cudaMemcpy(devEL.LF.w, EL.LF.w, 784 * 20 * sizeof(float), cudaMemcpyHostToDevice);
//    cudaCheckErrors("copy failed LF");
//    cudaMemcpy(devEL.LF.b, EL.LF.b, 20 * sizeof(float), cudaMemcpyHostToDevice);
//    cudaCheckErrors("copy failed LF");
//    cudaMemcpy(devEL.LF.Y, EL.LF.Y, 20 * sizeof(float), cudaMemcpyHostToDevice);
//    cudaCheckErrors("copy failed LF");
//
//    cudaMemcpy(devEL.LH[0].w, EL.LH[0].w, 20 * 20 * sizeof(float), cudaMemcpyHostToDevice);
//    cudaCheckErrors("copy failed LH");
//    cudaMemcpy(devEL.LH[0].b, EL.LH[0].b, 20 * sizeof(float), cudaMemcpyHostToDevice);
//    cudaCheckErrors("copy failed LH");
//    cudaMemcpy(devEL.LH[0].Y, EL.LH[0].Y, 20 * sizeof(float), cudaMemcpyHostToDevice);
//    cudaCheckErrors("copy failed LH");
//
//    cudaMemcpy(devEL.LH[1].w, EL.LH[1].w, 20 * 20 * sizeof(float), cudaMemcpyHostToDevice);
//    cudaCheckErrors("copy failed LH");
//    cudaMemcpy(devEL.LH[1].b, EL.LH[1].b, 20 * sizeof(float), cudaMemcpyHostToDevice);
//    cudaCheckErrors("copy failed LH");
//    cudaMemcpy(devEL.LH[1].Y, EL.LH[1].Y, 20 * sizeof(float), cudaMemcpyHostToDevice);
//    cudaCheckErrors("copy failed LH");
//
//    cudaMemcpy(devEL.LL.w, EL.LL.w, 20 * 10 * sizeof(float), cudaMemcpyHostToDevice);
//    cudaCheckErrors("copy failed LL");
//    cudaMemcpy(devEL.LL.b, EL.LL.b, 10 * sizeof(float), cudaMemcpyHostToDevice);
//    cudaCheckErrors("copy failed LL");
//    cudaMemcpy(devEL.LL.Y, EL.LL.Y, 10 * sizeof(float), cudaMemcpyHostToDevice);
//    cudaCheckErrors("copy failed LL");
//
//
//}
//
//int Kernal_cpy_device(float* des, float* src, int size)
//{
//    //cudaMemcpy()
//}
