
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "kernel.cuh"



#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

int Comp_Minimap(E_Layer & devEL, int i);
void calc_dw_db(E_Layer * EL, E_Layer & devEL, float* ds, float* devds, float* dw, float real_cost, int arrsize);
//cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

//__global__ void addKernel(int *c, const int *a, const int *b)
//{
//    int i = threadIdx.x;
//    c[i] = a[i] + b[i];
//}

//how it works
//__global__ void dot_w_X_f(float* refw, float* w, float* X)
//{
//    if (blockIdx.x < 20 && threadIdx.x < 784)
//    {
//        refw[blockIdx.x * 784 + threadIdx.x] = (w[blockIdx.x * 784 + threadIdx.x] * X[threadIdx.x]);
//
//    }
//}
// example 
//__global__ void Comp_y(float* Y, float* refw, float* b)
//{
//    if (threadIdx.x < 20 && blockIdx.x < 1)
//    {
//        for (int i = 0; i < 784; i++)
//        {
//            Y[threadIdx.x] += refw[threadIdx.x * 784 + i];
//        }
//        Y[threadIdx.x] += b[threadIdx.x];
//        Y[threadIdx.x] = (1.0f / (1.0f + powf(2.71828f, -(Y[threadIdx.x]))));
//    }
//
//}

// 'bl' is block and 'th' is thread
__global__ void Sqs_X(float* X)
{
    if (X[threadIdx.x] > 1)
    {
        X[threadIdx.x] = 1;
    }
    else if (X[threadIdx.x] < 0)
    {
        X[threadIdx.x] = 0;
    }
    /*if (threadIdx.x < 784)
    {
        X[threadIdx.x] = (1.0f / (1.0f + powf(2.71828f, -(X[threadIdx.x]))));

    }*/
}
__global__ void Dot_w_X(float *refw, float *w,float *X, int w_row, int w_col)
{
    if (blockIdx.x < w_row && threadIdx.x < w_col)
    {
        refw[blockIdx.x * w_col + threadIdx.x] = (w[blockIdx.x * w_col + threadIdx.x] * X[threadIdx.x]);

    }
}

// 'bl' is block and 'th' is thread
__global__ void Comp_y(float* Y,float* refw,float* b, int w_row, int w_col)
{
    if (threadIdx.x < w_row && blockIdx.x < 1)
    {
        Y[threadIdx.x] = 0;
        for (int i = 0; i < w_col; i++)
        {
            Y[threadIdx.x] += refw[threadIdx.x * w_col + i];
        }
        Y[threadIdx.x] += b[threadIdx.x];


        //use of ReLU
        if (Y[threadIdx.x] > 1)
        {
            Y[threadIdx.x] = 1;
        }
        else if(Y[threadIdx.x] < 0)
        {
            Y[threadIdx.x] = 0;
        }
        //use of Sigmoid
        //Y[threadIdx.x] = (1.0f / (1.0f + powf(2.71828f, -(Y[threadIdx.x]))));
    }
    
}

int c_main(E_Layer *EL)
{
    E_Layer devEL;

    float real_cost = 0;
    float calc_cost = 0;


    /*-----memeory allocation ---------*/
    cudaMalloc((void**)&devEL.LF.X, 784 * 5 * sizeof(float));
    cudaCheckErrors("Elayer allocation error");
    cudaMalloc((void**)&devEL.LF.w, 784 * 20 * sizeof(float));
    cudaMalloc((void**)&devEL.LF.refw, 784 * 20 * sizeof(float));
    cudaMalloc((void**)&devEL.LF.b, 20 * sizeof(float));
    cudaMalloc((void**)&devEL.LF.Y, 20 * sizeof(float));

    // devHF[0] mem allocation
    //cudaMalloc((void**)&devLF.X, 784 * sizeof(float));
    cudaMalloc((void**)&devEL.LH[0].w, 20 * 20 * sizeof(float));
    cudaCheckErrors("Elayer allocation error");
    cudaMalloc((void**)&devEL.LH[0].refw, 20 * 20 * sizeof(float));
    cudaMalloc((void**)&devEL.LH[0].b, 20 * sizeof(float));
    cudaMalloc((void**)&devEL.LH[0].Y, 20 * sizeof(float));

    // devHF[0] mem allocation
    //cudaMalloc((void**)&devLF.X, 784 * sizeof(float));
    cudaMalloc((void**)&devEL.LH[1].w, 20 * 20 * sizeof(float));
    cudaCheckErrors("Elayer allocation error");
    cudaMalloc((void**)&devEL.LH[1].refw, 20 * 20 * sizeof(float));
    cudaMalloc((void**)&devEL.LH[1].b, 20 * sizeof(float));
    cudaMalloc((void**)&devEL.LH[1].Y, 20 * sizeof(float));

    //devLL mem allocation
    cudaMalloc((void**)&devEL.LL.w, 20 * 10 * sizeof(float));
    cudaCheckErrors("Elayer allocation error");
    cudaMalloc((void**)&devEL.LL.refw, 20 * 10  * sizeof(float));
    cudaMalloc((void**)&devEL.LL.b, 10 * sizeof(float));
    cudaMalloc((void**)&devEL.LL.Y, 10 * sizeof(float));

    cudaMemcpy(devEL.LF.X, EL->io.X, 784 *5 * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("copy failed from io failed");

    cudaMemcpy(devEL.LF.w, EL->LF.w, 784 * 20 * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("copy failed LF");
    cudaMemcpy(devEL.LF.b, EL->LF.b, 20 * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("copy failed LF");
    cudaMemcpy(devEL.LF.Y, EL->LF.Y, 20 * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("copy failed LF");

    cudaMemcpy(devEL.LH[0].w, EL->LH[0].w, 20 * 20 * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("copy failed LH");
    cudaMemcpy(devEL.LH[0].b, EL->LH[0].b, 20 * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("copy failed LH");
    cudaMemcpy(devEL.LH[0].Y, EL->LH[0].Y, 20 * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("copy failed LH");

    cudaMemcpy(devEL.LH[1].w, EL->LH[1].w, 20 * 20 * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("copy failed LH");
    cudaMemcpy(devEL.LH[1].b, EL->LH[1].b, 20 * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("copy failed LH");
    cudaMemcpy(devEL.LH[1].Y, EL->LH[1].Y, 20 * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("copy failed LH");

    cudaMemcpy(devEL.LL.w, EL->LL.w, 20 * 10 * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("copy failed LL");
    cudaMemcpy(devEL.LL.b, EL->LL.b, 10 * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("copy failed LL");
    cudaMemcpy(devEL.LL.Y, EL->LL.Y, 10 * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("copy failed LL");


    //Y_ref is filled with output of io.x and weights and bias
    for (int i = 0; i < 5; i++)
    {

        Comp_Minimap(devEL,i);

        cudaMemcpy(EL->io.Y_ref, devEL.LL.Y, 10 * sizeof(float), cudaMemcpyDeviceToHost);
        cudaCheckErrors("cpy failed");

        real_cost += EL->SqrMean(i);
    }
    real_cost /= 5;

    printf("real_cost is %f \n", real_cost);
    

    calc_dw_db(EL, devEL, EL->LL.w, devEL.LL.w, EL->LL.dw, real_cost, 20 * 10);
    calc_dw_db(EL, devEL, EL->LL.b, devEL.LL.b,EL->LL.db,real_cost, 10);
    printf(" LL completed \n");
    calc_dw_db(EL, devEL, EL->LH[0].w, devEL.LH[0].w, EL->LH[0].dw, real_cost, 20 * 20);
    calc_dw_db(EL, devEL, EL->LH[0].b, devEL.LH[0].b, EL->LH[0].db, real_cost, 20);
    printf(" LH[0] completed \n");
    calc_dw_db(EL, devEL, EL->LH[1].w, devEL.LH[1].w, EL->LH[1].dw, real_cost, 20 * 20);
    calc_dw_db(EL, devEL, EL->LH[1].b, devEL.LH[1].b, EL->LH[1].db, real_cost, 20);
    printf(" LH[1] completed \n");
    calc_dw_db(EL, devEL, EL->LF.w, devEL.LF.w, EL->LF.dw, real_cost, 784 * 20);
    calc_dw_db(EL, devEL, EL->LF.b, devEL.LF.b, EL->LF.db, real_cost, 20);
    printf(" LF completed \n");



    cudaMemcpy(EL->LH[0].Y, devEL.LH[0].Y, 20 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckErrors("cpy failed");

    //cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(EL->LF.Y, devEL.LF.Y, 20 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckErrors("cpy failed");

    cudaMemcpy(EL->io.Y, devEL.LL.Y, 10 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckErrors("cpy failed");



    cudaFree(devEL.LF.X);
    cudaFree(devEL.LF.b);
    //cudaFree(&devLF.db);
    cudaFree(devEL.LF.w);
    cudaFree(devEL.LF.refw);
    //cudaFree(&devLF.dw);
    cudaFree(devEL.LF.Y);

    cudaFree(devEL.LH[0].b);
    //cudaFree(&devLH[0].db);
    cudaFree(devEL.LH[0].w);
    cudaFree(devEL.LH[0].refw);
    //cudaFree(&devLH[0].dw);
    cudaFree(devEL.LH[0].Y);

    cudaFree(devEL.LH[1].b);
    //cudaFree(&devLH[1].db);
    cudaFree(devEL.LH[1].w);
    cudaFree(devEL.LH[1].refw);
    //cudaFree(&devLH[1].dw);
    cudaFree(devEL.LH[1].Y);

    cudaFree(devEL.LL.b);
    //cudaFree(&devLL.db);
    cudaFree(devEL.LL.w);
    cudaFree(devEL.LL.refw);
    //cudaFree(&devLL.dw);
    cudaFree(devEL.LL.Y);
    cudaCheckErrors("free mem failed");

    printf("-- cuda completed --\n");



    /*-------end last layer output coding-------*/






    //const int arraySize = 5;
    //const int a[arraySize] = { 1, 2, 3, 4, 5 };
    //const int b[arraySize] = { 10, 20, 30, 40, 50 };
    //int c[arraySize] = { 0 };

    //// Add vectors in parallel.
    //cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "addWithCuda failed!");
    //    return 1;
    //}

    //printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
    //    c[0], c[1], c[2], c[3], c[4]);

    //// cudaDeviceReset must be called before exiting in order for profiling and
    //// tracing tools such as Nsight and Visual Profiler to show complete traces.
    //cudaStatus = cudaDeviceReset();
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "cudaDeviceReset failed!");
    //    return 1;
    //}

    return 0;
}



void calc_dw_db(E_Layer* EL, E_Layer& devEL, float* ds,float* devds,float* dw, float real_cost,int arrsize)
{
    float calc_cost;
    for (int j = 0; j < arrsize; j++)
    {
        //EL->LL.w[j] += 0.1f;
        ds[j] += 0.001f;
        //cudaMemcpy(&devEL.LL.w[j], &EL->LL.w[j], sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(&devds[j], &ds[j], sizeof(float), cudaMemcpyHostToDevice);


        calc_cost = 0;

        for (int i = 0; i < 5; i++)
        {

            Comp_Minimap(devEL, i);

            cudaMemcpy(EL->io.Y_ref, devEL.LL.Y, 10 * sizeof(float), cudaMemcpyDeviceToHost);
            cudaCheckErrors("cpy failed");

            calc_cost += EL->SqrMean(i);

        }
        calc_cost /= 5;

        if (arrsize > 784)
        {
            //printf("cost %.4f,%4f,, ",calc_cost, 10 * (real_cost - calc_cost));

            dw[j] = 0.1f * (real_cost - calc_cost)/0.001f;
        }
        else
        {
            dw[j] = 1.f*(real_cost - calc_cost)/0.001f;
        }


        ds[j] -= 0.001f;
        cudaMemcpy(&devds[j], &ds[j], sizeof(float), cudaMemcpyHostToDevice);
    }
    for (int i = 0; i < arrsize; i++)
    {
        ds[i] += (5*dw[i]);
    }
}

int Comp_Minimap(E_Layer& devEL, int i)
{
    Sqs_X << <1, 784 >> > (&devEL.LF.X[784 * i]);

    //compute io --> layer first out(devLF.Y)
    Dot_w_X << <20, 784 >> > (devEL.LF.refw, devEL.LF.w, &devEL.LF.X[784 * i], 20, 784);
    cudaDeviceSynchronize();
    //cudaCheckErrors("synchronize failed LF");

    Comp_y << <1, 20 >> > (devEL.LF.Y, devEL.LF.refw, devEL.LF.b, 20, 784);
    cudaDeviceSynchronize();
    //cudaCheckErrors("synchronize failed LF");



    //compute devLF.Y --> devLH[0].Y
    Dot_w_X << <20, 20 >> > (devEL.LH[0].refw, devEL.LH[0].w, devEL.LF.Y, 20, 20);
    //cudaCheckErrors("fun failed LH[0]");
    cudaDeviceSynchronize();
    //cudaCheckErrors("synchronize failed LH[0]");

    Comp_y << <1, 20 >> > (devEL.LH[0].Y, devEL.LH[0].refw, devEL.LH[0].b, 20, 20);
   // cudaCheckErrors("fun failed LH[0]");
    cudaDeviceSynchronize();
    //cudaCheckErrors("synchronize failed LH[0]");


    //compute devLH[0].Y --> devLH[1].Y
    Dot_w_X << <20, 20 >> > (devEL.LH[1].refw, devEL.LH[1].w, devEL.LH[0].Y, 20, 20);
    //cudaCheckErrors("fun failed LH[1]");
    cudaDeviceSynchronize();
    //cudaCheckErrors("synchronize failed LH[1]");

    Comp_y << <1, 20 >> > (devEL.LH[1].Y, devEL.LH[1].refw, devEL.LH[1].b, 20, 20);
    //cudaCheckErrors("fun failed LH[1]");
    cudaDeviceSynchronize();
    //cudaCheckErrors("synchronize failed LH[1]");





    //compute devLH[1].Y --> devLL.Y
    Dot_w_X << <10, 20 >> > (devEL.LL.refw, devEL.LL.w, devEL.LH[1].Y, 10, 20);
    //cudaCheckErrors("fun failed LL");
    cudaDeviceSynchronize();
    //cudaCheckErrors("synchronize failed LL");

    Comp_y << <1, 10 >> > (devEL.LL.Y, devEL.LL.refw, devEL.LL.b, 10, 20);
    //cudaCheckErrors("fun failed[10]");
    cudaDeviceSynchronize();
    cudaCheckErrors("synchronize failed[10]");
}







int check(E_Layer* EL)
{
    Layer_f devLF;
    Layer_h devLH[2];
    Layer_l devLL;

    /*-----memeory allocation ---------*/
    cudaMalloc((void**)&devLF.X, 784 * sizeof(float));
    cudaCheckErrors("Elayer allocation error");
    cudaMalloc((void**)&devLF.w, 784 * 20 * sizeof(float));
    cudaMalloc((void**)&devLF.refw, 784 * 20 * sizeof(float));
    cudaMalloc((void**)&devLF.b, 20 * sizeof(float));
    cudaMalloc((void**)&devLF.Y, 20 * sizeof(float));

    // devHF[0] mem allocation
    //cudaMalloc((void**)&devLF.X, 784 * sizeof(float));
    cudaMalloc((void**)&devLH[0].w, 20 * 20 * sizeof(float));
    cudaCheckErrors("Elayer allocation error");
    cudaMalloc((void**)&devLH[0].refw, 20 * 20 * sizeof(float));
    cudaMalloc((void**)&devLH[0].b, 20 * sizeof(float));
    cudaMalloc((void**)&devLH[0].Y, 20 * sizeof(float));

    // devHF[0] mem allocation
    //cudaMalloc((void**)&devLF.X, 784 * sizeof(float));
    cudaMalloc((void**)&devLH[1].w, 20 * 20 * sizeof(float));
    cudaCheckErrors("Elayer allocation error");
    cudaMalloc((void**)&devLH[1].refw, 20 * 20 * sizeof(float));
    cudaMalloc((void**)&devLH[1].b, 20 * sizeof(float));
    cudaMalloc((void**)&devLH[1].Y, 20 * sizeof(float));

    //devLL mem allocation
    cudaMalloc((void**)&devLL.w, 20 * 10 * sizeof(float));
    cudaCheckErrors("Elayer allocation error");
    cudaMalloc((void**)&devLL.refw, 20 * 10 * sizeof(float));
    cudaMalloc((void**)&devLL.b, 10 * sizeof(float));
    cudaMalloc((void**)&devLL.Y, 10 * sizeof(float));


    cudaMemcpy(devLF.w, EL->LF.w, 784 * 20 * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("copy failed LF");
    cudaMemcpy(devLF.b, EL->LF.b, 20 * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("copy failed LF");
    cudaMemcpy(devLF.Y, EL->LF.Y, 20 * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("copy failed LF");

    cudaMemcpy(devLH[0].w, EL->LH[0].w, 20 * 20 * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("copy failed LH");
    cudaMemcpy(devLH[0].b, EL->LH[0].b, 20 * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("copy failed LH");
    cudaMemcpy(devLH[0].Y, EL->LH[0].Y, 20 * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("copy failed LH");

    cudaMemcpy(devLH[1].w, EL->LH[1].w, 20 * 20 * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("copy failed LH");
    cudaMemcpy(devLH[1].b, EL->LH[1].b, 20 * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("copy failed LH");
    cudaMemcpy(devLH[1].Y, EL->LH[1].Y, 20 * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("copy failed LH");

    cudaMemcpy(devLL.w, EL->LL.w, 20 * 10 * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("copy failed LL");
    cudaMemcpy(devLL.b, EL->LL.b, 10 * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("copy failed LL");
    cudaMemcpy(devLL.Y, EL->LL.Y, 10 * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("copy failed LL");



    /*--------mini map loop ---------*/

        cudaMemcpy(devLF.X, EL->io.User_X, 784 * sizeof(float), cudaMemcpyHostToDevice);
        cudaCheckErrors("copy failed from io failed");



        /*-------upto last layer output coding-------*/
        Sqs_X << <1, 784 >> > (devLF.X);
        cudaMemcpy(EL->LF.Y, devLF.X, 20 * sizeof(float), cudaMemcpyDeviceToHost);


        //compute io --> layer first out(devLF.Y)
        Dot_w_X << <20, 784 >> > (devLF.refw, devLF.w, devLF.X, 20, 784);
        cudaDeviceSynchronize();
        cudaCheckErrors("synchronize failed LF");

        Comp_y << <1, 20 >> > (devLF.Y, devLF.refw, devLF.b, 20, 784);
        cudaDeviceSynchronize();
        cudaCheckErrors("synchronize failed LF");





        //compute devLF.Y --> devLH[0].Y
        Dot_w_X << <20, 20 >> > (devLH[0].refw, devLH[0].w, devLF.Y, 20, 20);
        cudaCheckErrors("fun failed LH[0]");
        cudaDeviceSynchronize();
        cudaCheckErrors("synchronize failed LH[0]");

        Comp_y << <1, 20 >> > (devLH[0].Y, devLH[0].refw, devLH[0].b, 20, 20);
        cudaCheckErrors("fun failed LH[0]");
        cudaDeviceSynchronize();
        cudaCheckErrors("synchronize failed LH[0]");


        //compute devLH[0].Y --> devLH[1].Y
        Dot_w_X << <20, 20 >> > (devLH[1].refw, devLH[1].w, devLH[0].Y, 20, 20);
        cudaCheckErrors("fun failed LH[1]");
        cudaDeviceSynchronize();
        cudaCheckErrors("synchronize failed LH[1]");

        Comp_y << <1, 20 >> > (devLH[1].Y, devLH[1].refw, devLH[1].b, 20, 20);
        cudaCheckErrors("fun failed LH[1]");
        cudaDeviceSynchronize();
        cudaCheckErrors("synchronize failed LH[1]");





        //compute devLH[1].Y --> devLL.Y
        Dot_w_X << <10, 20 >> > (devLL.refw, devLL.w, devLH[1].Y, 10, 20);
        cudaCheckErrors("fun failed LL");
        cudaDeviceSynchronize();
        cudaCheckErrors("synchronize failed LL");

        Comp_y << <1, 10 >> > (devLL.Y, devLL.refw, devLL.b, 10, 20);
        cudaCheckErrors("fun failed[10]");
        cudaDeviceSynchronize();
        cudaCheckErrors("synchronize failed[10]");


    
    //cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(EL->LF.Y, devLF.Y, 20 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckErrors("cpy failed");

    cudaMemcpy(EL->LH[0].Y, devLH[0].Y, 20 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckErrors("cpy failed");

    cudaMemcpy(EL->LH[1].Y, devLH[1].Y, 20 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckErrors("cpy failed");

    cudaMemcpy(EL->LL.Y, devLL.Y, 10 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckErrors("cpy failed");



    cudaFree(devLF.X);
    cudaFree(devLF.b);
    //cudaFree(&devLF.db);
    cudaFree(devLF.w);
    cudaFree(devLF.refw);
    //cudaFree(&devLF.dw);
    cudaFree(devLF.Y);

    cudaFree(devLH[0].b);
    //cudaFree(&devLH[0].db);
    cudaFree(devLH[0].w);
    cudaFree(devLH[0].refw);
    //cudaFree(&devLH[0].dw);
    cudaFree(devLH[0].Y);

    cudaFree(devLH[1].b);
    //cudaFree(&devLH[1].db);
    cudaFree(devLH[1].w);
    cudaFree(devLH[1].refw);
    //cudaFree(&devLH[1].dw);
    cudaFree(devLH[1].Y);

    cudaFree(devLL.b);
    //cudaFree(&devLL.db);
    cudaFree(devLL.w);
    cudaFree(devLL.refw);
    //cudaFree(&devLL.dw);
    cudaFree(devLL.Y);
    cudaCheckErrors("free mem failed");

    printf("-- cuda completed for check --\n");



    /*-------end last layer output coding-------*/


    return 0;
}


// Helper function for using CUDA to add vectors in parallel.
//cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
//{
//    int *dev_a = 0;
//    int *dev_b = 0;
//    int *dev_c = 0;
//    cudaError_t cudaStatus;
//
//    // Choose which GPU to run on, change this on a multi-GPU system.
//    cudaStatus = cudaSetDevice(0);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
//        goto Error;
//    }
//
//    // Allocate GPU buffers for three vectors (two input, one output)    .
//    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//
//    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//
//    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//
//    // Copy input vectors from host memory to GPU buffers.
//    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//
//    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//
//    // Launch a kernel on the GPU with one thread for each element.
//    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);
//
//    // Check for any errors launching the kernel
//    cudaStatus = cudaGetLastError();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
//        goto Error;
//    }
//    
//    // cudaDeviceSynchronize waits for the kernel to finish, and returns
//    // any errors encountered during the launch.
//    cudaStatus = cudaDeviceSynchronize();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
//        goto Error;
//    }
//
//    // Copy output vector from GPU buffer to host memory.
//    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//
//Error:
//    cudaFree(dev_c);
//    cudaFree(dev_a);
//    cudaFree(dev_b);
//    
//    return cudaStatus;
//}
