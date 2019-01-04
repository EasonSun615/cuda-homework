#include<cuda.h>
#include<stdio.h>
#include<math.h>

#define TILE_WIDTH 16 

extern "C" void gpu_mat_mul(float* h_M, float* h_N, float* h_P, int M_row, int width, int N_col);

__global__
void gpu_mat_mul_kernel(float* M, float* N, float* P, int M_row, int width, int N_col){

  __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
  __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

  int bx = blockIdx.x; 
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // Identify the row and column of the P element to work on
  // Each thread works on an element of P
  int Row = by * TILE_WIDTH + ty;
  int Col = bx * TILE_WIDTH + tx;

 // if(Row<M_row && Col<N_col){
  int phase_num = ceil(width / (float)TILE_WIDTH);
  float Pvalue = 0;
  // Each thread loads 'Row'th row of M and 'Col'th column of N
  for (int ph = 0; ph < phase_num; ++ph) {

    if((ph*TILE_WIDTH+tx) < width && Row<M_row )     
      Mds[ty][tx] = M[Row * width + ph * TILE_WIDTH + tx];   
    else
      Mds[ty][tx] = 0;
    if((ph*TILE_WIDTH+ty) <width && Col<N_col)
   	 Nds[ty][tx] = N[(ph * TILE_WIDTH + ty) * N_col + Col];
    else 
	Nds[ty][tx] = 0;

    __syncthreads();

    for (int k = 0; k < TILE_WIDTH; ++k) { 
      Pvalue += Mds[ty][k] * Nds[k][tx];
    }

    __syncthreads();
  }
  if(Row<M_row && Col<N_col)
  P[Row * N_col + Col] = Pvalue;
  //}
}

void gpu_mat_mul(float* h_M, float* h_N, float* h_P, int M_row, int width, int N_col) {
  float *d_M, *d_N, *d_P;

  size_t size_of_float = sizeof(float);
  size_t size_M = M_row * width * size_of_float;
  size_t size_N = width * N_col * size_of_float;
  size_t size_P = M_row * N_col * size_of_float;

  cudaMalloc((void**)&d_M, size_M);
  cudaMalloc((void**)&d_N, size_N);
  cudaMalloc((void**)&d_P, size_P);
    
  cudaMemcpy(d_M, h_M, size_M, cudaMemcpyHostToDevice);
  cudaMemcpy(d_N, h_N, size_N, cudaMemcpyHostToDevice);

  cudaEvent_t start, stop;
  float elapsed_time = 0.0;
    
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  dim3 grid_dim(ceil(N_col/ (float)(TILE_WIDTH)), ceil(M_row/ (float)(TILE_WIDTH)), 1);
  dim3 block_dim(TILE_WIDTH, TILE_WIDTH, 1);
  gpu_mat_mul_kernel<<<grid_dim, block_dim>>>(d_M, d_N, d_P, M_row, width, N_col);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  cudaMemcpy(h_P, d_P, size_P, cudaMemcpyDeviceToHost);
    
  // Free device memory for M, N, P
  cudaFree(d_M);
  cudaFree(d_N);
  cudaFree(d_P);

  cudaEventElapsedTime(&elapsed_time, start, stop);
    
  printf("  grid  dim:  %d, %d, %d.\n", grid_dim.x, grid_dim.y, grid_dim.z);
  printf("  block dim: %d, %d, %d.\n", block_dim.x, block_dim.y, block_dim.z);
  printf("  kernel time: %.5f sec\n", elapsed_time / 1000);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

