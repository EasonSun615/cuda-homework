#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mat_mul.h"

int main(int argc, char **argv){

  int M_row = 256;   // default value
  int width = 512;   
  int N_col = 1024; 
  if (argc > 3) {
    width = atoi(argv[2]); // user-specified value
    M_row = atoi(argv[1]);
    N_col = atoi(argv[3]);
  }

  printf("\nMatrix M:   row: %d,  col: %d.\n", M_row, width);
  printf("Matrix N:   row: %d,  col: %d.\n",width, N_col);

  srand(time(NULL));
  float* M = rand_mat(M_row, width);
  float* N = rand_mat(width, N_col);

  float* cpu_P = raw_mat(M_row, N_col);
  float* gpu_P = raw_mat(M_row, N_col);

  long long cpu_start_time = start_timer();
  cpu_mat_mul(M, N, cpu_P, M_row, width, N_col);
  long long cpu_time = stop_timer(cpu_start_time, "CPU");

  long long gpu_start_time = start_timer();
  gpu_mat_mul(M, N, gpu_P, M_row, width, N_col);
  long long gpu_time = stop_timer(gpu_start_time, "GPU");


  // Check the correctness of the GPU results
  int num_wrong = 0;
  for (int i = 0; i < M_row * N_col; i++) {
    if (fabs(cpu_P[i] - gpu_P[i]) > 0.000001) num_wrong++;
  }
	
  // Report the correctness results
  if (num_wrong) printf("GPU %d / %d values incorrect\n", num_wrong, N);
  else           printf("GPU all values correct\n");

}
