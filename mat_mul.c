void cpu_mat_mul(float* M, float* N, float* P, int M_row, int width, int N_col) {
  for(int i = 0; i < M_row; i++) {
    for(int j = 0; j < N_col; j++) {
      float sum = 0.0;
      for(int k = 0; k < width; k++) {
        sum += M[i * width + k] * N[k * N_col + j];
      }
      P[i * N_col + j] = sum;
    }
  }
}
