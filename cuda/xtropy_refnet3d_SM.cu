
__device__ int get_index_x (int ncols, int index ) { 
  if (index == -1)  {
    index = blockDim.x * blockIdx.x + threadIdx.x;
  } else {
    index += gridDim.x;
  }

  if (index >= ncols) index = -1;

  return index;
}

__device__ int get_index_y (int nrows, int index ) { 
  if (index == -1)  {
    index = blockDim.y * blockIdx.y + threadIdx.y;
  } else {
    index += gridDim.y;
  }

  if (index >= nrows) index = -1;

  return index;
}

__device__ int get_index_z (int depth, int index ) { 
  if (index == -1)  {
    index = blockDim.z * blockIdx.z + threadIdx.z;
  } else {
    index += gridDim.z;
  }

  if (index >= depth) index = -1;

  return index;
}


__device__ float Gaussian (float x, float y, float sigma) {
  return (1/sqrt(2*M_PI*sigma)) * exp((-pow(x - y,2)) / (2*pow(sigma,2)));
}

__device__ float Gaussian_prime (float x, float y, float sigma) {
  return ( -(x - y) / ((pow(sigma,3))*sqrt(2*M_PI)) ) *  exp((-pow(x - y,2)) / (2*pow(sigma,2)));
}

#define BLOCKDIM 32

__global__ void CC_reduction ( float *redOut, const float *xline, const float *yline, const float *wxline, const float *wyline, int mvalue, float sigma, int linesize) {

  extern __shared__ float gSM[];

  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x + tid;
  unsigned int blockSize = blockDim.x;
  unsigned int gridSize = gridDim.x;

  while (i < linesize) {
    if (i - mvalue < 0 || i - mvalue > linesize) {
      gSM[tid] = 0;
      continue;
    }

    gSM[tid] = Gaussian(xline[i] * wxline[i], yline[i-mvalue] * wyline[i-mvalue], sigma);    
    i += gridSize;
  }

  __syncthreads();

  if (blockSize >= 512) { if (tid < 256) { gSM[tid] += gSM[tid + 256]; } __syncthreads(); }
  if (blockSize >= 256) { if (tid < 128) { gSM[tid] += gSM[tid + 128]; } __syncthreads(); }
  if (blockSize >= 128) { if (tid < 64) { gSM[tid] += gSM[tid + 64]; } __syncthreads(); }

  if (tid < 32) {
    if (blockSize >= 64) gSM[tid] += gSM[tid + 32];
    if (blockSize >= 32) gSM[tid] += gSM[tid + 16];
    if (blockSize >= 16) gSM[tid] += gSM[tid + 8];
    if (blockSize >= 8) gSM[tid] += gSM[tid + 4];
    if (blockSize >= 4) gSM[tid] += gSM[tid + 2];
    if (blockSize >= 2) gSM[tid] += gSM[tid + 1];
  }

  if (tid == 0) redOut[blockIdx.x] = gSM[0];

}

__global__ void ACm( float *out, const float *x, const float *y, const float *wx, const float *wy,  const int *marray, float sigma, int msize, int ncols, int nrows, int depth) {

  int idm = get_index_x(msize, -1);
  int idy = get_index_y(nrows, -1);
  int idz = get_index_z(depth, -1);

  float *redOut;

  redOut = (float*) malloc((ncols/BLOCKDIM) * sizeof(float));
  

  while(idz >= 0) {
    while (idy >= 0) {
      while(idm >= 0) {
        int m = marray[idm];

        float *xp = (float*) &x[idy*ncols + idz*nrows*ncols];
        float *yp = (float*) &y[idy*ncols + idz*nrows*ncols];
        float *wxp = (float*) &wx[idm*ncols];
        float *wyp = (float*) &wy[idm*ncols];
       

        CC_reduction<<<ncols/BLOCKDIM,BLOCKDIM>>>(redOut, xp, yp, wxp, wyp, m, sigma, ncols);

        float sum = 0;
        for (int i = 0; i < ncols/BLOCKDIM; i++) {
          sum += redOut[i];
        }

        out[idm + idy*msize + idz*nrows*msize] = sum * (1 / ncols - abs(m));

        idm = get_index_x(msize, idm);
      }
      idy = get_index_y (nrows, idy);
    }
    idz = get_index_z (depth, idz);
  }
}



__global__ void ACm_prime( float *out, const float *x, const float *y, const float *wx, const float *wy,  const int *marray, float sigma, int msize, int ncols, int nrows, int depth) {
  float sum = 0;
  int i = 0;
  int idm = get_index_x(msize, -1);
  int idy = get_index_y(nrows, -1);
  int idz = get_index_z(depth, -1);
  int m = marray[idm];
  int cn = 1;

  while(idz >= 0) {
    while (idy >= 0) {
      while(idm >= 0) {
        sum = 0;
        cn = 1;

        for (i=m; i < ncols; i++) {
          if (i < 0 || i-m > ncols) {
            continue;
          }

          sum += Gaussian_prime (x[i + idy*ncols + idz*nrows*ncols] * wx[i + idm*ncols], y[abs(i-m) + idy*ncols + idz*nrows*ncols] * wy[i-m + idm*ncols], sigma);

          cn = cn + 1;
        }

        out[idm + idy*msize + idz*nrows*msize] = ( 1/((float)cn) ) * sum;

        idm = get_index_x(msize, idm);
        m = marray[idm];

      }
      idy = get_index_y (nrows, idy);
    }
    idz = get_index_z (depth, idz);
  }
}


