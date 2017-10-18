
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


__global__ void ACm( float *out, const float *x, const float *y, const float *wx, const float *wy,  const int *marray, float sigma, int msize, int ncols, int nrows, int depth) {
  float sum = 0;
  int i = 0;
  int idm = get_index_x(msize, -1);
  int idy = get_index_y(nrows, -1);
  int idz = get_index_z(depth, -1);
  int m = marray[idm];
  int cn = 1;

  int li;
  int lm;

  while(idz >= 0) {
    while (idy >= 0) {
      while(idm >= 0) {
        sum = 0;
        cn = 1;

        li = max(0,m);
        if (m < 0)
          lm = ncols + m ;
        else
          lm = ncols;


        for (i=li; i < lm; i++) {

          sum += Gaussian (x[i + idy*ncols + idz*nrows*ncols] * wx[idm + i*msize], y[abs(i-m) + idy*ncols + idz*nrows*ncols] * wy[idm + abs(i-m)*msize], sigma);

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

__global__ void ACm_prime( float *out, const float *x, const float *y, const float *wx, const float *wy,  const int *marray, float sigma, int msize, int ncols, int nrows, int depth) {
  float sum = 0;
  int i = 0;
  int idm = get_index_x(msize, -1);
  int idy = get_index_y(nrows, -1);
  int idz = get_index_z(depth, -1);
  int m = marray[idm];
  int cn = 1;

  int li;
  int lm;

  while(idz >= 0) {
    while (idy >= 0) {
      while(idm >= 0) {
        sum = 0;
        cn = 1;

        li = max(0,m);
        if (m < 0)
          lm = ncols + m ;
        else
          lm = ncols;

        for (i=li; i < lm; i++) {

          sum += Gaussian_prime (x[i + idy*ncols + idz*nrows*ncols] * wx[idm + i*msize], y[abs(i-m) + idy*ncols + idz*nrows*ncols] * wy[idm + abs(i-m)*msize], sigma);

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
