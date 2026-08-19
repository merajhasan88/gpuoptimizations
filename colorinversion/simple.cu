#include <cuda_runtime.h>

__global__ void invert_kernel(unsigned char* image, int width, int height) {
// image_input, image_output are device pointers (i.e. pointers to memory on the GPU)
int pixel = blockIdx.x * blockDim.x + threadIdx.x;
int total_pixels = width * height;
if(pixel<total_pixels){
int index =pixel*4;
image[index + 0]=255-image[index+0];
image[index + 1]=255-image[index+1];
image[index + 2]=255-image[index+2];

}
}
extern "C" void solve(unsigned char* image, int width, int height) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (width * height + threadsPerBlock - 1) / threadsPerBlock;

    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(image, width, height);
    cudaDeviceSynchronize();
}
