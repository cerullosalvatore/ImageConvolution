#include "Image.h"
#include "PPM.h"
#include <cstdio>
#include <cassert>
#include <iostream>
#include <sys/time.h>

#define Mask_Width 5
#define Mask_Height 5
#define Mask_Radius (Mask_Width/2)
#define Thread_Block_Dim 16
#define Dim_Shared (Thread_Block_Dim + (Mask_Radius*2))

__constant__ float deviceDataMask[Mask_Height * Mask_Width];

__global__ void functionConv(float *N, float *P, int width, int height, int channels){

	__shared__ float N_ds[Thread_Block_Dim + Mask_Radius*2][Thread_Block_Dim + Mask_Radius*2];	//Product Matrix
		int y = (blockIdx.y * blockDim.y + threadIdx.y);	//Thread row identification
		int x = (blockIdx.x * blockDim.x + threadIdx.x);	//Thread column identification


		//Cycle inside the channels
		for(int k=0 ; k<channels;k++){
			//Case 0: UP_SX
			int xx = x - Mask_Radius;
			int yy = y - Mask_Radius;
			if(xx< 0 || yy < 0){
				N_ds[threadIdx.y][threadIdx.x] = 0;
			}else{
				N_ds[threadIdx.y][threadIdx.x] = N[(yy*width+xx)*channels+k];
			}

			//Case 1: UP_DX
			xx = x + Mask_Radius;
			yy = y - Mask_Radius;
			if(xx>=width-1 || yy < 0){
				N_ds[threadIdx.y][threadIdx.x + 2*Mask_Radius] = 0;
			}else{
				N_ds[threadIdx.y][threadIdx.x + 2*Mask_Radius] = N[(yy*width+xx)*channels+k];
			}

			//Case 2: DOWN_SX
			xx = x - Mask_Radius;
			yy = y + Mask_Radius;
			if(xx<0 || yy > height-1){
				N_ds[threadIdx.y + 2*Mask_Radius][threadIdx.x] = 0;
			}else{
				N_ds[threadIdx.y + 2*Mask_Radius][threadIdx.x] = N[(yy*width+xx)*channels+k];
			}

			//Case 2: DOWN_DX
			xx = x + Mask_Radius;
			yy = y + Mask_Radius;
			if(xx>width-1 || yy > height-1){
				N_ds[threadIdx.y + 2*Mask_Radius][threadIdx.x + 2*Mask_Radius] = 0;
			}else{
				N_ds[threadIdx.y + 2*Mask_Radius][threadIdx.x + 2*Mask_Radius] = N[(yy*width+xx)*channels+k];
			}

			__syncthreads();

			//Calculate the sum of the elements of the product matrix
			float valPi = 0;
			for(int i = -Mask_Radius ; i <= Mask_Radius ; i++){
				for(int j = -Mask_Radius ; j <= Mask_Radius; j++){
					valPi += N_ds[i+Mask_Radius+threadIdx.y][j+Mask_Radius+threadIdx.x]*deviceDataMask[(i+Mask_Radius)*Mask_Width+(j+Mask_Radius)];
				}
			}

			//Set the new value in the output matrix
			if(y < height && x < width){
				P[(y*width+x)*channels + k] = valPi;
			}
		}
		__syncthreads();
}

int main() {
	int imageChannels;
	int imageWidth;
	int imageHeight;
	int sizeAllocImage;
	int sizeAllocMask;

	Image_t* inputImage;
	Image_t* outputImage;

	float *hostDataImageInput;
	float *deviceDataImageInput;
	float *hostDataImageOutput;
	float *deviceDataImageOutput;
	//float hostDataMask[Mask_Height * Mask_Width] = {-1,-1,-1,-1,8,-1,-1,-1,-1};
	//float hostDataMask[Mask_Height * Mask_Width] = {0,-1,0,-1,5,-1,0,-1,0};
	float hostDataMask[Mask_Height * Mask_Width]={(float)1/256, (float)4/256, (float)6/256, (float)4/256,(float)1/256, (float)4/256, (float)16/256, (float)24/256, (float)16/256, (float)4/256, (float)6/256, (float)24/256, (float)36/256, (float)24/256, (float)6/256, (float)4/256, (float)16/256, (float)24/256, (float)16/256, (float)4/256, (float)1/256, (float)4/256, (float)1/256, (float)4/256, (float)1/256};

	const char* pathIn = "/home/salvatore/Scrivania/ImgPPM/1.ppm";
	const char* pathOut = "/home/salvatore/Scrivania/out.ppm";

	inputImage = PPM_import(pathIn);

	imageWidth = Image_getWidth(inputImage);
	imageHeight = Image_getHeight(inputImage);
	imageChannels = Image_getChannels(inputImage);

	sizeAllocImage = imageHeight * imageChannels * imageWidth * sizeof(float);
	sizeAllocMask = Mask_Width * Mask_Height * sizeof(float);

	outputImage = Image_new(imageWidth, imageHeight, imageChannels);

	hostDataImageInput = Image_getData(inputImage);
	hostDataImageOutput = Image_getData(outputImage);

	timeval t1, t2;

	cudaDeviceReset();

	//Device memory allocation
	cudaMalloc((void**) &deviceDataImageInput, sizeAllocImage);
	cudaMalloc((void**) &deviceDataImageOutput, sizeAllocImage);

	//Copying data from the host to the device
	cudaMemcpy(deviceDataImageInput, hostDataImageInput, sizeAllocImage, cudaMemcpyHostToDevice);
	cudaMemcpy(deviceDataImageOutput, hostDataImageOutput, sizeAllocImage, cudaMemcpyHostToDevice);

	//Copying in constant memory
	cudaMemcpyToSymbol(deviceDataMask, hostDataMask, sizeAllocMask);

	//Setting the dimensions of the grid and blocks
	dim3 dimGrid(ceil((float) imageWidth/Thread_Block_Dim),ceil((float) imageHeight/Thread_Block_Dim), 1);
	dim3 dimBlock(Thread_Block_Dim,Thread_Block_Dim, 1);

	gettimeofday(&t1,NULL);

	//Kernel call
	functionConv<<<dimGrid, dimBlock>>>(deviceDataImageInput, deviceDataImageOutput, imageWidth, imageHeight, imageChannels);
	cudaDeviceSynchronize();
	gettimeofday(&t2,NULL);

	//Copying data from the device to the host
	cudaMemcpy(hostDataImageOutput, deviceDataImageOutput, sizeAllocImage, cudaMemcpyDeviceToHost);

	PPM_export(pathOut, outputImage);

	//Free space from both the host and the device
	cudaFree(deviceDataImageInput);
	cudaFree(deviceDataImageOutput);
	cudaFree(deviceDataMask);
	Image_delete(outputImage);
	Image_delete(inputImage);

	double elapsedTime=(t2.tv_sec - t1.tv_sec) * 1000.0;
	elapsedTime += (t2.tv_usec - t1.tv_usec)/1000.0;
	printf("Elaborazione Completata in %f milliseconds", elapsedTime);

	return 0;
}
