#include "Image.h"
#include "PPM.h"
#include <cstdio>
#include <cassert>
#include <iostream>
#include <sys/time.h>

#define Mask_Width 5
#define Mask_Height 5
#define Mask_Radius Mask_Width/2
#define Thread_Block_Dim 16

__global__ void functionConv(float *N, float *M, float *P, int width, int height, int channels){
	float N_ds[Mask_Height][Mask_Width];				//Product Matrix
	int y = (blockIdx.y * blockDim.y + threadIdx.y);	//Thread row identification
	int x = (blockIdx.x * blockDim.x + threadIdx.x);	//Thread column identification

	//Cycle inside the channels
	for(int k=0 ; k<channels;k++){
		//Cycle to calculate the Product Matrix N_ds
		for(int ym = 0; ym < Mask_Height; ym++){
			for(int xm = 0 ; xm < Mask_Width; xm++){
				int xx = x - Mask_Radius + xm;	//Identification of the column in the original matrix
				int yy = y - Mask_Radius + ym;	//Identification of the row  in the original matrix
				if(xx<0 || xx >= width || yy >= height || yy < 0){
					N_ds[xm][ym] = 0;
				}else{
					N_ds[xm][ym] = M[ym*Mask_Width+xm]*N[(yy*width+xx)*channels+k];
				}
			}
		}

		//Calculate the sum of the elements of the product matrix
		float valPi = 0;
		for(int i = 0 ; i < Mask_Height ; i++){
			for(int j = 0 ; j < Mask_Width; j++){
				valPi += N_ds[i][j];
			}
		}

		//Set the new value in the output matrix
		if(y < height && x < width){
			P[(y*width+x)*channels + k] = valPi;
		}
	}
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
	float *deviceDataMask;
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
	cudaMalloc((void**) &deviceDataMask, sizeAllocMask);

	//Copying data from the host to the device
	cudaMemcpy(deviceDataImageInput, hostDataImageInput, sizeAllocImage, cudaMemcpyHostToDevice);
	cudaMemcpy(deviceDataImageOutput, hostDataImageOutput, sizeAllocImage, cudaMemcpyHostToDevice);
	cudaMemcpy(deviceDataMask, hostDataMask, sizeAllocMask, cudaMemcpyHostToDevice);

	//Setting the dimensions of the grid and blocks
	dim3 dimGrid(ceil((float) imageWidth/Thread_Block_Dim),ceil((float) imageHeight/Thread_Block_Dim), 1);
	dim3 dimBlock(Thread_Block_Dim,Thread_Block_Dim, 1);

	gettimeofday(&t1,NULL);

	//Kernel call
	functionConv<<<dimGrid, dimBlock>>>(deviceDataImageInput, deviceDataMask, deviceDataImageOutput, imageWidth, imageHeight, imageChannels);
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
	printf("Elaboration time: %f ms", elapsedTime);

	return 0;
}
