#include <iostream>
#include <omp.h>
#include <sys/time.h>
#include "Image.h"
#include "PPM.h"
#include "Utils.h"


#define clamp(x) (_min(_max((x), 0.0), 1.0))
#define Mask_width 3
#define Mask_radius Mask_width / 2

int main() {
    int imageChannels;
    int imageWidth;
    int imageHeight;

    Image* inputImage;
    Image* outputImage;

    float* dataImageInput;
    float* dataImageOutput;


    const char* pathIn = "/home/salvatore/Scrivania/ImgPPM/7.ppm";        //Path inputImage
    const char* pathOut = "/home/salvatore/Scrivania/ImgPPM/out.ppm";     //Path outputImage

    //3x3 Mask - To use it, set Mask_width =3
    //Sharpen
    //float dataMask[Mask_width*Mask_width] = {0,-1, 0, -1, 5, -1, 0, -1, 0};
    //Edge Detection--
    //float dataMask[Mask_width*Mask_width] = {0, 1, 0, 1, -4, 1, 0, 1, 0};
    //Edge Detection
    float dataMask[Mask_width*Mask_width] = {-1,-1, -1, -1, 8, -1, -1, -1, -1};
    //Gaussian Blur 3x3
    //float dataMask[Mask_width*Mask_width] = {(float)1/16,(float)2/16, (float)1/16, (float)2/16, (float)4/16, (float)2/16, (float)1/16, (float)2/16, (float)1/16};

    //5x5 Mask - To use it, set Mask_width =5
    //Gaussian Blur 5x5
    //float dataMask[Mask_width*Mask_width] = {(float)1/256,(float)4/256,(float)6/256, (float)4/256, (float)1/256,(float)4/256,(float)16/256,(float)24/256, (float)16/256, (float)4/256, (float)6/256,(float)24/256,(float)36/256, (float)24/256, (float)6/256, (float)4/256,(float)16/256,(float)24/256, (float)16/256, (float)4/256, (float)1/256,(float)4/256,(float)6/256, (float)4/256, (float)1/256};
    //Unsharp Masking 5x5
    //float dataMask[Mask_width*Mask_width] = {(float)-1/256,(float)-4/256,(float)-6/256, (float)-4/256, (float)-1/256,(float)-4/256,(float)-16/256,(float)-24/256, (float)-16/256, (float)-4/256, (float)-6/256,(float)-24/256,(float)476/256, (float)-24/256, (float)-6/256, (float)-4/256,(float)-16/256,(float)-24/256, (float)-16/256, (float)-4/256, (float)-1/256,(float)-4/256,(float)-6/256, (float)-4/256, (float)-1/256};


    inputImage = PPM_import(pathIn);

    imageWidth = Image_getWidth(inputImage);
    imageHeight = Image_getHeight(inputImage);
    imageChannels = Image_getChannels(inputImage);

    outputImage = newImage(imageWidth, imageHeight, imageChannels);

    dataImageInput = Image_getData(inputImage);
    dataImageOutput = Image_getData(outputImage);

    float N_ds[Mask_width][Mask_width]; //Product Matrix

    timeval t1, t2;
    gettimeofday(&t1, NULL);
    //Loop for channels
    int yi, xi, k, tid;


    //Parallelization
    #pragma omp parallel for private(yi, xi, k, N_ds, tid) num_threads(omp_get_max_threads())
    //Loop for rows
    for(yi = 0; yi < imageHeight; yi++) {
        //Loop for columns
        for (xi = 0; xi < imageWidth; xi++) {
            //Loop for channels
            for (k=0; k<imageChannels; k++) {
                //Loop for Mask rows
                for (int ym = 0; ym < Mask_width; ym++) {
                    //Loop for columns rows
                    for (int xm = 0; xm < Mask_width; xm++) {
                        int xx = xi - Mask_radius + xm;     //Column position on the original Vector with offset
                        int yy = yi - Mask_radius + ym;     //Row position on the original Vector with offset
                        if (xx < 0 || xx >= imageWidth || yy >= imageHeight || yy < 0) {
                            N_ds[xm][ym] = 0;   //Set the ghost value to 0
                        } else {
                            N_ds[xm][ym] = dataMask[ym * Mask_width + xm] *
                                           dataImageInput[(yy * imageWidth + xx) * imageChannels + k];
                        }
                    }
                }

                float valPi = 0;    //Value of the pixel on the processed image
                //Loop for Mask's rows
                for (int y = 0; y < Mask_width; y++) {
                    //Loop for Mask's columns
                    for (int x = 0; x < Mask_width; x++) {
                        valPi += N_ds[y][x];    //Sum the value on valPi
                    }
                }

                if (yi < imageHeight && xi < imageWidth) {
                    dataImageOutput[(yi * imageWidth + xi) * imageChannels + k] = clamp(valPi);     //Assign the value to processed image
                }
            }
        }
    }
    gettimeofday(&t2, NULL);
    double elapsedTimeMs = (t2.tv_sec - t1.tv_sec) * 1000.0;        // sec to ms
    elapsedTimeMs += (t2.tv_usec - t1.tv_usec) / 1000.0;            // us to ms
    printf("Elaboration Time: %f milliseconds\n", (elapsedTimeMs));

    PPM_export(pathOut, outputImage);   //Create the new Image and write the result at the specified path
}


