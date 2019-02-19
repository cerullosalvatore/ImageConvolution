#include "Image.h"
#include <iostream>

Image* newImage(int width, int height, int channels, float *data) {
    Image* img;

    img = (Image*) malloc(sizeof(Image));

    Image_setWidth(img, width);
    Image_setHeight(img, height);
    Image_setChannels(img, channels);
    Image_setPitch(img, width * channels);

    Image_setData(img, data);
    return img;
}

Image* newImage(int width, int height, int channels) {
    float *data = (float*) malloc(sizeof(float) * width * height * channels);
    return newImage(width, height, channels, data);
}

void Image_delete(Image* img) {
    if (img != NULL) {
        if (Image_getData(img) != NULL) {
            free(Image_getData(img));
        }
        free(img);
    }
}