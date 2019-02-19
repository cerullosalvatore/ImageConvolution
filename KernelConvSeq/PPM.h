#ifndef KERNELCONVSEQ_PPM_H
#define KERNELCONVSEQ_PPM_H

#include "Image.h"

Image* PPM_import(const char *filename);
bool PPM_export(const char *filename, Image* img);


#endif //KERNELCONVSEQ_PPM_H
