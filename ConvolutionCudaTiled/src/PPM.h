/*
 * PPM.h
 *
 *  Created on: 10/feb/2019
 *      Author: salvatore
 */

#ifndef PPM_H_
#define PPM_H_

#include "Image.h"

Image_t* PPM_import(const char *filename);
bool PPM_export(const char *filename, Image_t* img);

#endif /* PPM_H_ */
