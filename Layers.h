#ifndef LAYERS_H
#define LAYERS_H

void Conv2d(double* img_in, long in_channels, long out_channels,
            long width, long height, 
            double**** weights, double* bias, 
            long kernel_size, long padding, long stride,
            double* img_out) ;

void MaxPool2d(double* img_in, long in_channels, long width, long height,
               long pool, double* img_out);

void BatchNorm2d(double* img_in, double* gamma, double* beta,long in_channels, long width, long height, 
                 double* img_out);

void Dropout(double* img_in, long in_channels, long width, long height, long p,
             double* img_out);

void Linear(double* img_in, long size, long out_channels, double** weights, 
            double* bias, double* img_out);
void Relu(double* img_in, long in_channels, long width, long height,
                double* img_out);

#endif
