#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "Layers.h"

double* CNN(double* img_in, long in_channels, long width, long height,
            double*** conv_weights1, double* conv_bias1, double*** conv_weights2, double* conv_bias2,
            double*** conv_weights3, double* conv_bias3,
            double** linear_weights1, double *linear_bias1,
            double** linear_weights2, double *linear_bias2,
            double** gamma1, double *beta1,
            double** gamma2, double *beta2,
            double** gamma3, double *beta3,long num_classes)
{
    long kernel_size = 3;
    long padding = 1;
    long stride = 1;
    long pool = 2;

    // Layer 1: Conv(1,32)
    long out_channels_1 = 32;
    long width_1 = (width + 2*padding - kernel_size) / stride + 1;
    long height_1 = (height + 2*padding - kernel_size) / stride + 1;
    double* layer1_out = malloc(out_channels_1 * width_1 * height_1 * sizeof(double));
    
    if (layer1_out == NULL) {
        fprintf(stderr, "Memory allocation failed for layer1_out\n");
        return NULL;
    }
    
    Conv2d(img_in, in_channels, out_channels_1, width, height, conv_weights1, conv_bias1, kernel_size, padding, stride, layer1_out);
    BatchNorm2d(layer1_out, gamma1, beta1, out_channels_1, width_1, height_1, layer1_out);
    Relu(layer1_out, out_channels_1, width_1, height_1, layer1_out);
    
    // Pooling: 32x32 -> 16x16
    long width_pool1 = width_1 / pool;
    long height_pool1 = height_1 / pool;
    double* layer_pool1_out = malloc(out_channels_1 * width_pool1 * height_pool1 * sizeof(double));

    if (layer_pool1_out == NULL) {
        fprintf(stderr, "Memory allocation failed for layer_pool1_out\n");
        free(layer1_out);
        return NULL;
    }
    
    MaxPool2d(layer1_out, out_channels_1, width_1, height_1, pool, layer_pool1_out);
    Dropout(layer_pool1_out, out_channels_1, width_pool1, height_pool1, 0.25, layer_pool1_out);

    // Layer 2: Conv(32,64)
    long out_channels_2 = 64;
    long width_2 = (width_pool1 + 2*padding - kernel_size) / stride + 1;
    long height_2 = (height_pool1 + 2*padding - kernel_size) / stride + 1;
    double* layer2_out = malloc(out_channels_2 * width_2 * height_2 * sizeof(double));
    

    if (layer2_out == NULL) {
        fprintf(stderr, "Memory allocation failed for layer2_out\n");
        free(layer1_out);
        free(layer_pool1_out);
        return NULL;
    }
    
    Conv2d(layer_pool1_out, out_channels_1, out_channels_2, width_pool1, height_pool1, conv_weights2, conv_bias2, kernel_size, padding, stride, layer2_out);
    BatchNorm2d(layer2_out,gamma2, beta2, out_channels_2, width_2, height_2, layer2_out);
    Relu(layer2_out, out_channels_2, width_2, height_2, layer2_out);

    // Pooling: 16x16 -> 8x8
    long width_pool2 = width_2 / pool;
    long height_pool2 = height_2 / pool;
    double* layer_pool2_out = malloc(out_channels_2 * width_pool2 * height_pool2 * sizeof(double));
    
    if (layer_pool2_out == NULL) {
        fprintf(stderr, "Memory allocation failed for layer_pool2_out\n");
        free(layer1_out);
        free(layer_pool1_out);
        free(layer2_out);
        return NULL;
    }
    
    MaxPool2d(layer2_out, out_channels_2, width_2, height_2, pool, layer_pool2_out);
    Dropout(layer_pool2_out, out_channels_2, width_pool2, height_pool2, 0.25, layer_pool2_out);

    // Layer 3: Conv(64,128)
    long out_channels_3 = 128;
    long width_3 = (width_pool2 + 2*padding - kernel_size) / stride + 1;
    long height_3 = (height_pool2 + 2*padding - kernel_size) / stride + 1;
    double* layer3_out = malloc(out_channels_3 * width_3 * height_3 * sizeof(double));
    
    if (layer3_out == NULL) {
        fprintf(stderr, "Memory allocation failed for layer3_out\n");
        free(layer1_out);
        free(layer_pool1_out);
        free(layer2_out);
        free(layer_pool2_out);
        return NULL;
    }
    
    Conv2d(layer_pool2_out, out_channels_2, out_channels_3, width_pool2, height_pool2, conv_weights3, conv_bias3, kernel_size, padding, stride, layer3_out);
    BatchNorm2d(layer3_out, gamma3, beta3,out_channels_3, width_3, height_3, layer3_out);
    Relu(layer3_out, out_channels_3, width_3, height_3, layer3_out);

    // Pooling: 8x8 -> 4x4
    long width_pool3 = width_3 / pool;
    long height_pool3 = height_3 / pool;
    double* layer_pool3_out = malloc(out_channels_3 * width_pool3 * height_pool3 * sizeof(double));
    
    if (layer_pool3_out == NULL) {
        fprintf(stderr, "Memory allocation failed for layer_pool3_out\n");
        free(layer1_out);
        free(layer_pool1_out);
        free(layer2_out);
        free(layer_pool2_out);
        free(layer3_out);
        return NULL;
    }
    
    MaxPool2d(layer3_out, out_channels_3, width_3, height_3, pool, layer_pool3_out);

    // Dropout and Linear layers
    Dropout(layer_pool3_out, out_channels_3, width_pool3, height_pool3, 0.50, layer_pool3_out);

    // Flatten
    long flattened_size = out_channels_3 * width_pool3 * height_pool3;
    double* flattened = malloc(flattened_size * sizeof(double));
    
    if (flattened == NULL) {
        fprintf(stderr, "Memory allocation failed for flattened\n");
        free(layer1_out);
        free(layer_pool1_out);
        free(layer2_out);
        free(layer_pool2_out);
        free(layer3_out);
        free(layer_pool3_out);
        return NULL;
    }
    
    for (long i = 0; i < flattened_size; i++) {
        flattened[i] = layer_pool3_out[i];
    }

    // First Linear Layer
    long hidden_size = 512;
    double* hidden_layer = malloc(hidden_size * sizeof(double));
    
    if (hidden_layer == NULL) {
        fprintf(stderr, "Memory allocation failed for hidden_layer\n");
        free(layer1_out);
        free(layer_pool1_out);
        free(layer2_out);
        free(layer_pool2_out);
        free(layer3_out);
        free(layer_pool3_out);
        free(flattened);
        return NULL;
    }
    
    Linear(flattened, flattened_size, hidden_size, linear_weights1, linear_bias1, hidden_layer);
    Relu(hidden_layer, 1, hidden_size, 1, hidden_layer);

    // Dropout on hidden layer
    Dropout(hidden_layer, 1, hidden_size, 1, 0.50, hidden_layer);

    // Final Linear Layer for classification
    double* output = malloc(num_classes * sizeof(double));
    
    if (output == NULL) {
        fprintf(stderr, "Memory allocation failed for output\n");
        free(layer1_out);
        free(layer_pool1_out);
        free(layer2_out);
        free(layer_pool2_out);
        free(layer3_out);
        free(layer_pool3_out);
        free(flattened);
        free(hidden_layer);
        return NULL;
    }
    
    Linear(hidden_layer, hidden_size, num_classes, linear_weights2, linear_bias2, output);

    free(layer1_out);
    free(layer_pool1_out);
    free(layer2_out);
    free(layer_pool2_out);
    free(layer3_out);
    free(layer_pool3_out);
    free(flattened);
    free(hidden_layer);

    return output;
}


