#include<stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "Layers.h"

void Conv2d(double* img_in, long in_channels, long out_channels,
            long width, long height, 
            double**** weights, double* bias, 
            long kernel_size, long padding, long stride,
            double* img_out) {
    //printf("Conv\n");
    long padded_width = width + 2 * padding;
    long padded_height = height + 2 * padding;
    
    long out_width = ((width + 2 * padding - kernel_size) / stride) + 1;
    long out_height = ((height + 2 * padding - kernel_size) / stride) + 1;
    
    double* padded_input = (double*)calloc(in_channels * padded_width * padded_height, sizeof(double));
    
    for (long c = 0; c < in_channels; c++) {
        for (long h = 0; h < height; h++) {
            for (long w = 0; w < width; w++) {
                long padded_idx = c * padded_height * padded_width + 
                                (h + padding) * padded_width + (w + padding);
                long input_idx = c * height * width + h * width + w;
                padded_input[padded_idx] = img_in[input_idx];
            }
        }
    }
    
    #pragma omp parallel for collapse(4)
    for (long out_c = 0; out_c < out_channels; out_c++) {
        for (long oh = 0; oh < out_height; oh++) {
            for (long ow = 0; ow < out_width; ow++) {
                for (long in_c = 0; in_c < in_channels; in_c++) {
                    double sum = 0.0;
                    
                    for (long kh = 0; kh < kernel_size; kh++) {
                        for (long kw = 0; kw < kernel_size; kw++) {
                            long h_in = oh * stride + kh;
                            long w_in = ow * stride + kw;
                            
                            long in_idx = in_c * padded_height * padded_width + 
                                        h_in * padded_width + w_in;
                            
                            sum += padded_input[in_idx] * weights[out_c][in_c][kh][kw];
                        }
                    }
                    
                    long out_idx = out_c * out_height * out_width + 
                                 oh * out_width + ow;
                    
                    if (in_c == 0) {
                        img_out[out_idx] = sum + bias[out_c];
                    } else {
                        img_out[out_idx] += sum;
                    }
                }
            }
        }
    }
    
    free(padded_input);
}
void MaxPool2d(double* img_in, long in_channels, long width, long height,
                      long pool, double* img_out) {
    long pooled_width = width / pool;
    long pooled_height = height / pool;
    //printf("Max_pool\n");
    for (long in_c = 0; in_c < in_channels; in_c++) {
        for (long y = 0; y < pooled_height; y++) {
            for (long x = 0; x < pooled_width; x++) {
                double max = -INFINITY;

                for (long ky = 0; ky < pool; ky++) {
                    for (long kx = 0; kx < pool; kx++) {
                        long src_y = y * pool + ky; 
                        long src_x = x * pool + kx;

                        if (src_y < height && src_x < width) {
                            long input_idx = in_c * width * height + src_y * width + src_x;
                            if (img_in[input_idx] > max) {
                                max = img_in[input_idx];
                            }
                        }
                    }
                }
                long out_idx = in_c * pooled_width * pooled_height + y * pooled_width + x;
                img_out[out_idx] = max;
            }
        }
    }
}


void BatchNorm2d(double* img_in, double* gamma, double* beta,long in_channels, long width, long height, 
                        double* img_out)
{
    //printf("BN\n");
    double* channel_mean = malloc(in_channels * sizeof(double));
    double* channel_variance = malloc(in_channels * sizeof(double));
    
    double epsilon = 1e-5;
    
    for (long c = 0; c < in_channels; c++) {
        double sum = 0.0;
        for (long y = 0; y < height; y++) {
            for (long x = 0; x < width; x++) {
                long idx = c * width * height + y * width + x;
                sum += img_in[idx];
            }
        }
        channel_mean[c] = sum / (width * height);
    }
    
    for (long c = 0; c < in_channels; c++) {
        double variance_sum = 0.0;
        for (long y = 0; y < height; y++) {
            for (long x = 0; x < width; x++) {
                long idx = c * width * height + y * width + x;
                double diff = img_in[idx] - channel_mean[c];
                variance_sum += diff * diff;
            }
        }
        channel_variance[c] = variance_sum / (width * height);
    }
    
    for (long c = 0; c < in_channels; c++) {
        for (long y = 0; y < height; y++) {
            for (long x = 0; x < width; x++) {
                long idx = c * width * height + y * width + x;
                img_out[idx] = (img_in[idx] - channel_mean[c]) / 
                               sqrt(channel_variance[c] + epsilon) * gamma[idx] + beta[idx];
            }
        }
    }
    free(channel_mean);
    free(channel_variance);
}


void Dropout(double* img_in, long in_channels, long width, long height, long p,
                    double* img_out)
{
    srand(time(NULL));
    
    double keep_prob = 1.0 - p;
    
    for (long c = 0; c < in_channels; c++) {
        for (long y = 0; y < height; y++) {
            for (long x = 0; x < width; x++) {
                long idx = c * width * height + y * width + x;
                double r = (double)rand() / RAND_MAX;
                if (r < keep_prob) {
                    img_out[idx] = img_in[idx] / keep_prob;
                } else {
                    img_out[idx] = 0.0;
                }
            }
        }
    }
}

void Linear(double* img_in, long size, long out_channels, double** weights, double* bias
            , double* img_out)
{
    // printf("toto_Lin\n");
   long input_size = size;

    for (long out_c = 0; out_c < out_channels; out_c++) {
       img_out[out_c] =0;
    }
    for (long in_c = 0; in_c < input_size; in_c++) {
        for (long out_c = 0; out_c < out_channels; out_c++) {
            img_out[out_c] += img_in[in_c] * weights[out_c][in_c];
        }
    }
    for (long out_c = 0; out_c < out_channels; out_c++) {
       img_out[out_c] +=  bias[out_c];
    }
}

void Relu(double* img_in, long in_channels, long width, long height,
                double* img_out)
{
    //printf("RELU\n");
   for (long c = 0; c < in_channels; c++) {
       for (long y = 0; y < height; y++) {
           for (long x = 0; x < width; x++) {
               long idx = c * width * height + y * width + x;
               img_out[idx] = (img_in[idx] > 0) ? img_in[idx] : 0.0;
           }
       }
   }
}

/*
#define IMG_WIDTH 4
#define IMG_HEIGHT 4
#define IN_CHANNELS 1
#define OUT_CHANNELS 1
#define KERNEL_SIZE 3
#define STRIDE 1
#define PADDING 1
#define POOL_SIZE 2
#define DROPOUT_PROB 0.5

// Vos fonctions (Conv2d, MaxPool2d, etc.) ici...

// Fonction pour afficher une image ou un tableau
void print_image(double* img, long channels, long width, long height) {
    for (long c = 0; c < channels; c++) {
        printf("Channel %ld:\n", c);
        for (long y = 0; y < height; y++) {
            for (long x = 0; x < width; x++) {
                printf("%.2f ", img[c * width * height + y * width + x]);
            }
            printf("\n");
        }
        printf("\n");
    }
}

// Fonction pour afficher les masques 3D
void print_mask(double*** mask3d, long out_channels, long in_channels, long kernel_size) {
    for (long out_c = 0; out_c < out_channels; out_c++) {
        printf("Mask for output channel %ld:\n", out_c);
        for (long in_c = 0; in_c < in_channels; in_c++) {
            printf("  Input channel %ld:\n", in_c);
            for (long y = 0; y < kernel_size; y++) {
                for (long x = 0; x < kernel_size; x++) {
                    printf("%.2f ", mask3d[out_c][in_c][y * kernel_size + x]);
                }
                printf("\n");
            }
            printf("\n");
        }
    }
}

// Fonction principale pour les tests
int main() {
    srand(time(NULL));

    // Créer une image d'entrée aléatoire
    double img_in[IN_CHANNELS * IMG_WIDTH * IMG_HEIGHT];
    for (long i = 0; i < IN_CHANNELS * IMG_WIDTH * IMG_HEIGHT; i++) {
        img_in[i] = (double)(rand() % 10); // Valeurs entre 0 et 9
    }

    printf("Input Image:\n");
    print_image(img_in, IN_CHANNELS, IMG_WIDTH, IMG_HEIGHT);

    // Conv2D test
    double img_out_conv[OUT_CHANNELS * IMG_WIDTH * IMG_HEIGHT];
    double** mask3d[OUT_CHANNELS];
    for (long out_c = 0; out_c < OUT_CHANNELS; out_c++) {
        mask3d[out_c] = (double**)malloc(IN_CHANNELS * sizeof(double*));
        for (long in_c = 0; in_c < IN_CHANNELS; in_c++) {
            mask3d[out_c][in_c] = (double*)malloc(KERNEL_SIZE * KERNEL_SIZE * sizeof(double));
            for (long k = 0; k < KERNEL_SIZE * KERNEL_SIZE; k++) {
                mask3d[out_c][in_c][k] = (double)(rand() % 3 - 1); // Valeurs entre -1 et 1
            }
        }
    }

    printf("Masks (Kernels):\n");
    print_mask(mask3d, OUT_CHANNELS, IN_CHANNELS, KERNEL_SIZE);

    Conv2d(img_in, IN_CHANNELS, OUT_CHANNELS, IMG_WIDTH, IMG_HEIGHT,
           mask3d, KERNEL_SIZE, PADDING, STRIDE, img_out_conv);

    printf("Output after Conv2D:\n");
    print_image(img_out_conv, OUT_CHANNELS, IMG_WIDTH, IMG_HEIGHT);

    // MaxPool2D test
    double img_out_pool[OUT_CHANNELS * (IMG_WIDTH / POOL_SIZE) * (IMG_HEIGHT / POOL_SIZE)];
    MaxPool2d(img_out_conv, OUT_CHANNELS, IMG_WIDTH, IMG_HEIGHT, POOL_SIZE, img_out_pool);

    printf("Output after MaxPool2D:\n");
    print_image(img_out_pool, OUT_CHANNELS, IMG_WIDTH / POOL_SIZE, IMG_HEIGHT / POOL_SIZE);

    // BatchNorm2D test
    double img_out_bn[OUT_CHANNELS * IMG_WIDTH * IMG_HEIGHT];
    BatchNorm2d(img_out_conv, OUT_CHANNELS, IMG_WIDTH, IMG_HEIGHT, img_out_bn);

    printf("Output after BatchNorm2D:\n");
    print_image(img_out_bn, OUT_CHANNELS, IMG_WIDTH, IMG_HEIGHT);

    // Dropout test
    double img_out_dropout[OUT_CHANNELS * IMG_WIDTH * IMG_HEIGHT];
    Dropout(img_out_conv, OUT_CHANNELS, IMG_WIDTH, IMG_HEIGHT, DROPOUT_PROB, img_out_dropout);

    printf("Output after Dropout:\n");
    print_image(img_out_dropout, OUT_CHANNELS, IMG_WIDTH, IMG_HEIGHT);

    // Relu test
    double img_out_relu[OUT_CHANNELS * IMG_WIDTH * IMG_HEIGHT];
    Relu(img_out_conv, OUT_CHANNELS, IMG_WIDTH, IMG_HEIGHT, img_out_relu);

    printf("Output after ReLU:\n");
    print_image(img_out_relu, OUT_CHANNELS, IMG_WIDTH, IMG_HEIGHT);

    // Libérer la mémoire des masques 3D
    for (long out_c = 0; out_c < OUT_CHANNELS; out_c++) {
        for (long in_c = 0; in_c < IN_CHANNELS; in_c++) {
            free(mask3d[out_c][in_c]);
        }
        free(mask3d[out_c]);
    }

    return 0;
}
*/