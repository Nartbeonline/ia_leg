// #include <stdio.h>
// #include <stdlib.h>
// #include <time.h>
// #include <math.h>
// #include "Layers.h"

// double* MLP(double* img_in, long in_channels, long width, long height,
//             double** linear_weights1, double *linear_bias1,
//             double** linear_weights2, double *linear_bias2,
//             double** linear_weights3, double *linear_bias3,
//             double** linear_weights4, double *linear_bias4,
//             double* gamma1, double *beta1,
//             double* gamma2, double *beta2,
//             double* gamma3, double *beta3, long num_classes)
// {
//     long hidden_size = 1024;
//     printf("\ntoto1\n");
//     double* hidden_layer1 = malloc(hidden_size * sizeof(double));
//     Linear(img_in, width*height, hidden_size, linear_weights1, linear_bias1, hidden_layer1);
//     BatchNorm2d(hidden_layer1,gamma1,beta1, 1, 1, hidden_size, hidden_layer1);
//     Relu(hidden_layer1, 1, hidden_size, 1, hidden_layer1);
//     Dropout(hidden_layer1, 1, hidden_size, 1, 0.25, hidden_layer1);
//     printf("\ntoto2\n")
//     hidden_size = 512;
//     double* hidden_layer2 = malloc(hidden_size * sizeof(double));
//     Linear(hidden_layer1, width*height, hidden_size, linear_weights1, linear_bias1, hidden_layer2);
//     BatchNorm2d(hidden_layer2,gamma2,beta2, 1, 1, hidden_size, hidden_layer2);
//     Relu(hidden_layer2, 1, hidden_size, 1, hidden_layer2);
//     Dropout(hidden_layer2, 1, hidden_size, 1, 0.25, hidden_layer2);
//     printf("\ntoto3\n")
//     hidden_size = 256;
//     double* hidden_layer3 = malloc(hidden_size * sizeof(double));
//     Linear(hidden_layer2, width*height, hidden_size, linear_weights1, linear_bias1, hidden_layer3);
//     BatchNorm2d(hidden_layer3,gamma3,beta3, 1, 1, hidden_size, hidden_layer3);
//     Relu(hidden_layer3, 1, hidden_size, 1, hidden_layer3);
//     Dropout(hidden_layer3, 1, hidden_size, 1, 0.25, hidden_layer3);

//     printf("\ntoto\n")
//     hidden_size = 256;
//     double* hidden_layer4 = malloc(hidden_size * sizeof(double));
//     Linear(hidden_layer3, width*height, hidden_size, linear_weights1, linear_bias1, hidden_layer4);


//     free(hidden_layer1);
//     free(hidden_layer2);
//     free(hidden_layer3);

//     return hidden_layer4;
// }