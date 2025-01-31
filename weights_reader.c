#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "Layers.h"
#include "Bmp2Matrix.h"

double* MLP(double* img_in, long in_channels, long width, long height,
            double** linear_weights1, double *linear_bias1,
            double** linear_weights2, double *linear_bias2,
            double** linear_weights3, double *linear_bias3,
            double** linear_weights4, double *linear_bias4,
            double* gamma1, double *beta1,
            double* gamma2, double *beta2,
            double* gamma3, double *beta3, long num_classes)
{
    long hidden_size1 = 4096;
    double* hidden_layer1 = malloc(hidden_size1 * sizeof(double));
    Linear(img_in, width*height, hidden_size1, linear_weights1, linear_bias1, hidden_layer1);
    BatchNorm2d(hidden_layer1,gamma1,beta1, 1, 1, hidden_size1, hidden_layer1);
    Relu(hidden_layer1, 1, hidden_size1, 1, hidden_layer1);
    // Dropout(hidden_layer1, 1, hidden_size1, 1, 0.25, hidden_layer1);

    long hidden_size2 = 2048;
    double* hidden_layer2 = malloc(hidden_size2 * sizeof(double));
    Linear(hidden_layer1, hidden_size1, hidden_size2, linear_weights2, linear_bias2, hidden_layer2);
    BatchNorm2d(hidden_layer2,gamma2,beta2, 1, 1, hidden_size2, hidden_layer2);
    Relu(hidden_layer2, 1, hidden_size2, 1, hidden_layer2);
    // Dropout(hidden_layer2, 1, hidden_size2, 1, 0.25, hidden_layer2);

    long hidden_size3 = 1024;
    double* hidden_layer3 = malloc(hidden_size3 * sizeof(double));
    Linear(hidden_layer2, hidden_size2, hidden_size3, linear_weights3, linear_bias3, hidden_layer3);
    BatchNorm2d(hidden_layer3,gamma3,beta3, 1, 1, hidden_size3, hidden_layer3);
    Relu(hidden_layer3, 1, hidden_size3, 1, hidden_layer3);
    // Dropout(hidden_layer3, 1, hidden_size3, 1, 0.25, hidden_layer3);

    double* hidden_layer4 = malloc(num_classes * sizeof(double));
    Linear(hidden_layer3, hidden_size3, num_classes, linear_weights4, linear_bias4, hidden_layer4);


    free(hidden_layer1);
    free(hidden_layer2);
    free(hidden_layer3);

    return hidden_layer4;
}

double* MLP_gen(double* img_in, long in_channels, long width, long height,
            char** layer_names, long* layer_sizes, int num_layers,
            double*** linear_weights, double** linear_biases,
            double** gammas, double** betas, long num_classes)
{
    double* hidden_layer_prev = img_in;
    long size_hidden_layer_prev = width * height;
    double* hidden_layer = NULL;

    // Indices pour suivre les paramètres de fc, bn, etc.
    int fc_count = 0;
    int bn_count = 0;

    for (int i = 0; i < num_layers; i++) {
        // Allouer la mémoire pour la couche cachée actuelle
        hidden_layer = malloc(layer_sizes[i] * sizeof(double));

        // Appliquer la couche correspondante
        if (strcmp(layer_names[i], "fc") == 0) {
            // Couche fully connected
            Linear(hidden_layer_prev, size_hidden_layer_prev, layer_sizes[i],
                   linear_weights[fc_count], linear_biases[fc_count], hidden_layer);
            fc_count++;  // Incrémenter le compteur de couches fc
        } else if (strcmp(layer_names[i], "bn") == 0) {
            // Couche batch normalization
            BatchNorm2d(hidden_layer_prev, gammas[bn_count], betas[bn_count], 1, 1, size_hidden_layer_prev, hidden_layer);
            bn_count++;  // Incrémenter le compteur de couches bn

            // Appliquer ReLU après une séquence fc + bn
            Relu(hidden_layer, 1, layer_sizes[i], 1, hidden_layer);
        }

        // Mettre à jour les variables pour la prochaine itération
        if (hidden_layer_prev != img_in) {
            free(hidden_layer_prev);  // Libérer la mémoire de la couche précédente
        }
        hidden_layer_prev = hidden_layer;
        size_hidden_layer_prev = layer_sizes[i];
    }

    // Couche de sortie (fully connected)
    double* output_layer = malloc(num_classes * sizeof(double));
    Linear(hidden_layer_prev, size_hidden_layer_prev, num_classes,
           linear_weights[fc_count], linear_biases[fc_count], output_layer);

    // Libérer la mémoire de la dernière couche cachée
    free(hidden_layer_prev);

    return output_layer;
}

double* load_weights_layer(char* filename,long debut,long* end, char* name_layer,int* number_weights,int* entry_weight,int* out_weight,int* h_filter,int* w_filter){
    FILE *file = fopen(filename, "rb");
    fseek(file, debut, SEEK_SET);
    char line[256];

    // Check if the file was opened successfully.
    name_layer[0] = fgetc(file);
    name_layer[1] = fgetc(file);
    name_layer[2] = '\0';  // Null-terminate the string
    fgets(line, sizeof(line), file);

    fgets(line, sizeof(line), file);
    *number_weights= atoi(line);
    fgets(line, sizeof(line), file);
    if(atoi(line)==2){
        fgets(line, sizeof(line), file);
        *out_weight= atoi(line);
        fgets(line, sizeof(line), file);
        *entry_weight= atoi(line);
        *w_filter= 0;
        *h_filter= 0;
    }
    else if(atoi(line)==4){
        fgets(line, sizeof(line), file);
        *out_weight= atoi(line);
        fgets(line, sizeof(line), file);
        *entry_weight= atoi(line);
        fgets(line, sizeof(line), file);
        *h_filter= atoi(line);
        fgets(line, sizeof(line), file);
        *w_filter= atoi(line);
    }
    else{
        fgets(line, sizeof(line), file);
        *out_weight= atoi(line);
        *entry_weight=0;
        *w_filter= 0;
        *h_filter= 0;
    }


    double *weigths = malloc(*(number_weights)*sizeof(double));
    for (int i = 0; i < *number_weights; i++) {
        if (fgets(line, sizeof(line), file) != NULL) {
            weigths[i] = atof(line);
        }
        
    }
    *end = ftell(file);
    // fread(weights, sizeof(float), size, file);
    fclose(file);
    return weigths;
}

double** resize_1d_to_2d(double* original_array, int original_size, int rows, int cols) {
    // Vérifier que la nouvelle taille correspond à l'ancien tableau
    if (rows * cols != original_size) {
        return NULL;
    }
    
    // Allouer un nouveau tableau 2D
    double** resized_array = malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; i++) {
        resized_array[i] = malloc(cols * sizeof(double));
        
        // Copier les éléments depuis le tableau original
        for (int j = 0; j < cols; j++) {
            resized_array[i][j] = original_array[i * cols + j];
        }
    }
    
    return resized_array;
}

double**** resize_1d_to_4d(double* original_array, int original_size, int dim1, int dim2, int dim3, int dim4) {
    if (dim1 * dim2 * dim3 * dim4 != original_size) {
        return NULL;
    }

    double**** resized_array = malloc(dim1 * sizeof(double***));
    for (int i = 0; i < dim1; i++) {
        resized_array[i] = malloc(dim2 * sizeof(double**));
        for (int j = 0; j < dim2; j++) {
            resized_array[i][j] = malloc(dim3 * sizeof(double*));
            for (int k = 0; k < dim3; k++) {
                resized_array[i][j][k] = malloc(dim4 * sizeof(double));
            }
        }
    }

    int index = 0;
    for (int i = 0; i < dim1; i++) {
        for (int j = 0; j < dim2; j++) {
            for (int k = 0; k < dim3; k++) {
                for (int l = 0; l < dim4; l++) {
                    resized_array[i][j][k][l] = original_array[index++];
                }
            }
        }
    }

    return resized_array;
}

void free_4d_array(double**** array, int dim1, int dim2, int dim3) {
    for (int i = 0; i < dim1; i++) {
        for (int j = 0; j < dim2; j++) {
            for (int k = 0; k < dim3; k++) {
                free(array[i][j][k]);
            }
            free(array[i][j]);
        }
        free(array[i]);
    }
    free(array);
}

// 'bn' 1D + 1D ,'fc' 2D + 1D ,'cn'


double* process_layers(
    double* img_in,                   
    int* tab_out_weights,            
    double** weights_size,            
    char** layer_name_full,           
    int current_size,                 
    long width,                       
    long height,
    int* tab_num_weights,
    int* tab_entry_weights,
    long kernel_size,
    long padding,
    long stride,
    long pool
) {
    long in_channels = 1;
    long k = 0;
    int count_fc = 0;
    int count_bn = 0;
    int count_conv = 0;

    // Find first layer with non-zero weights
    while (tab_out_weights[k] == 0) {
        k++;
    }

    long size_hidden_layer_prev = width * height;
    long size_hidden_layer = tab_out_weights[k];
    double* hidden_layer = NULL;
    double* hidden_layer_prev = img_in;
    int flag_init = 1;
    long width_curr = width;
    long height_curr = height;
    long width_next, height_next;

    for (int i = 0; i < current_size; i++) {
        // printf("Layer: %s; Iteration: %d\n", layer_name_full[i], i);

        if (strcmp(layer_name_full[i], "fc") == 0) {
            if (count_fc == 0) {
                // Handle flattening for conv to fc transition
                if (!flag_init && strcmp(layer_name_full[i-3], "co") == 0) {
                    long flattened_size = size_hidden_layer_prev * width_curr * height_curr;
                    double* flattened = malloc(flattened_size * sizeof(double));
                    for (long j = 0; j < flattened_size; j++) {
                        flattened[j] = hidden_layer_prev[j];
                    }
                    hidden_layer_prev = flattened;
                    size_hidden_layer_prev = flattened_size;
                } else if (!flag_init) {
                    Relu(hidden_layer_prev, 1, size_hidden_layer_prev, 1, hidden_layer_prev);
                }

                size_hidden_layer = tab_out_weights[i];
                hidden_layer = malloc(size_hidden_layer * sizeof(double));
                if (!hidden_layer) {
                    fprintf(stderr, "Memory allocation error\n");
                    exit(1);
                }

                Linear(hidden_layer_prev, size_hidden_layer_prev, size_hidden_layer, 
                       resize_1d_to_2d(weights_size[i], tab_num_weights[i], tab_out_weights[i], tab_entry_weights[i]), 
                       weights_size[i + 1], hidden_layer);
                count_fc = 1;
                flag_init = 0;
                hidden_layer_prev = hidden_layer;
                size_hidden_layer_prev = size_hidden_layer;
            } else {
                count_fc = 0;
            }
        }
        
        if (strcmp(layer_name_full[i], "co") == 0) {
            if (count_conv == 0) {
                width_next = (width_curr + 2*padding - kernel_size) / stride + 1;
                height_next = (height_curr + 2*padding - kernel_size) / stride + 1;

                size_hidden_layer = tab_out_weights[i];
                hidden_layer = malloc(size_hidden_layer * width_next * height_next * sizeof(double));
                if (!hidden_layer) {
                    fprintf(stderr, "Memory allocation error\n");
                    exit(1);
                }

                Conv2d(hidden_layer_prev, in_channels, size_hidden_layer, width_curr, height_curr, 
                       resize_1d_to_4d(weights_size[i], tab_num_weights[i], tab_out_weights[i], tab_entry_weights[i], kernel_size, kernel_size), 
                       weights_size[i+1], kernel_size, padding, stride, hidden_layer);
                count_conv = 1;
                flag_init = 0;
                in_channels = size_hidden_layer;
                hidden_layer_prev = hidden_layer;
                size_hidden_layer_prev = size_hidden_layer;
                width_curr = width_next;
                height_curr = height_next;
            } else {
                count_conv = 0;
            }
        }

        if (strcmp(layer_name_full[i], "bn") == 0) {
            if (count_bn == 0) {
                BatchNorm2d(hidden_layer_prev, weights_size[i], weights_size[i + 1], 1, 1, size_hidden_layer_prev, hidden_layer_prev);
                
                if (strcmp(layer_name_full[i-2], "co") == 0) {
                    Relu(hidden_layer_prev, 1, size_hidden_layer_prev, 1, hidden_layer_prev);
                    
                    double* layer_pool_out = malloc(size_hidden_layer_prev * width_curr/pool * height_curr/pool * sizeof(double));
                    MaxPool2d(hidden_layer_prev, size_hidden_layer_prev, width_curr, height_curr, pool, layer_pool_out);
                    
                    hidden_layer_prev = layer_pool_out;
                    size_hidden_layer_prev /= pool;
                    width_curr /= pool;
                    height_curr /= pool;
                }
                count_bn++;
            } else {
                count_bn = 0;
            }
        }
    }

    return hidden_layer;
}

int main(int argc, char *argv[]) {
    // Test loading weights
    FILE *file = fopen(argv[1], "rb");
    fseek(file, 0, SEEK_END);
    long size_file= ftell(file);
    fclose(file);
    char layer_name_tmp[3] ;
    char** layer_name_full = NULL;
    int num_weights = 0;
    long deb = 0;
    long fin =0;
    double **weights_size = NULL;
    int* tab_num_weights =NULL;
    int* tab_entry_weights =NULL;
    int* tab_out_weights =NULL;
    int entry_weights;
    int out_weights;
    int h_filter;
    int w_filter;
    // "test_weights.txt"
    long current_size  = 0;
    while(size_file-(fin)>=3){
        double *weights = load_weights_layer(argv[1],deb ,&fin ,layer_name_tmp, &num_weights,&entry_weights,&out_weights,&h_filter,&w_filter);
        // printf("Pos fin du fichier %ld \n",fin);
        // printf("Pos final du fichier %ld \n",size_file);
        // Verify results
        if (weights == NULL) {
            printf("Failed to load weights\n");
            return 1;
        }

        // printf("Layer Name: %s\n", layer_name_tmp);
        // printf("Number of Weights: %d\n", num_weights);
        // printf("Entry %d , Out %d\n",entry_weights,out_weights);//inverse
        // printf("Weights:\n");
        // for (int i = 0; i < num_weights; i++) {
        //     printf("Weight %d: %0.8f\n", i, weights[i]);
        // }
        deb =fin+1;
        current_size ++;
        weights_size = realloc(weights_size,current_size*sizeof(double*));
        weights_size[current_size-1] = malloc(num_weights* sizeof(double));
        weights_size[current_size-1] = weights;
        tab_num_weights = realloc(tab_num_weights,current_size*sizeof(int));
        tab_num_weights[current_size-1]= num_weights;
        tab_entry_weights = realloc(tab_entry_weights,current_size*sizeof(int));
        tab_entry_weights[current_size-1]= entry_weights;
        tab_out_weights = realloc(tab_out_weights,current_size*sizeof(int));
        tab_out_weights[current_size-1]= out_weights;
        layer_name_full = realloc(layer_name_full,current_size*sizeof(char*));
        layer_name_full[current_size-1] = malloc(2* sizeof(char));
        
        strcpy(layer_name_full[current_size-1], layer_name_tmp);
    }
    // printf("name of the file\n%s",argv[1]);
    // printf("Total layers loaded: %ld\n", current_size);

    for(int i = 0 ; i<current_size;i++){
            // printf("%s\n",layer_name_full[i]);
        }
    

    BMP bitmap;    
    FILE* pFichier=fopen(argv[2], "rb");     //Ouverture du fichier contenant l'image

    LireBitmap(pFichier, &bitmap);
    fclose(pFichier);               //Fermeture du fichier contenant l'image

    ConvertRGB2Gray(&bitmap);
    
    double* img_in = (double*)malloc(1024*sizeof(double));
    for(int i = 0; i< 32;i++){
        for(int j =0;j<32;j++){
            img_in[i*32+j]=bitmap.mPixelsGray[i][j];
            // printf("%d ",bitmap.mPixelsGray[i][j]);
        }
    }
    // printf("\n");
    long width = bitmap.infoHeader.largeur;
    long height =bitmap.infoHeader.hauteur;
    

    double* resii = process_layers(img_in,tab_out_weights,weights_size,layer_name_full,current_size,width,height,tab_num_weights,tab_entry_weights,3,1,1,2) ;

    int pred_class =-1;
    double tmp_class = -10000000000000;
    for(int i = 0; i< 10;i++){
        printf("res %lf \n",resii[i]);
        if(resii[i]>tmp_class){
            tmp_class = resii[i];
            pred_class = i;
        }
    }
    printf("The predicted class is: %d \n",pred_class);
    // Free the allocated memory
    for (long i = 0; i < current_size; i++) {
        free(weights_size[i]);
    }
    free(weights_size);
    return 0;
}


