/*******************************************************
Nom ......... : main.c
Role ........ : Programme principal executant la lecture
                d'une image bitmap
Auteur ...... : Frédéric CHATRIE
Version ..... : V1.1 du 1/2/2021
Licence ..... : /

Compilation :
make veryclean
make
Pour exécuter, tapez : ./all
********************************************************/

#include <stdio.h>
#include <stdlib.h>

#include "Bmp2Matrix.h"

int main(int argc, char* argv[]){
   BMP bitmap;
   FILE* pFichier=NULL;

   pFichier=fopen("/mnt/c/Users/maxim/ecole/IA_leg/docker_start/IA/student/Work/new_data/0/0_0.bmp", "rb");     //Ouverture du fichier contenant l'image
   if (pFichier==NULL) {
       printf("%s\n", "0_1.bmp");
       printf("Erreur dans la lecture du fichier\n");
   }
   LireBitmap(pFichier, &bitmap);
   fclose(pFichier);               //Fermeture du fichier contenant l'image

   ConvertRGB2Gray(&bitmap); //fct pas !!! 
   printf("%d\n", bitmap.mPixelsGray[0][14]);
   for(int i = 0; i< bitmap.infoHeader.hauteur;i++){
        for(int j =0;j<bitmap.infoHeader.largeur;j++){
            printf("%ld ",bitmap.mPixelsGray[i][j]);
        }
    }
   DesallouerBMP(&bitmap);

   return 0;
}
