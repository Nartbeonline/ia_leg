Ce dépôt contient le code créer durant un projet à l'ENSEIRB-MATMECA en spécialité TSI. 

Il comprend tous les fichiers permettant de décrire une architecture IA:MLP et CNN, l'apprendre sur une base de donnée et implémenter sa phase d'inférence sur une Raspberry Pi.

Les fichier CNN_main.py et main_3.py contient respectivement la desciption et la phase d'apprentissage de l'architecture CNN et MLP.

Les fichier tmp.pt contient les poids du meilleur MLP et le fichier temp_conv.pt les poids du meilleur CNN.

Le fichier aff_arci.py permet la conversion des points .pt en un fichier text repectant un format particulier.

Le fichier weights_reader.c permet l'éxecution de la phase d'inférence d'un modèle MLP ou/et CNN (en lisant les poids associé à chaque couche et en affichant le résulats pour chaque classe et la classe prédite).

Les fichiers flow_MLP.sh et flow_CNN.sh permettent d'effectuer toutes les étapes (apprentissage + passage sur raspberry pi + étape d'inférence).

Le docker utilisé pour ce projet est le suivant: nartbeonline/ia_leg_marzoug_losilla 
