#!/bin/bash

# Define variables
pt_file="tmp_conv.pt"
txt_file="weights_tmp_conv.txt"
image_path="/home/marzoug_losilla/Traitement/new_data/0_0.bmp"

# Run Python scripts (if needed)
python3 CNN_main.py "$pt_file"
python3 aff_arci.py "$pt_file" "$txt_file"

# Transfer the text file to the remote server
scp -P 33333 "$txt_file" marzoug_losilla@chatrie.freeboxos.fr:/home/marzoug_losilla/

# Execute commands on the remote server
ssh -p 33333 marzoug_losilla@chatrie.freeboxos.fr <<EOF
gcc -Wall -o test_load weights_reader.c Layers.c Bmp2Matrix.c -lm
./test_load "$txt_file" "$image_path"
exit
EOF