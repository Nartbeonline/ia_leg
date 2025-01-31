import torch
import numpy as np
import sys
# Load the state_dict
weights_path =  sys.argv[1] # Replace with your file path
state_dict = torch.load(weights_path)


# Write all the layer names and weights to a text file
with open(sys.argv[2], "w") as f:
    for name, param in state_dict.items():
        if "weight" in name or "bias" in name:
            f.write(f"{name}\n")
            
            if isinstance(param, torch.Tensor):
                np_array = param.numpy()
                nb_param = len(np_array.flatten())
                f.write(f"{nb_param:d}\n")
                # Check if the parameter is a 0D array (scalar)
                if np_array.ndim == 0:
                    f.write(f"{1}\n")
                    f.write(f"{nb_param:d}\n")
                    f.write(f"{np_array.item()}\n")  # Write scalar directly
                else:
                    if len(param.shape) >= 2:
                        f.write(f"{len(param.shape)}\n")
                        for para in range(len(param.shape)):
                            f.write(f"{param.shape[para]}\n")
                    else:
                        f.write(f"{1}\n")
                        f.write(f"{nb_param:d}\n")
                    # Flatten the array and write each value in a new line
                    for value in np_array.flatten():
                        f.write(f"{value:.8f}\n")  # Write each value with 8 decimal places
            elif isinstance(param, (list, np.ndarray)):
                # If it's a list or numpy array, save it directly
                np_array = np.array(param)
                if len(param.shape) >= 2:
                        f.write(f"{len(param.shape)}\n")
                        for para in range(len(param.shape)):
                            f.write(f"{param.shape[para]}\n")
                for value in np_array.flatten():
                    f.write(f"{value:.8f}\n")  # Write each value with 8 decimal places
            else:
                f.write(f"{1}\n")
                f.write(f"{nb_param:d}\n")
                # If it's a scalar or other data type, write it directly
                f.write(f"{param}\n")
            
            f.write("\n")  # Add a newline for readability

# Print layer names and their shapes or types to verify
for layer_name, params in state_dict.items():
    if isinstance(params, torch.Tensor):
        print(f"{layer_name}: {params.shape}")
    else:
        print(f"{layer_name}: {type(params)}")
