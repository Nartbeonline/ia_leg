import torch as t
import PIL.Image as Image
import torchvision.transforms as T
import os
import random

class MNISTDataset(t.utils.data.Dataset):
    def __init__(self, MNIST_dir, samples_per_class=1000):
        self.MNIST_dir = MNIST_dir
        self.num_classes = 10
        self.samples_per_class = samples_per_class
        
        self.img_list = []
        self.label_list = []
        
        for i in range(self.num_classes):
            path_cur = os.path.join(self.MNIST_dir, '{}'.format(i))
            img_list_cur = os.listdir(path_cur)
            
            # Sélectionner aléatoirement 10 images si il y en a plus que 10
            if len(img_list_cur) > self.samples_per_class:
                img_list_cur = random.sample(img_list_cur, self.samples_per_class)
            
            # Ajouter le chemin complet pour chaque image
            img_list_cur = [os.path.join('{}'.format(i), file) for file in img_list_cur]
            self.img_list += img_list_cur
            
            # Créer les labels correspondants
            label_list_cur = [i] * len(img_list_cur)
            self.label_list += label_list_cur
            
    def __len__(self):
        return len(self.label_list)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.MNIST_dir, self.img_list[idx])
        
        I_PIL = Image.open(img_path)
        transforms = T.Compose([
            T.Resize((32, 32)),
            T.Grayscale(),
            T.ToTensor(),
            T.Normalize(mean=0.5,std=0.5)
        ])
        I = transforms(I_PIL)
        # print(f"Output shape: {I.shape}")
        return I, t.tensor(self.label_list[idx]), img_path