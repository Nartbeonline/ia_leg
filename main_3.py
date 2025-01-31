
#%%
import torch
#%%
from MNISTDataset import MNISTDataset
from MNISTDataset_copy import MNISTDataset_new
#%%
import torchvision.transforms as T
import matplotlib.pyplot as plt
import torch.nn as nn
import sys

torch.random.manual_seed(0)
# path_MNIST_train = '/home/docker/Work/MNIST/Training'
path_MNIST_train_old = '/home/docker/Work/Test_manuscrit'
path_MNIST_train_new = '/home/docker/Work/new_data'
# path_MNIST_train = "C:\\Users\\maxim\\ecole\\IA_leg\\docker_start\\IA\\student\\Work\\Test_manuscrit"
# training_set = MNISTDataset(path_MNIST_train)
training_set_old = MNISTDataset(path_MNIST_train_old)
training_set_new = MNISTDataset_new(path_MNIST_train_new)
training_set = torch.utils.data.ConcatDataset([training_set_old,training_set_new])
training_set,valid_set= torch.utils.data.random_split(training_set, [0.8, 0.2])
# plt.figure(1)
# for i in range(4):
#     image, label, _ = training_set[i]
#     plt.subplot(1,4,i+1)
#     plt.imshow(T.ToPILImage()(image))
#     plt.title('True label {}'.format(label))

# plt.pause(1.)



batch_size = 16
train_loader = torch.utils.data.DataLoader(dataset = training_set,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       num_workers=2)
images, labels, _ = next(iter(train_loader))

# plt.figure(2)
# for i in range(4):
#     plt.subplot(1,4,i+1)
#     plt.imshow(T.ToPILImage()(images[i,:,:,:]))
#     plt.title('True label {}'.format(labels[i]))

# plt.pause(1.)
# path_MNIST_train = '/home/docker/Work/MNIST/training'
# path_MNIST_valid = '/home/docker/Work/MNIST/Testing'
# valid_set = MNISTDataset(path_MNIST_valid)
# #valid_set,_= torch.utils.data.random_split(valid_set, [10, len(valid_set)-10])
valid_loader = torch.utils.data.DataLoader(dataset = valid_set,
                                       batch_size=batch_size,
                                       shuffle=False,
                                       num_workers=2)



# class MLP(nn.Module):
#     def __init__(self, H, input_size):
#         super(MLP, self).__init__()

#         self.C = 10
#         self.D = input_size
#         self.H = H


#         self.fc1 = nn.Linear(self.D, self.H)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(self.H, self.C)


#     def forward(self,X):

#         X1 = self.fc1(X) #NxH
#         X2 = self.relu(X1) #NxH
#         O = self.fc2(X2) #NxC

#         return O

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes=[4096,2048,1024], output_size=10, dropout_rate=0.25):#0.25
        super(MLP, self).__init__()

        # Première couche cachée
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.bn1 = nn.BatchNorm1d(hidden_sizes[0])
        self.rl1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)

        # Deuxième couche cachée
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.bn2 = nn.BatchNorm1d(hidden_sizes[1])
        self.dropout2 = nn.Dropout(dropout_rate)

        # Deuxième couche cachée
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.bn3 = nn.BatchNorm1d(hidden_sizes[2])
        self.dropout3 = nn.Dropout(dropout_rate)

        # Couche de sortie
        self.fc4 = nn.Linear(hidden_sizes[2], output_size)

    def forward(self, x):
        # Aplatir l'image
        x = x.view(x.size(0), -1)

        # Première couche
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.nn.functional.relu(x)
        x = self.dropout1(x)

        # Deuxième couche
        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.nn.functional.relu(x)
        x = self.dropout2(x)

        # Deuxième couche
        x = self.fc3(x)
        x = self.bn3(x)
        x = torch.nn.functional.relu(x)
        x = self.dropout3(x)

        # Couche de sortie
        x = self.fc4(x)

        return x


def validation(valid_loader, model):
    # Test the model
    # In test phase, we don't need to compute gradients (for memory efficiency)
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels, _ in valid_loader:
            images_vec = images.view(-1, 32*32)
            
            outputs = model(images_vec)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            # print("images vec {} ,{},{}".format(images_vec.shape,predicted.shape,labels.shape))
            correct += (predicted == labels).sum().item()
    return (correct, total)

H = 30  
lr = 1e-3  
beta = 0.9  
n_epoch = 500  
input_size = 32 * 32  

model = MLP(input_size)

optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=beta)
criterion = nn.CrossEntropyLoss()

best_valid_acc = 0.0
epochs_without_improvement = 0
early_stopping_patience = 20 

training_loss_v = []
valid_acc_v = []

correct_best, total = validation(valid_loader, model)
print('Epoch [{}/{}], Valid Acc: {:.2f} %'
      .format(0, n_epoch, 100 * correct_best / total))
valid_acc_v.append(correct_best / total)

for epoch in range(n_epoch):
    model.train()
    loss_tot = 0

    for i, (images, labels, _) in enumerate(train_loader):
        images_vec = images.view(-1, input_size)


        outputs = model(images_vec)
        loss = criterion(outputs, labels)

        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        
        loss_tot += loss.item()

       
        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Batch Loss: {:.4f}'
                  .format(epoch + 1, n_epoch, i + 1, len(train_loader), loss.item() / len(labels)))

    
    avg_train_loss = loss_tot / len(train_loader.dataset)
    training_loss_v.append(avg_train_loss)

    
    correct, total = validation(valid_loader, model)
    valid_acc = correct / total
    valid_acc_v.append(valid_acc)

    
    print('Epoch [{}/{}], Training Loss: {:.4f}, Valid Acc: {:.2f} %'
          .format(epoch + 1, n_epoch, avg_train_loss, 100 * valid_acc))

    
    if valid_acc > best_valid_acc:
        best_valid_acc = valid_acc
        epochs_without_improvement = 0
        
        torch.save(model.state_dict(), sys.argv[1])
        print('Saving model: {:.2f}% valid accuracy'.format(100 * valid_acc))
    else:
        epochs_without_improvement += 1
        print(f'No improvement for {epochs_without_improvement} epochs')

    
    if epochs_without_improvement >= early_stopping_patience:
        print(f'Early stopping at epoch {epoch + 1}')
        break

print('Best validation accuracy: {:.2f}%'.format(100 * best_valid_acc))