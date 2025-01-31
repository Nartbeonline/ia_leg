import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset
# from dataset import CustomMNISTDataset
from MNISTDataset import MNISTDataset
from MNISTDataset_copy import MNISTDataset_new
import sys

torch.random.manual_seed(0)

path_MNIST_train_old = '/home/docker/Work/Test_manuscrit'
path_MNIST_train_new = '/home/docker/Work/new_data'

training_set_old = MNISTDataset(path_MNIST_train_old)
training_set_new = MNISTDataset_new(path_MNIST_train_new)
training_set = torch.utils.data.ConcatDataset([training_set_old,training_set_new])
training_set,valid_set= torch.utils.data.random_split(training_set, [0.8, 0.2])




batch_size = 16
train_loader = torch.utils.data.DataLoader(dataset = training_set,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       num_workers=2)
images, labels, _ = next(iter(train_loader))


valid_loader = torch.utils.data.DataLoader(dataset = valid_set,
                                       batch_size=batch_size,
                                       shuffle=False,
                                       num_workers=2)

#%%
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),  # 32x32 -> 30x30
            nn.ReLU(),
            nn.MaxPool2d(2),  # 30x30 -> 15x15
            nn.Conv2d(32, 64, kernel_size=3),  # 15x15 -> 13x13
            nn.ReLU(),
            nn.MaxPool2d(2),  # 13x13 -> 6x6
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64 * 6 * 6, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class ImprovedCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(ImprovedCNN, self).__init__()

        # First conv block
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout2d(0.25)

        # Second conv block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)
        self.dropout2 = nn.Dropout2d(0.25)

        # Third conv block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2)

        # Classifier layers
        self.dropout3 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.dropout4 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        # Third conv block
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)

        # Classifier
        x = torch.flatten(x, 1)
        x = self.dropout3(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout4(x)
        x = self.fc2(x)

        return x
class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes=[2048, 1024, 512], output_size=10, dropout_rate=0.3):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        # self.bn1 = nn.BatchNorm1d(hidden_sizes[0])
        # self.dropout1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        # self.bn2 = nn.BatchNorm1d(hidden_sizes[1])
        # self.dropout2 = nn.Dropout(dropout_rate)

        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        # self.bn3 = nn.BatchNorm1d(hidden_sizes[2])
        # self.dropout3 = nn.Dropout(dropout_rate)

        self.fc4 = nn.Linear(hidden_sizes[2], output_size)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        # x = self.bn1(x)
        x = F.relu(x)
        # x = self.dropout1(x)

        x = self.fc2(x)
        # x = self.bn2(x)
        x = F.relu(x)
        # x = self.dropout2(x)

        x = self.fc3(x)
        # x = self.bn3(x)
        x = F.relu(x)
        # x = self.dropout3(x)

        x = self.fc4(x)
        return x

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = ImprovedCNN(10).to(device)
# state_dict = torch.load('model_manuscrit.pt')
# model.load_state_dict(state_dict)
# model = MLP(input_size=1024).to(device)



#HYPERPARAMETRES
lr = 1e-3 #learning rate
n_epoch = 500 #number of iterations
for param in model.parameters():
    param.requires_grad = True

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

def validation(valid_loader, model):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels,_ in valid_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return (correct, total)

num_batch = len(train_loader)

training_loss_v = []
valid_acc_v = []
epochs_without_improvement = 0
early_stopping_patience = 15 

(correct_best, total) = validation(valid_loader, model)
print('Epoch [{}/{}], Valid Acc: {} %'
      .format(0, n_epoch, 100 * correct_best / total))
valid_acc_v.append(correct_best / total)

for epoch in range(n_epoch):
    model.train()
    loss_tot = 0

    for i, (images, labels,_) in enumerate(train_loader):

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss_tot += loss.item()

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Batch Loss: {:.4f}'
                  .format(epoch + 1, n_epoch, i + 1, len(train_loader), loss.item()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    correct, total = validation(valid_loader, model)
    valid_acc = 100 * correct / total

    if correct > correct_best:
        epochs_without_improvement=0
        correct_best = correct
        torch.save(model.state_dict(), sys.argv[1])
        print('Saving model: {:.2f}% valid accuracy'.format(valid_acc))
    elif(early_stopping_patience>epochs_without_improvement):
        epochs_without_improvement+=1
    else:
        print(f'Early stopping at epoch {epoch + 1}')
        break

    print('Epoch [{}/{}], Training Loss: {:.4f}, Valid Acc: {:.2f} %'
          .format(epoch + 1, n_epoch, loss_tot / len(train_loader), valid_acc))

    training_loss_v.append(loss_tot / len(train_loader))
    valid_acc_v.append(valid_acc)







