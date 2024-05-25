import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.optim import Adam
from tqdm import tqdm
from torch.utils.data import DataLoader

img_size = 28
batch_size = 128
epochs = 10
lr = 0.005
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def setup_torch_seed(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

class cnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.l1 = nn.Linear(7*7*32, 1024)
        self.l2 = nn.Linear(1024, 2048)
        self.l3 = nn.Linear(2048, 10)
        self.pool = nn.MaxPool2d(2,2)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size()[0], -1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x



setup_torch_seed()
preprocess = transforms.ToTensor()
train_dataset = torchvision.datasets.MNIST(root='./data', download=True, transform=preprocess, train=True)
test_dataset = torchvision.datasets.MNIST(root='./data', download=True, transform=preprocess, train=False)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

model = cnn()
model.to(device)
optimizer = Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()
criterion.to(device)

train_losses = []
test_losses = []
accuracies = []
for epoch in range(epochs):
    #train
    loss_sum = 0.0
    cnt = 0
    model.train()
    for images, labels in tqdm(train_dataloader):
        x = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        
        outputs = model(x)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        loss_sum += loss.item()
        cnt += len(images)
    
    loss_avg = loss_sum / cnt
    train_losses.append(loss_avg)
    
    #test
    loss_sum = 0.0
    cnt = 0
    correct = 0
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            x = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(x)
            loss = criterion(outputs, labels)
            
            loss_sum += loss.item()
            pred = torch.argmax(outputs, dim=1)
            correct += torch.sum(pred == labels)
            cnt += len(images)
    
    loss_avg = loss_sum / cnt
    test_losses.append(loss_avg)
    accuracy_avg = (correct / cnt).cpu().detach()
    accuracies.append(accuracy_avg)
    
    print(f'Epoch {epoch} | TrainLoss: {train_losses[epoch]} | TestLoss: {test_losses[epoch]} | Accuracy: {accuracies[epoch]}')
    
torch.save(model.to('cpu').state_dict(), 'cnn_model.pth')


plt.plot(range(0, len(train_losses), 1), train_losses, c='b', label='TrainLoss')
plt.plot(range(0, len(test_losses), 1), test_losses, c='r', label='TestLoss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()

plt.plot(range(0, len(accuracies), 1), accuracies, c='b', label='accuracy_rate')
plt.xlabel("epoch")
plt.ylabel("accuracy_rate")
plt.legend()
plt.grid()
plt.show()