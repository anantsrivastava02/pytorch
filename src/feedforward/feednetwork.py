import torch 
import torch.nn as nn 
import torchvision 
import torchvision.transforms as transforms
import matplotlib.pyplot as plt 

#device config 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_size = 784 
hidden_size = 100 
num_classes = 10
epochs = 4
batch_size = 100 
learning_rate = 0.001

import torch.utils
import torch.utils.data


train_dataset = torchvision.datasets.MNIST(root='./data', train = True , transform=transforms.ToTensor() , download= True)
test_dataset = torchvision.datasets.MNIST(root='./data', train = False , transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset= train_dataset ,batch_size = batch_size, shuffle= True )
test_loader = torch.utils.data.DataLoader(dataset= test_dataset , batch_size= batch_size, shuffle= False)

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetwork, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out
    
model = NeuralNetwork(input_size, hidden_size, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/600], Loss: {loss.item():.4f}')
    

    with torch.no_grad():
      n_correct = 0 
      n_sample = 0
      for images , labels in test_loader:
        images = images.reshape(-1 , 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)

        _, pred = torch.max(outputs , 1)
        n_sample += labels.shape[0]
        n_correct = (pred == labels).sum().item()

        acc = 100.0 *n_correct/n_sample 
        print(acc)