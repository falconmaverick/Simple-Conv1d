import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class SimpleConv1d:
    def __init__(self, input_channels, output_channels, filter_size, padding=0, stride=1):
        self.C_in = input_channels  # Number of input channels
        self.C_out = output_channels  # Number of output channels
        self.F = filter_size  # Filter size
        self.P = padding  # Padding size
        self.S = stride  # Stride size
        
        # Xavier initialization for weights and biases
        self.W = np.random.randn(self.C_out, self.C_in, self.F) * np.sqrt(1 / (self.C_in * self.F))
        self.b = np.random.randn(self.C_out) * np.sqrt(1 / (self.C_in * self.F))
    
    def forward(self, x):
        self.x = x  # Store input for backpropagation
        batch_size, _, N_in = x.shape
        
        # Apply padding
        x_padded = np.pad(x, ((0, 0), (0, 0), (self.P, self.P)), mode='constant', constant_values=0)
        N_in_padded = x_padded.shape[2]
        N_out = (N_in_padded - self.F) // self.S + 1  # Compute output size with stride
        
        # Perform convolution
        self.a = np.zeros((batch_size, self.C_out, N_out))
        for c_out in range(self.C_out):
            for i in range(N_out):
                start = i * self.S
                self.a[:, c_out, i] = np.sum(x_padded[:, :, start:start+self.F] * self.W[c_out], axis=(1, 2)) + self.b[c_out]
        
        return self.a
    
    def backward(self, delta_a, learning_rate=0.01):
        batch_size, _, N_out = delta_a.shape
        _, _, N_in = self.x.shape
        
        # Initialize gradients
        self.delta_W = np.zeros_like(self.W)
        self.delta_b = np.zeros_like(self.b)
        delta_x = np.zeros_like(self.x)
        
        # Compute gradients
        for c_out in range(self.C_out):
            for i in range(N_out):
                start = i * self.S
                self.delta_W[c_out] += np.sum(delta_a[:, c_out, i][:, None, None] * self.x[:, :, start:start+self.F], axis=0)
                self.delta_b[c_out] += np.sum(delta_a[:, c_out, i])
                delta_x[:, :, start:start+self.F] += delta_a[:, c_out, i][:, None, None] * self.W[c_out]
        
        # Update weights and biases using gradient descent
        self.W -= learning_rate * self.delta_W
        self.b -= learning_rate * self.delta_b
        
        return delta_x
    
    def get_parameters(self):
        return self.W, self.b
    
    def integrate_with_mnist(self, model, train_loader, test_loader, num_epochs=10, learning_rate=0.01):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for images, labels in train_loader:
                images = images.view(images.size(0), 1, -1)  # Reshape to (batch_size, channels, features)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in test_loader:
                    images = images.view(images.size(0), 1, -1)
                    outputs = model(images)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            accuracy = 100 * correct / total
            print(f"Epoch {epoch+1}, Loss: {running_loss:.4f}, Accuracy: {accuracy:.2f}%")
        
        print("Training and evaluation complete. Final Accuracy: {:.2f}%".format(accuracy))
        return accuracy

# Example usage
if __name__ == "__main__":
    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Define a simple model using nn.Conv1d
    class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=10, kernel_size=5, stride=1, padding=2)
        # The output size after the Conv1d layer is (batch_size, out_channels, (input_size + 2*padding - kernel_size)/stride + 1)
        # In this case, it's (batch_size, 10, (784 + 2*2 - 5)/1 + 1) = (batch_size, 10, 784)
        # After flattening, it becomes (batch_size, 10*784) = (batch_size, 7840)
        # So the input size of the fully connected layer should be 7840
        self.fc1 = nn.Linear(10 * 784, 10)  # Adjust input size to match flattened Conv1d output
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = x.view(x.size(0), -1) # Flatten the output for the fully connected layer
        x = self.fc1(x)
        return x
    
    model = SimpleModel()
    conv1d_layer = SimpleConv1d(input_channels=1, output_channels=10, filter_size=5, padding=2, stride=1)
    
    # Train and evaluate
    conv1d_layer.integrate_with_mnist(model, train_loader, test_loader)
