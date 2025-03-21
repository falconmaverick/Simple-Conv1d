import numpy as np

class SimpleConv1d:
    def __init__(self, input_channels, output_channels, filter_size, padding=0, stride=1):
        """
        Problem 1 & 7: Initializes the 1D Convolution layer with multiple channels, optional padding, and variable stride.
        - input_channels: Number of input channels.
        - output_channels: Number of output channels.
        - filter_size: Size of the filter (kernel) used in the convolution.
        - padding: Number of zeros to pad on both sides of the input.
        - stride: Step size for moving the filter during convolution.
        """
        self.C_in = input_channels  # Number of input channels
        self.C_out = output_channels  # Number of output channels
        self.F = filter_size  # Filter size
        self.P = padding  # Padding size
        self.S = stride  # Stride size
        
        # Xavier initialization for weights and biases
        self.W = np.random.randn(self.C_out, self.C_in, self.F) * np.sqrt(1 / (self.C_in * self.F))
        self.b = np.random.randn(self.C_out) * np.sqrt(1 / (self.C_in * self.F))
    
    def forward(self, x):
        """
        Problem 2, 6 & 7: Forward propagation for 1D convolution with multiple channels, padding, mini-batches, and stride.
        - x: Input array with shape (batch_size, C_in, N_in).
        Returns the convolved output with shape (batch_size, C_out, N_out).
        """
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
        """
        Problem 3, 6 & 7: Backpropagation to compute gradients and update weights for mini-batches and stride.
        - delta_a: Gradient of the loss with respect to output of this layer, shape (batch_size, C_out, N_out).
        - learning_rate: Learning rate for weight updates.
        Returns the gradient with respect to the input (delta_x).
        """
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
        """
        Problem 8: Retrieve current weights and biases for integration into a neural network.
        Returns:
        - W: Weight array (C_out, C_in, F)
        - b: Bias term (C_out)
        """
        return self.W, self.b
    
    def integrate_with_mnist(self, model, train_loader, test_loader, num_epochs=10, learning_rate=0.01):
        """
        Problem 8: Train and evaluate the model using the MNIST dataset.
        - model: Neural network model with Conv1d layers replacing some fully connected layers.
        - train_loader: Training dataset loader.
        - test_loader: Test dataset loader.
        - num_epochs: Number of training epochs.
        - learning_rate: Learning rate for optimization.
        """
        import torch
        import torch.nn as nn
        import torch.optim as optim
        
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
