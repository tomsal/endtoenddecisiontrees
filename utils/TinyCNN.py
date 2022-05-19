import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyCNN(nn.Module):
  """
  A small neural network architecture to extract features from MNIST samples.
  """
  def __init__(self):
    super(TinyCNN, self).__init__()
    self.conv1 = nn.Conv2d(1, 3, kernel_size=5)
    self.conv2 = nn.Conv2d(3, 6, kernel_size=5)
    #self.conv2_drop = nn.Dropout2d()
    self.fc1 = nn.Linear(6*16, 50) # input size = 50
    self.output_size = 50
    #self.output_size = 784*3 #50

  def forward(self, X):
    #X = F.relu(self.conv1(X))
    #X = F.relu(self.conv2(X))
    X = F.relu(F.max_pool2d(self.conv1(X), 2))
    X = F.relu(F.max_pool2d(self.conv2(X), 2))
    #X = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(X)), 2))
    X = X.view(X.shape[0], -1)
    #X = X.view(-1, 784*3)
    X = F.relu(self.fc1(X))
    return X


