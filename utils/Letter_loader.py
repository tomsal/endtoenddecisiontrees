import numpy as np
import torch
import torch.utils.data

from sklearn.preprocessing import StandardScaler

#import ipdb

DATA_DIR = "data/UCI/"
PATH = "{}/letter-recognition.data".format(DATA_DIR)

def map_capital_alphabet(c):
  return ord(c) - ord('A')

def parse_data_file():
  with open(PATH, 'r') as fp:
    X = []
    Y = []
    for line in fp:
      data = line.split(",")
      X.append([float(x) for x in data[1:]])
      Y.append(map_capital_alphabet(data[0]))
 
  return np.array(X), np.array(Y)

def load_Letter(validation=True):
  X, Y = parse_data_file()
  if validation:
    X_train, Y_train = X[:13000], Y[:13000]
    X_test, Y_test = X[13000:16000], Y[13000:16000]
  else:
    X_train, Y_train = X[:16000], Y[:16000]
    X_test, Y_test = X[16000:], Y[16000:]

  # zero mean, unit var
  scaler = StandardScaler().fit(X_train)
  X_train = scaler.transform(X_train) #.astype(np.float32)
  X_test = scaler.transform(X_test)#.astype(np.int)

  X_train = torch.from_numpy(X_train).float()
  Y_train = torch.from_numpy(Y_train).long()
  train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
  X_test = torch.from_numpy(X_test).float()
  Y_test = torch.from_numpy(Y_test).long()
  test_dataset = torch.utils.data.TensorDataset(X_test, Y_test)
 
  n_features = X_train.shape[1]
  n_classes = 26
  return train_dataset, test_dataset, n_features, n_classes
