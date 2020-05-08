import numpy as np
import torch
import torch.utils.data
import h5py
#import ipdb

DATA_DIR = "data/"
PATH = "{}/usps.h5".format(DATA_DIR)

def load_USPS(validation=True):
  with h5py.File(PATH, 'r') as hf:
    train = hf.get('train')
    X_train = train.get('data')[:]
    Y_train = train.get('target')[:]
    test = hf.get('test')
    X_test = test.get('data')[:]
    Y_test = test.get('target')[:]

  rnd_state = np.random.get_state()
  # make reproducible splits
  np.random.seed(0)
  idxs = np.random.permutation(range(len(Y_train)))
  np.random.set_state(rnd_state)

  X_train = X_train.reshape(-1, 1, 16, 16)
  X_test = X_test.reshape(-1, 1, 16, 16)

  X_train = torch.from_numpy(X_train).float()
  Y_train = torch.from_numpy(Y_train).long()
  #X_valid = torch.from_numpy(X_valid).float()
  #Y_valid = torch.from_numpy(Y_valid).long()
  X_test = torch.from_numpy(X_test).float()
  Y_test = torch.from_numpy(Y_test).long()

  if validation:
    #split_idx = int(len(Y_train.shape[0])*factor) # split testing/validation
    split_idx = 1500 # split testing/validation
    X_valid, Y_valid = X_train[idxs[:split_idx]], Y_train[idxs[:split_idx]]
    X_train, Y_train = X_train[idxs[split_idx:]], Y_train[idxs[split_idx:]]

    train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    valid_dataset = torch.utils.data.TensorDataset(X_valid, Y_valid)
    return train_dataset, valid_dataset, 256, 10
  else:
    train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    test_dataset = torch.utils.data.TensorDataset(X_test, Y_test)

    return train_dataset, test_dataset, 256, 10
