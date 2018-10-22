#!/usr/bin/env python

# standard libraries
import warnings
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_svmlight_file

DATA_DIR = "data/LIBSVM/"

def load_LIBSVM(data, validation=False):
  if data == 'connect4':
    train_set, valid_set, test_set = load_connect4(test_split=0.2, factor=0.36,
                                                   normalized=False)
  if data == 'protein':
    train_set, valid_set, test_set = load_protein(normalized=False)
  if data == 'SensIT':
    train_set, valid_set, test_set = load_SensIT(normalized=False)

  X_train, Y_train = train_set
  X_valid, Y_valid = valid_set
  X_test,  Y_test = test_set
  if not validation:
    X_train = np.concatenate([X_train,X_valid])
    Y_train = np.concatenate([Y_train,Y_valid])

  n_train_samples = int(X_train.shape[0])
  n_test_samples = int(X_test.shape[0])
  n_features = int(X_train.shape[1])
  n_classes = int(Y_train.max() + 1)

  n_test_batch_size = 1000
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

  return train_dataset, test_dataset, n_features, n_classes

def test_duplicates(X, Y):
  norms = (X**2).sum(axis=1)
  s_idxs = np.argsort(norms)
  candidate_idxs = np.where(np.diff(norms[s_idxs]) == 0)[0]
 
  for c_idx in candidate_idxs:
    if ((X[s_idxs[c_idx]] - X[s_idxs[c_idx+1]])**2).sum() == 0:
      if Y[s_idxs[c_idx]] != Y[s_idxs[c_idx+1]]:
        print("Found at {} and {}".format(s_idxs[c_idx], s_idxs[c_idx+1]))
      else:
        print("same class")

  #for idx, t in enumerate(X):
  #  if idx % 1000 == 0:
  #    print(idx)
  #  diffs = np.abs(t - X).sum(axis=1)
  #  if (diffs == 0).sum() > 2:
  #    print("Found one at {}".format(idx))

def load_SensIT(normalized = False):
  train_set, valid_set, _ = \
    loadlibsvm(DATA_DIR+"SensIT_comb_train.txt",
               factor = 0.2, labelfct = lambda x: x-1, 
               normalized = normalized)

  test_set, _, _ = \
    loadlibsvm(DATA_DIR+"SensIT_comb_test.txt",
               factor = 0.0, labelfct = lambda x: x-1, 
               normalized = normalized)

  return train_set, valid_set, test_set

def load_protein(normalized = False):
  train_set, _, _ = \
    loadlibsvm(DATA_DIR+"protein_train.txt",
               factor = 0.0, labelfct = lambda x: x, 
               normalized = normalized)
  test_set, _, _ = \
    loadlibsvm(DATA_DIR+"protein_test.txt",
               factor = 0.0, labelfct = lambda x: x, 
               normalized = normalized)

  valid_set, _, _ = \
    loadlibsvm(DATA_DIR+"protein_valid.txt",
               factor = 0.0, labelfct = lambda x: x, 
               normalized = normalized)

  return train_set, valid_set, test_set

def load_connect4(factor = 0.2, test_split = 0.0, normalized = True):
  # labels before are +1 and -1, not scaled...
  train_set, valid_set, test_set = \
    loadlibsvm(DATA_DIR+"connect-4.txt",
               factor = factor, test_split = test_split,
               labelfct = lambda x: (x+1), 
               normalized = normalized)

  return train_set, valid_set, test_set

def load_german_numer(factor = 0.2, normalized = True):
  # labels before are +1 and -1, not scaled...
  train_set, valid_set, test_set = \
    loadlibsvm(DATA_DIR+"german_numer.txt",
               factor = factor, labelfct = lambda x: (x+1) / 2, 
               normalized = normalized)

  return train_set, valid_set, test_set

def load_diabetes(factor = 0.2, normalized = False):
  # labels before are +1 and -1
  train_set, valid_set, test_set = \
    loadlibsvm(DATA_DIR+"diabetesscale.txt",
               factor = factor, labelfct = lambda x: (x+1) / 2, 
               normalized = normalized)

  return train_set, valid_set, test_set

def load_sonar(factor = 0.2, normalized = False):
  # labels before are +1 and -1
  train_set, valid_set, test_set = \
    loadlibsvm(DATA_DIR+"sonarscale.txt",
               factor = factor, labelfct = lambda x: (x+1) / 2, 
               normalized = normalized)

  return train_set, valid_set, test_set

def load_covtype(factor = 0.2, normalized = False):
  # labels are 1 and 2
  train_set,valid_set,test_set = \
    loadlibsvm(DATA_DIR+"covtypelibsvmbinaryscale.txt",
               factor = factor, labelfct = lambda x: x-1, 
               normalized = normalized)

  return train_set, valid_set, test_set

def loadlibsvm(datafile = "covtypelibsvmbinaryscale.txt",factor = 0.2,
               test_split = 0.0,
               labelfct = lambda x: x, normalized = False):
  datax, datay = load_svmlight_file(datafile)
  datax = np.array(datax.todense())
  datay = labelfct(datay)

  rnd_state = np.random.get_state()
  # make reproducible splits
  np.random.seed(0)
  idxs = np.random.permutation(range(len(datay)))
  np.random.set_state(rnd_state)
  if test_split > 0.0:
    test_split_idx = int(len(datay)*test_split) # split testing/validation
    valid_split_idx = int(len(datay)*factor) # split testing/validation
    test = (datax[idxs[:test_split_idx]],datay[idxs[:test_split_idx]])
    valid = (datax[idxs[test_split_idx:valid_split_idx]],
             datay[idxs[test_split_idx:valid_split_idx]])
    train = (datax[idxs[valid_split_idx:]],datay[idxs[valid_split_idx:]])
  else:
    split_idx = int(len(datay)*factor) # split testing/validation
    valid = (datax[idxs[:split_idx]],datay[idxs[:split_idx]])
    train = (datax[idxs[split_idx:]],datay[idxs[split_idx:]])
    test = (0, 0)

  #if normalized:
  #  train_set, valid_set, dummy_set = normalize(train, valid)

  return train, valid, test

if __name__ == "__main__":
  warnings.filterwarnings("ignore", category = DeprecationWarning)

  load_SensIT()
  # covtypelibsvmbinaryscale.txt
  # sonarscale.txt
  for fct in [load_diabetes, load_SensIT, load_sonar, load_covtype, load_german_numer]:
    print("--- Testing for {}".format(fct.__name__))
    train_set, valid_set, test_set = fct(normalized = True)

    test_set_x, test_set_y = test_set
    valid_set_x, valid_set_y = valid_set
    train_set_x, train_set_y = train_set

    print("train_shape_x = {}".format(train_set_x.shape))
    print("train_shape_y = {}".format(train_set_y.shape))
    print("train_shape_y.max() = {}".format(train_set_y.max()))
    print("train_shape_y.min() = {}".format(train_set_y.min()))
    print("Has only two classes = {}\n"\
          .format(np.all(  (train_set_y == train_set_y.max()) \
                         + (train_set_y == train_set_y.min()) )))
