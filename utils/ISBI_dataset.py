#!/usr/bin/env python

# standard libraries
import numpy as np
import h5py as h5
import torch
from torch.utils.data.dataset import Dataset
from .visualizeMNIST import save_img

DATA_DIR = "data/ISBI/"

def load_ISBI(kernel_size=9, normalize=False):
  train_data = ISBI_dataset(image_idxs=[24,26], kernel_size=kernel_size)
  test_data = ISBI_dataset(image_idxs=[6], kernel_size=kernel_size)

  n_features = kernel_size**2 # 784
  n_classes = 2

  if normalize:
    feature_mean = train_data[0]
    # online variance from wikipedia
    n = 0
    mean = 0.0 #torch.zeros(train_data[0][0].shape)
    M2 = 0.0 #torch.zeros(train_data[0][0].shape)

    for i in range(len(train_data)):
      for x in train_data[i][0].contiguous().view(n_features):
        #x = train_data[i][0].view(n_features, 1).squeeze()[j]
        n += 1
        delta = x - mean
        mean += delta/n
        delta2 = x - mean
        M2 += delta*delta2

    var = float(M2 / (n - 1))
    std = float(np.sqrt(var))
    train_data.transform = lambda x: (x - mean) / std
    test_data.transform = lambda x: (x - mean) / std

  return train_data, test_data, n_features, n_classes

class ISBI_dataset(Dataset):
  def __init__(self, image_idxs=[0], kernel_size=9, transform=None):
    self.image_idxs = image_idxs
    self.kernel_size = int(kernel_size)
    self.transform = transform
    assert self.kernel_size % 2 == 1,\
           "Kernel size is even!"

    self.images = []
    self.labels = []
    with h5.File(DATA_DIR+"train-volume.h5",'r') as volume:
      with h5.File(DATA_DIR+"train-labels.h5",'r') as labels:
        for image_idx in self.image_idxs:
          self.images.append(torch.from_numpy(np.array(volume['data'][image_idx])))
          self.labels.append(torch.from_numpy(np.array(labels['labels'][image_idx])).long())
          self.labels[-1][self.labels[-1] == 255] = 1
        #self.image = np.array(volume['data'][image_idx])
        #self.labels = np.array(labels['labels'][image_idx])
    half_kernel = self.kernel_size // 2
    for image, image_idx in zip(self.images, self.image_idxs):
      img = image[half_kernel:-half_kernel,
                  half_kernel:-half_kernel].numpy()
      save_img(img, img.shape[0], 'plot/images/ISBI_{}.pdf'.format(image_idx))

    # assert same size
    for image in self.images:
      assert image.shape[0] == 512
      assert image.shape[1] == 512

  def __getitem__(self, idx):
    width, height = self.images[0].shape
    # remove halo
    width -= self.kernel_size - 1
    height -= self.kernel_size - 1

    image_idx = idx // (width*height)
    idx -= image_idx * (width*height)
    r = idx // height
    c = idx % height

    data_x = self.images[image_idx][r:r+self.kernel_size,c:c+self.kernel_size]
    data_y = self.labels[image_idx][r+self.kernel_size//2,c+self.kernel_size//2]
    if self.transform is not None:
      data_x = self.transform(data_x)
    return data_x, data_y

  def __len__(self):
    width, height = self.images[0].shape
    # remove halo
    width -= self.kernel_size - 1
    height -= self.kernel_size - 1
    return height * width * len(self.images)

