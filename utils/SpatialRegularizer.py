#!/usr/bin/python

import torch
from torch.autograd import Variable
import numpy as np
from scipy.signal import convolve2d

class SpatialRegularizer:
  def __init__(self, n_features, strength=0.001, cuda=False):
    self.strength = strength
    self.dtype = torch.FloatTensor
    if cuda:
      self.dtype = torch.cuda.FloatTensor

    self.laplacian = self._create_laplacian(width=np.sqrt(n_features).astype(int))
    self.laplacian = Variable(torch.from_numpy(self.laplacian).type(self.dtype), 
                              requires_grad=False)

  def __call__(self, named_parameters):
    regularization_loss = 0
    for name, parameter in named_parameters:
      if name[-6:] == 'weight':
        parameter = parameter.squeeze()
        regularizer = self.laplacian.matmul(parameter)
        regularizer = parameter.dot(regularizer)
        regularization_loss = regularization_loss + regularizer
    return self.strength * regularization_loss

  def _create_laplacian(self, neighborhood=4, width=28):
    size = (width,width)
    width = size[0]
    height = size[1]

    if neighborhood == 4:
      kernel = np.array([[0,1,0],[1,0,1],[0,1,0]])
    if neighborhood == 8:
      kernel = np.array([[1,1,1],[1,0,1],[1,1,1]])
    L = np.diag(convolve2d(np.ones(size), kernel, mode='same').flatten())

    for row in range(L.shape[0]):
      # 4-neighborhood
      if (row + 1) // width == row // width:
        L[row,row+1] = -1
      if (row - 1) // width == row // width:
        L[row,row-1] = -1
      if row + width < width*height:
        L[row,row+width] = -1
      if row - width >= 0:
        L[row,row-width] = -1

      # 8-neighborhood (untested)
      if neighborhood == 8:
        if (row + 1) // width == row // width:
          if row + 1 + width < width*height:
            L[row,row+width] = -1
          if row + 1 - width >= 0:
            L[row,row-width] = -1
        if (row - 1) // width == row // width:
          if row - 1 + width < width*height:
            L[row,row+width] = -1
          if row - 1 - width >= 0:
            L[row,row-width] = -1

    L = (L + L.T) / 2
    
    return L

