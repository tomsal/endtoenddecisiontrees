#!/usr/bin/env python

import torch
from torch.utils.data.dataset import Dataset

# Wrapper class to return only subset of samples of original dataset
class MaskedDataset(Dataset):
  def __init__(self, dataset, mask=None):
    self.dataset = dataset
    if mask is not None:
      self.mask = mask.byte()
      if len(self.mask) != len(self.dataset):
        raise TypeError("Length of mask and dataset must match.")
      self.idxs = self.mask.nonzero().long().reshape(-1)
    else: # don't mask anything
      self.mask = torch.ones(len(self.dataset)).byte()
      self.idxs = torch.arange(0, len(self.dataset)).long()

  def __getitem__(self, idx):
    data, target = self.dataset[self.idxs[idx]]
    return data, target

  def __len__(self):
    return len(self.idxs)

# Wrapper class to return samples with sample idx
class IndexedDataset(Dataset):
  def __init__(self, dataset):
    self.dataset = dataset

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    return (idx, *self.dataset[idx])
