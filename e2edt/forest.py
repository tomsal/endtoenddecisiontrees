"""
Implementation of a RandomForest class as an ensemble of DecisionTrees. It
mainly wraps around a list of DecisionTrees to provide a simple interface.
"""
from . import DecisionTree
import torch
import torch.nn as nn
import torch.optim as optim
from .datasets import IndexedDataset

class Random1DFeatureSelector():
  def __init__(self, new_features, old_features):
    if new_features > old_features:
      raise ValueError("New features have to be less than old features.")
    self.features = torch.randperm(old_features)[:new_features]
    self.output_size = len(self.features)

  def __call__(self, x):
    return x[:,self.features]

class Random2DPatchSelector():
  # only square patches
  def __init__(self, patch_width, org_width):
    if patch_width > org_width:
      raise ValueError("New features have to be less than old features.")
    self.start = int(torch.randint(org_width-patch_width, (1,1)))
    self.end = self.start + patch_width
    self.output_size = patch_width**2

  def __call__(self, x):
    return x[:,:,self.start:self.end,self.start:self.end].contiguous()

class RandomForest(nn.Module):
  def __init__(self, n_features, n_classes, feature_selector):
    super(RandomForest, self).__init__()
    self.new_trees = []
    self.n_classes = n_classes
    self.trees = []

    self.features = n_features
    self.feature_selector = feature_selector

    self.use_cuda = False

  def build_balanced(self, depth, initial_steepness, args):
    raise ValueError("Not implemented.")
    for i in range(self.n_trees):
      print("Build balanced tree {}".format(i))
      self.trees.append(
          DecisionTree(self.features, self.n_classes,
                       initial_steepness=initial_steepness,
                       regularizer=None,
                       non_linear_module=self.feature_selector,
                       batch_size=args.batch_size))
      self.trees[i].build_balanced(depth, initial_steepness)
      if self.use_cuda:
        self.trees[i].cuda()

  def add_trees(self, n_trees, args):
    for i in range(n_trees):
      print("Add tree {}".format(i+len(self.trees)))
      self.new_trees.append(
          DecisionTree(self.features, self.n_classes,
                       initial_steepness=1.0,
                       regularizer=None,
                       non_linear_module=self.feature_selector,
                       batch_size=args.batch_size))
      if self.use_cuda:
        self.new_trees[i].cuda()

  def fit_greedy(self, train_set, args):
    if len(self.new_trees) == 0:
      print("Add trees first.")
      return

    for i,t in enumerate(self.new_trees):
        i = i + len(self.trees)
        print("Fit tree {}".format(i))
        # --- greedy training
        t.fit_greedy(train_set, epochs=args.epochs,
                     max_depth=args.depths[0], algo=args.algo,
                     steepness_inc=args.steepness_inc,
                     weight_decay=args.weightdecay)
                     #n_max_nodes=args.leaves)

  def refine(self, train_set, epochs=100, algo='EM', weight_decay=0.0,
             batch_size=1000):
    """
    Slightly optimized version of refine that does not load data separately for
    each tree.
    """
    optimizers = None
    # --- sequential loader
    sequential_loader = torch.utils.data.DataLoader(
                          train_set,
                          batch_size=batch_size,
                          shuffle=False)
                          #num_workers=1)
    # --- indexed random batch loader to assign hidden values to samples
    indexed_dataset = IndexedDataset(train_set)
    indexed_loader = torch.utils.data.DataLoader(
                          indexed_dataset,
                          batch_size=batch_size,
                          shuffle=True)
                          #num_workers=1)
    if len(self.new_trees) == 0:
      self.new_trees = self.trees

    # create unshuffled target tensor if necessary
    if isinstance(train_set, torch.utils.data.TensorDataset):
      target_tensor = train_set.tensors[1]
    else:
      target_tensor = []
      for (_, target) in sequential_loader:
        target_tensor.append(target)
      target_tensor = torch.cat(target_tensor, dim=0)
      del target
    if self.use_cuda:
      target_tensor = target_tensor.cuda()

    for epoch in range(1, epochs+1):
      total_f_loss = 0.0
      if algo == 'alt':
        raise NotImplemented
      else: # algo == 'EM'
        # --- E-step
        for t in self.new_trees:
          t.eval()
        for batch_idx, (data, target) in enumerate(sequential_loader):
          for t in self.new_trees:
            t.eval()
            # compute hidden per sample per leaves raw
            with torch.no_grad():
              accumulated_hidden_batch =\
                  t.root.EM_accumulate_hidden(data, target, path_prob=1)
              # normalize hidden
              stack = (batch_idx == len(sequential_loader)-1) # true for last batch
              t.root.EM_normalize_hidden(accumulated_hidden_batch, stack)
        for t in self.new_trees:
          t.train()

        # --- M-step
        if optimizers is not None:
          for batch_idx, (sample_idx, data, _) in enumerate(indexed_loader):
            for (t, o) in zip(self.new_trees, optimizers):
              total_f_loss += t.EM_M_Step(sample_idx, data, o)

        # compute leaf posterior...
        for t in self.new_trees:
          with torch.no_grad():
            t.root.EM_compute_posterior(target_tensor)

      if optimizers is None: # do not optimize decisions in 1st step
        optimizers = [optim.Adam(t.parameters(), lr=0.001,
                                 weight_decay=weight_decay)\
                      for t in self.new_trees]

      if epoch == epochs:
        self.trees += self.new_trees
        self.new_trees = []
      yield total_f_loss

  def refine_noopt(self, train_set, epochs=100, algo='EM', weight_decay=0.0):
    refiners = [tree.refine(train_set, epochs, algo, weight_decay)\
                for tree in self.new_trees]
    for f_losses in zip(*refiners):
      yield sum(f_losses)

  def set_steepness(self, inc_value, all_trees=True):
    if all_trees:
      trees = self.trees
    else:
      trees = self.new_trees
    for tree in trees:
      tree.root.set_steepness(inc_value=inc_value)

  def cuda(self):
    self.use_cuda = True
    super(RandomForest, self).cuda()
    for t in self.trees:
      t.cuda()

  def forward(self, x, discrete=False):
    probability = 0
    for tree in self.trees:
      probability += tree(x, discrete)

    probability /= len(self.trees)

    return probability

  def count_nodes(self):
    count = [0]
    for t in self.trees:
      t.root.foreach(lambda n: count.__setitem__(0,count[0] + 1))
    return count[0]

  def count_leaf_nodes(self):
    count = [0]
    for t in self.trees:
      t.root.foreach(lambda n: count.__setitem__(0,count[0] + int(n._path_end)))
    return count[0]

  def save(self, filename):
    dicts = []
    for t in self.trees:
      state_dict = t.state_dict()
      dicts.append(state_dict)
    torch.save(dicts, filename)
