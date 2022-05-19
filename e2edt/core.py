"""
This file implements the core of the End-to-end deterministic decision tree
implementation. The DecisionTree class acts as the interface class of the
module. The DecisionNode class is exlusively used by the DecisionTree class.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from .datasets import MaskedDataset, IndexedDataset
from .gate import Gate

class DecisionTree(nn.Module):
  def __init__(self, n_features, n_classes, initial_steepness=1.0,
               regularizer=None, non_linear_module=None, batch_size=1000):
    super(DecisionTree, self).__init__()
    self.use_cuda = False
    self.dtype = torch.FloatTensor
    self.regularizer = regularizer
    self.initial_steepness = initial_steepness
    self.root = DecisionNode(n_features, n_classes, 
                             depth=0, initial_steepness=initial_steepness,
                             regularizer=regularizer,
                             non_linear_module=non_linear_module, batch_size=batch_size)
    self.batch_size = batch_size

  def forward(self, X, discrete=False):
    return self.root(X, discrete)

  def cuda(self):
    self.use_cuda = True
    self.dtype = torch.cuda.FloatTensor
    self.root.foreach(lambda node: node.cuda())
    super(DecisionTree, self).cuda()

  def fit_greedy(self, train_set, epochs, algo, steepness_inc, 
                 max_depth=4, n_max_nodes=0, weight_decay=0.0):
    """
    Function to invoke the greedy tree building procedure described in the
    paper.
    train_set: a pytorch dataset which can be used with the pytorch dataloader.
    epochs: number of epochs to train each decision.
    algo: choose between Expectation-Maximization (EM) or alternating approach
      (alt) described in supplementary.
    steepness_inc: increase of steepness hyperparameter per epoch.
    max_depth: limit the depth of the resulting decision tree. It can also be
      used to learn extend the depth of an existing decision tree.
    n_max_nodes: if set > 0, the tree will be built in best first manner and 
      max_depth is ignored. At the moment, this will always build a new tree
      from scratch.
    weight_decay: L2 regularization of parameters as described in the pytorch
      documentation of the Adam optimizer.
    """
    train_set = MaskedDataset(train_set)
    if n_max_nodes > 0: # max_depth is not valid here...
      # for best first
      #print("Best first training!")
      path_end_nodes = self.root.fit_greedy(train_set, epochs, algo, 
                                            steepness_inc, weight_decay)
      splitted = 0
      while len(path_end_nodes) < n_max_nodes:
        node_to_split = max(path_end_nodes, 
                            key=lambda node: node.total_info_gain)
        #print("split node with info gain {}".\
        #      format(node_to_split.total_info_gain))
        node_to_split.split(self.initial_steepness)
        splitted += 1
        node_to_split.node_id = splitted
        path_end_nodes = self.root.fit_greedy(train_set, epochs, algo, 
                                              steepness_inc, weight_decay)
        #print("gains")
        #self.root.foreach(lambda x: print(x.total_info_gain) if x._path_end else False)
    else: # depth first
      #print("Depth first training!")
      splitted = 0
      need_fit = True
      while need_fit:
        need_fit = False
        path_end_nodes = self.root.fit_greedy(train_set, epochs, algo, 
                                              steepness_inc, weight_decay)
        for node in path_end_nodes:
          if node.depth < max_depth and node.splitable:
            #print("split node at depth {}".format(node.depth))
            node.split(self.initial_steepness)
            if node.depth != max_depth-1:
              need_fit = True
          splitted += 1
          node.node_id = splitted
    #print("finished fit")

  def build_balanced(self, max_depth, initial_steepness=1.0):
    depth = 0
    nodes = [[self.root]]
    while depth < max_depth:
      nodes = [c_node.split(initial_steepness) for c_nodes in nodes
                                               for c_node in c_nodes]
      depth += 1

  def refine(self, train_set, epochs=100, algo='EM', weight_decay=0.0):
    """
    This function is responsible for the actual training of the splits. It
    is not only used for refinement, but also for greedy fitting. When doing
    a greedy fit a tree stump is refined.
    Parameters should be self-explanatory.
    """
    #print(algo)
    optimizer = None
    # --- sequential loader
    sequential_loader = torch.utils.data.DataLoader(
                          train_set,
                          batch_size=self.batch_size,
                          shuffle=False)
                          #num_workers=1)
    # create unshuffled target tensor if necessary
    target_tensor = []
    for (_, target) in sequential_loader:
      target_tensor.append(target)
    target_tensor = torch.cat(target_tensor, dim=0)
    del target, sequential_loader
    if self.use_cuda:
      target_tensor = target_tensor.cuda()

    #print("Epochs: {}".format(epochs))
    for epoch in range(1, epochs+1):
      if algo == 'alt':
        optimizer, f_loss = self.alt_refine_step(train_set, optimizer)
      else: # algo == 'EM'
        optimizer, f_loss = self.EM_refine_step(train_set, optimizer,
                                                target_tensor)
      if optimizer is None: # do not optimize decisions in 1st step
        #print("wd = {}".format(weight_decay))
        optimizer = optim.Adam(self.parameters(), lr=0.001,
                               weight_decay=weight_decay)
      yield f_loss
 
  def alt_refine_step(self, train_set, 
                      optimizer=None):
    # --- train loader
    train_loader = torch.utils.data.DataLoader(
                          train_set,
                          batch_size=self.batch_size,
                          shuffle=True)
                          #num_workers=1)

    #~if optimizer is None:
    #~  optimizer = optim.Adam(self.parameters())
    # --- optimize leaf prediction
    self.eval()
    for batch_idx, (data, target) in enumerate(train_loader):
      data = Variable(data.type(self.dtype))
      target = Variable(target.type(self.dtype).long())
      # normalize posterior: true for last batch
      normalize = (batch_idx == len(train_loader)-1)
      # accumulate posterior over batches, normalize for last batch
      self.root.alt_accumulate_posterior(data, target,
                                         path_prob=1, normalize=normalize)

    # --- optimize split parameters
    total_f_loss = 0.0
    if optimizer is not None:
      self.train()
      for batch_idx, (data, target) in enumerate(train_loader):
        data = Variable(data.type(self.dtype))
        target = Variable(target.type(self.dtype).long())
        optimizer.zero_grad()

        # gradient
        f_loss = - self.root.alt_decision_loss(data, target, 1).log().sum()\
                 / len(target)
        if self.regularizer is not None:
          f_loss += self.regularizer(self.named_parameters())
        f_loss.backward()
        optimizer.step()
        # leaf prediction
        total_f_loss += f_loss.item()

    return optimizer, total_f_loss

  def EM_E_Step(self, sequential_loader):
    # --- E-step
    self.eval()
    for batch_idx, (data, target) in enumerate(sequential_loader):
      data = Variable(data.type(self.dtype))
      target = Variable(target.type(self.dtype).long())
      # compute hidden per sample per leaves raw
      #path_probs = Variable(torch.ones(len(target)).float())
      accumulated_hidden_batch = self.root.EM_accumulate_hidden(data, target,
                                                                path_prob=1)
      # normalize hidden
      stack = (batch_idx == len(sequential_loader)-1) # true for last batch
      self.root.EM_normalize_hidden(accumulated_hidden_batch, stack)

  def EM_M_Step(self, sample_idx, data, optimizer):
    sample_idx = Variable(sample_idx.long())
    if self.use_cuda:
      sample_idx = sample_idx.cuda()
    data = Variable(data.type(self.dtype))
    optimizer.zero_grad()

    # gradient
    f_loss = - self.root.EM_decision_loss(data, sample_idx, 1).sum()\
             / len(sample_idx)
    if self.regularizer is not None:
      f_loss = f_loss + self.regularizer(self.named_parameters())
    f_loss.backward() 
    optimizer.step()
    return f_loss.item()

  def EM_refine_step(self, train_set, 
                     optimizer=None, target_tensor=None):
    # --- sequential loader
    sequential_loader = torch.utils.data.DataLoader(
                          train_set,
                          batch_size=self.batch_size,
                          shuffle=False)
                          #num_workers=1)
    # --- indexed random batch loader to assign hidden values to samples
    indexed_dataset = IndexedDataset(train_set)
    indexed_loader = torch.utils.data.DataLoader(
                          indexed_dataset,
                          batch_size=self.batch_size,
                          shuffle=True)
                          #num_workers=1)

    #print("batches = {}".format(sequential_loader))
    # --- E-step
    self.eval()
    self.EM_E_Step(sequential_loader)

    # --- M-Step
    # - decision boundaries
    total_f_loss = 0.0
    if optimizer is not None:
      self.train()
      for batch_idx, (sample_idx, data, _) in enumerate(indexed_loader):
        total_f_loss += self.EM_M_Step(sample_idx, data, optimizer)
      del data, sample_idx

    # - leaf prediction
    # Compute target_tensor if not given.
    self.eval()
    if target_tensor is None:
      if isinstance(train_set, torch.utils.data.TensorDataset):
        target_tensor = train_set.target_tensor
      else:
        target_tensor = []
        for (_, target) in sequential_loader:
          target_tensor.append(target)
        target_tensor = torch.cat(target_tensor, dim=0)

      if self.use_cuda:
        target_tensor = target_tensor.cuda()
    # compute leaf posterior...
    self.root.EM_compute_posterior(target_tensor)

    return optimizer, total_f_loss

class DecisionNode(nn.Module):
  """
  Each decision node can be either an end node (self._path_end == True)
  or a split node.
  """
  def __init__(self, n_features, n_classes, depth, 
               initial_steepness=1.0, leaf_predictions=None, regularizer=None,
               non_linear_module=None, batch_size=1000):
    super(DecisionNode, self).__init__()
    self.use_cuda = False
    self.dtype = torch.FloatTensor
    self.n_features = n_features
    self.n_classes = n_classes
    self.initial_steepness = initial_steepness
    self.non_linear_module = non_linear_module
    self.gate = None
    self.depth = depth
    self._path_end = True
    self.splitable = True
    self.left_child = None
    self.right_child = None
    self.samples_R = 0
    self.samples_L = 0
    self.total_info_gain = None
    self.hidden = []
    self.hidden_normalized = True
    self.prediction_normalized = True
    self.batch_size = batch_size
    self.categorical_probs_0 = torch.ones(n_classes).float()/n_classes
    self.categorical_probs_1 = torch.ones(n_classes).float()/n_classes
    self.regularizer = regularizer
    self.leaf_predictions = leaf_predictions if leaf_predictions is not None\
                            else torch.ones(n_classes).float()/n_classes

  def cuda(self):
    self.categorical_probs_0 = self.categorical_probs_0.cuda()
    self.categorical_probs_1 = self.categorical_probs_1.cuda()

    self.leaf_predictions = self.leaf_predictions.cuda()
    self.dtype = torch.cuda.FloatTensor
    self.use_cuda = True
    super(DecisionNode, self).cuda()

  def entropy(self, probabilities):
    entropy = probabilities.data.clone()
    entropy[entropy != 0] *= entropy[entropy != 0].log()
    return -float(entropy.sum())

  def gating(self, X):
    return self.gate(X).clamp(min=0.00001, max=0.99999)

  def get_leaf_id(self, xi): 
    if not self._path_end:
      gating = self.gating(xi)
      gating = gating.round()
      if gating.data[0][0] == 1:
        return self.right_child.get_leaf_id(xi)
      else:
        return self.left_child.get_leaf_id(xi)
    return self.leaf_id

  def forward(self, X, discrete=False): 
    if not self._path_end:
      gating = self.gating(X)
      if discrete:
        gating = gating.round()
      return gating       * self.right_child(X, discrete=discrete)\
           + (1 - gating) * self.left_child(X, discrete=discrete)
    return self.leaf_predictions

  def split(self, initial_steepness):
    if self._path_end and self.splitable:
      if self.gate is None:
        self.gate = Gate(self.n_features, self.initial_steepness,
                         self.non_linear_module)
        if self.use_cuda:
          self.gate.cuda()
      self._path_end = False
      self.left_child = DecisionNode(self.n_features, self.n_classes,
                                     self.depth+1, initial_steepness, 
                                     self.categorical_probs_0, self.regularizer,
                                     self.non_linear_module, self.batch_size)
      self.right_child = DecisionNode(self.n_features, self.n_classes,
                                      self.depth+1, initial_steepness, 
                                      self.categorical_probs_1, self.regularizer,
                                      self.non_linear_module, self.batch_size)
      if self.use_cuda:
        self.left_child.cuda()
        self.right_child.cuda()
      # n_train_samples is used for visualization
      self.left_child.n_train_samples = self.samples_L
      self.right_child.n_train_samples = self.samples_R
      return self.left_child, self.right_child
    elif self.splitable:
      raise TypeError("Trying to split an inner node")

  def set_steepness(self, new_value=None, inc_value=None, 
                    normalize_gate=False):
    if not self._path_end:
      if new_value is not None:
        self.gate.steepness = new_value
      elif inc_value is not None:
        self.gate.steepness += inc_value

      if normalize_gate:
        norm = self.gate.linear.weight.data.norm() 
        self.gate.linear.weight.data /= norm
        self.gate.linear.bias.data /= norm
      self.left_child.set_steepness(new_value, inc_value, normalize_gate)
      self.right_child.set_steepness(new_value, inc_value, normalize_gate)

  def get_leaf_list(self, leaf_list=None):
    if leaf_list is None:
      leaf_list = []
    if not self._path_end:
      self.left_child.get_leaf_list(leaf_list)
      self.right_child.get_leaf_list(leaf_list)
    else:
      self.leaf_id = len(leaf_list)
      leaf_list.append(self)

    return leaf_list

  def foreach(self, func):
    func(self)
    if not self._path_end:
      self.left_child.foreach(func)
      self.right_child.foreach(func)

  def split_mask(self, train_set):
    # single batch sequential loader
    left_mask_crop = []
    right_mask_crop = []
    sequential_loader = torch.utils.data.DataLoader(train_set,
                                                    batch_size=self.batch_size,
                                                    shuffle=False)
                                                    #num_workers=1)
    self.eval()
    for data, _ in sequential_loader:
      data = Variable(data.type(self.dtype))
      gating = self.gating(data).squeeze().round() 
      gating = gating.data.cpu()
      if gating.dim() == 0: # can't concatenate 0-dim tensor
        gating = gating.view(1)
      right_mask_crop.append(gating) 
      left_mask_crop.append(1-gating)
    del sequential_loader, data, gating
    right_mask_crop = torch.cat(right_mask_crop).byte()
    left_mask_crop = torch.cat(left_mask_crop).byte()

    if train_set.mask is not None:
      right_mask = torch.zeros(len(train_set.dataset)).byte()
      left_mask = torch.zeros(len(train_set.dataset)).byte()
      right_mask[train_set.mask] = right_mask_crop
      left_mask[train_set.mask] = left_mask_crop
    else:
      right_mask = right_mask_crop
      left_mask = left_mask_crop

    return left_mask, right_mask

  def compute_info_gain(self, data_set):
    sequential_loader = torch.utils.data.DataLoader(
                          data_set,
                          batch_size=self.batch_size,
                          shuffle=False)
                          #num_workers=1)

    # --- compute info gain and count samples
    total_info_gain = 0
    samples_R = 0
    samples_L = 0
    self.eval()
    for data, _ in sequential_loader:
      data = Variable(data.type(self.dtype))
      data = self.gating(data).round()

      samples_R += float(data.data.sum())
      samples_L += float((1 - data.data).sum())
    del data, sequential_loader
  
    total_samples = samples_R + samples_L
    self.samples_R = samples_R
    self.samples_L = samples_L
    samples_R /= total_samples
    samples_L /= total_samples
    total_info_gain = self.entropy(self.leaf_predictions)\
                      - (samples_R * self.entropy(self.categorical_probs_1)\
                         + samples_L * self.entropy(self.categorical_probs_0))

    return total_info_gain

  def check_data_purity(self, data_set):
    sequential_loader = torch.utils.data.DataLoader(
                          data_set,
                          batch_size=self.batch_size,
                          shuffle=False)
                          #num_workers=1)

    # check for purity.... target_tensor does not exists for vision dsets
    target_0 = None
    for _, target in sequential_loader:
      if target_0 is None:
        target_0 = target[0]
      if (target_0 != target).any():
        return False
    return True

  def fit_greedy(self, train_set, epochs, algo, 
                 steepness_inc=0, weight_decay=0.0):
    # --- Create new mask for dataset according to deterministic split.
    if not self._path_end:
      left_mask, right_mask = self.split_mask(train_set)

      if left_mask.sum() == 0 or right_mask.sum() == 0:
        print("Warning all labels on one side - not splitting this node.")
        print("left_mask.shape = {}".format(left_mask.shape))
        print("right_mask.shape = {}".format(right_mask.shape))
        self.splitable = False
        self._path_end = True
        return [self]

      # Recursive fitting of nodes. Not efficient, but nice to debug.
      return [*self.left_child.fit_greedy(
                  MaskedDataset(train_set.dataset, mask=left_mask),
                  epochs, algo, steepness_inc),
              *self.right_child.fit_greedy(
                  MaskedDataset(train_set.dataset, mask=right_mask),
                  epochs, algo, steepness_inc)]

    # --- Convert current leaf node into split node and train.
    if self.total_info_gain is None:
      # n_train_samples is used for visualization
      self.n_train_samples = len(train_set)
      
      # check dataset for purity and length
      if self.check_data_purity(train_set): # pure or empty train set
        self.total_info_gain = 0
        self.splitable = False
        self._path_end = True
        return [self]

      # --- Fitting tree stump, try at most 5 times. Usally 1 attempt is fine.
      for attempt in range(5):
        stump = DecisionTree(self.n_features, self.n_classes, 
                             self.initial_steepness, self.regularizer,
                             self.non_linear_module)
        if self.use_cuda:
          stump.cuda()
        stump.build_balanced(max_depth=1)
        for f_loss in stump.refine(train_set, epochs, algo, weight_decay):
          if steepness_inc > 0:
            stump.root.set_steepness(inc_value=steepness_inc)

        left_mask, right_mask = stump.root.split_mask(train_set)
        if left_mask.sum() > 0 and right_mask.sum() > 0:
          break # successful fit!
      else: # No successful fit.
        self.total_info_gain = 0
        self.splitable = False
        self._path_end = True
        return [self]

      # extract gate and probabilities of root node
      self.gate = stump.root.gate
      self.categorical_probs_1 = stump.root.right_child.leaf_predictions
      self.categorical_probs_0 = stump.root.left_child.leaf_predictions
      del stump

      # compute info gain
      self.total_info_gain = self.compute_info_gain(train_set)

    return [self]

  def alt_decision_loss(self, X, Y, path_prob):
    if self._path_end:
      if not self.prediction_normalized:
        raise ValueError("Predictions are not normalized!")

      #self.leaf_predictions = self.leaf_predictions.data.type(self.dtype)
      return self.leaf_predictions[Y] * path_prob
    else:
      gating = self.gating(X).squeeze()
      path_prob_right = path_prob * gating
      path_prob_left = path_prob * (1.0 - gating)
      return self.left_child.alt_decision_loss(X, Y, path_prob_left)\
             + self.right_child.alt_decision_loss(X, Y, path_prob_right)

  def alt_accumulate_posterior(self, X, Y, path_prob, normalize=False):
    if self.prediction_normalized: # accumulating new hidden vals
      self.leaf_predictions = torch.zeros(self.n_classes, requires_grad=False).type(self.dtype)
      self.prediction_normalized = False
    if self._path_end:
      leaf_samples = (path_prob == 1)
      for k in range(self.n_classes):
        self.leaf_predictions.data[k] += (leaf_samples * (Y == k)).data.float().sum()

      if normalize:
        sum_ = self.leaf_predictions.sum()
        if sum_.item() != 0:
          self.leaf_predictions = self.leaf_predictions / sum_
        # repackage... maybe not necessary but let's do it anyway
        self.leaf_predictions = torch.tensor(self.leaf_predictions.data.type(self.dtype),
                                             requires_grad=False)
        self.prediction_normalized = True
    else:
      gating = self.gating(X).squeeze().round()
      path_prob_right = path_prob * gating
      path_prob_left = path_prob * (1.0 - gating)
      self.left_child.alt_accumulate_posterior(X, Y, path_prob_left, 
                                               normalize)
      self.right_child.alt_accumulate_posterior(X, Y, path_prob_right, 
                                                normalize)

  def EM_decision_loss(self, X, sample_idx, path_prob, debug=False):
    if self._path_end: # leaf
      if not self.hidden_normalized:
        raise ValueError("hidden values are not normalized!")

      # repackage to clear stuff from previous optimization step
      self.hidden = Variable(self.hidden.data, requires_grad=False)
      # zeros in log are not nice...
      path_prob = path_prob.clamp(min=0.00001, max=0.99999)
      return self.hidden[sample_idx] * path_prob.log()
    else: # split
      gating = self.gating(X).squeeze()
      path_prob_right = path_prob * gating
      path_prob_left = path_prob * (1.0 - gating)
      return self.left_child.EM_decision_loss(X, sample_idx, path_prob_left)\
             + self.right_child.EM_decision_loss(X, sample_idx, path_prob_right)

  def EM_compute_posterior(self, target_tensor):
    if self._path_end:
      if not self.hidden_normalized:
        raise ValueError("hidden values are not normalized!")
      for k in range(self.n_classes):
        self.leaf_predictions[k] = self.hidden[target_tensor == k].detach().sum()
      self.leaf_predictions /= self.leaf_predictions.detach().sum()
      # repackage... maybe not necessary but let's do it anyway
      #self.leaf_predictions = torch.tensor(self.leaf_predictions.data.type(self.dtype),
      #                                     requires_grad=False)
    else:
      self.left_child.EM_compute_posterior(target_tensor)
      self.right_child.EM_compute_posterior(target_tensor)

  def EM_accumulate_hidden(self, X, Y, path_prob):
    if self.hidden_normalized: # accumulating new hidden vals
      self.hidden = []
      self.hidden_normalized = False
    if self._path_end:
      hidden_batch = self.leaf_predictions[Y] * path_prob.squeeze()
      self.hidden.append(hidden_batch.data)
      return hidden_batch
    else:
      gating = self.gating(X).squeeze()
      path_prob_right = path_prob * gating
      path_prob_left = path_prob * (1.0 - gating)
      return self.left_child.EM_accumulate_hidden(X, Y, path_prob_left)\
             + self.right_child.EM_accumulate_hidden(X, Y, path_prob_right)

  def EM_normalize_hidden(self, accumulated_hidden_batch, stack):
    if self._path_end:
      self.hidden[-1] = self.hidden[-1] / accumulated_hidden_batch.data
      if stack:
        self.hidden = Variable(torch.cat(self.hidden))
        self.hidden_normalized = True
    else:
      self.left_child.EM_normalize_hidden(accumulated_hidden_batch, stack)
      self.right_child.EM_normalize_hidden(accumulated_hidden_batch, stack)
