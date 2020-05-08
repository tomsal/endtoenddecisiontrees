"""
This file implements the smooth gating/split function including the linear
combination of features. If given, the features are sent through a non linear
module first, which may also be optimized thanks to autograd..
"""
import torch.nn as nn

class Gate(nn.Module):
  def __init__(self, input_size, initial_steepness, non_linear_module=None):
    super(Gate, self).__init__()
    self.steepness = initial_steepness
    self.input_size = input_size

    # --- setup non-linear split feature module
    self.non_linear = None
    if non_linear_module is not None:
      self.non_linear = non_linear_module()
      self.input_size = self.non_linear.output_size

    # --- setup linear combination of features and sigmoid
    self.linear = nn.Linear(self.input_size, 1)
    norm = self.linear.weight.data.norm()
    self.linear.weight.data /= norm
    self.linear.bias.data /= norm
    norm = self.linear.weight.data.norm()
    self.sigmoid = nn.Sigmoid()

  def forward(self, X, debug=False):
    if self.non_linear is not None:
      X = self.non_linear(X)
    gating_logits = self.linear(X.contiguous().view(-1,self.input_size))
    gating_weight = self.sigmoid(gating_logits * self.steepness)
    return gating_weight
