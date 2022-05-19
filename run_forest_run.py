#!/usr/bin/env python
"""
A command line interface to give an example how to run the end-to-end
random forest module.
"""

import time
import argparse
import json

import numpy as np
np.set_printoptions(precision=4, linewidth=100, suppress=True)

from e2edt import forest

# CNN
from utils.TinyCNN import TinyCNN
# Regularizer
from utils.SpatialRegularizer import SpatialRegularizer

# Datasets
from utils.load_LIBSVM import load_LIBSVM
from utils.Letter_loader import load_Letter
from utils.ISBI_dataset import load_ISBI
from utils.fashionMNIST import load_MNIST
from utils.USPS_loader import load_USPS

# Visualization and evaluation
from utils.compute_score import compute_score
from utils.visualizeMNIST import export_tikz, visualize_decisions, visualizeSegmentation

if __name__ == '__main__':
  TIMESTAMP = time.strftime('%y-%m-%d_%H-%M-%S')

  # --- Parsing command line arguments.
  parser = argparse.ArgumentParser(
                      description="Train an End-to-end learned Deterministic "
                                  "Decision Tree.")
  parser.add_argument("--data", default='MNIST', type=str,
                      #choices=['MNIST','FashionMNIST','protein','SensIT',
                      #         'connect4','USPS'],#,'ISBI'],
                      help="Choose dataset.")
  parser.add_argument("--depths", default=[10], type=int, nargs='+',
                      help="Max depth of tree. If given a list of increasing "
                           "depths, the script will fit a tree for a depth, "
                           "test it, refine it, test it again and do the same "
                           "for the next max depth starting from the current "
                           "non-refined tree.")
  parser.add_argument("-w", "--weightdecay", default=0.0, type=float,
                      help="Optimizer weight decay (L2).")
  parser.add_argument("-s", "--steepness_inc", default=0.1, type=float,
                      help="Steepness increase per epoch.")
  parser.add_argument("--epochs", default=20, type=int,
                      help="Max number of epochs per fit (i.e. each split in "
                           "greedy fitting and refinement).")
  parser.add_argument("--batch_size", default=2000, type=int,
                      help="Batch used by all loaders. "
                           "0 means a single batch. "
                           "Small batches slow down refinement! "
                           "Should be positive (no checks).")
  parser.add_argument("--algo", default='EM', type=str, choices=['EM','alt'],
                      help="Optimization algorithm to use.")
  parser.add_argument("--testing", action='store_true',
                      help="Use test set for testing instead of validation set.")
  parser.add_argument("--refine", action='store_true',
                      help="Run refinement at each depth.")
  parser.add_argument("--cuda", action='store_true',
                      help="Use cuda for computations.")
  args = parser.parse_args()

  print("Running with following command line arguments: {}".\
        format(args))

  # --- Set up plot folder.
  argstr = [k[0]+str(v) for k,v in vars(args).items()]
  argstr = ''.join(argstr)
  PLOT_FOLDER = 'plot/{}_{}'.format(argstr,TIMESTAMP)

  use_validation = False if args.testing else True
  # --- Load data.
  if args.data == 'MNIST':
    train_set, test_set, n_features, n_classes =\
      load_MNIST(use_validation)
  elif args.data == 'FashionMNIST':
    train_set, test_set, n_features, n_classes =\
      load_MNIST(use_validation, fashion=True)
  elif args.data in ['protein', 'connect4', 'SensIT']:
    train_set, test_set, n_features, n_classes =\
      load_LIBSVM(args.data, use_validation)
  elif args.data == 'USPS':
    train_set, test_set, n_features, n_classes =\
      load_USPS(use_validation)
  elif args.data == 'Letter':
    train_set, test_set, n_features, n_classes =\
      load_Letter(use_validation)
  else:
    raise ValueError("Data string not understood.")

  print("Data set properties: features = {}, classes = {}".\
        format(n_features, n_classes))

  # --- Model setup.
  batch_size = args.batch_size
  if batch_size == 0:
    batch_size = len(train_set)
  args.batch_size = batch_size

  # steepness for CNN 25, for rest: 1
  feature_selector = None
  # MNIST
  if args.data == 'MNIST':
    trees = 80
    feature_selector =\
        lambda: forest.Random2DPatchSelector(15, 28)
  elif args.data == 'USPS':
    trees = 100
    feature_selector =\
        lambda: forest.Random2DPatchSelector(10, 16)
  elif args.data == 'Letter':
    trees = 70
    feature_selector =\
        lambda: forest.Random1DFeatureSelector(8, n_features)
  elif args.data == 'protein':
    trees = 10
    feature_selector =\
        lambda: forest.Random1DFeatureSelector(175, n_features)
  model = forest.RandomForest(n_features, n_classes, feature_selector)

  print("Trees = {}".format(trees))

  if args.cuda:
    model.cuda()

  steepness = 1.0
  n_trees = [trees] # Add all trees at once according to dataset.
  # Use the following to add trees per iteration.
  #n_trees = [1, 2, 2, 5] + 18 * [5] # 1, 3, 5, 10, 15, 20, ..

  greedy_scores = []
  #ref_scores = {s: [] for s in steepnesses}
  ref_scores = {steepness: []}

  for n_tree in n_trees:
    # --- setup forest for greedy training
    model.add_trees(n_tree, args)
    print("Number of trees: {}".format(len(model.trees)))
    model.fit_greedy(train_set, args)
    print("Done fitting.")
    print("Number of trees: {}".format(len(model.trees)))

    #~# Nasty hack to count leaves without implementing a function.
    #~count = [0]
    #~model.root.foreach(lambda n: count.__setitem__(0,count[0] + int(n._path_end)))
    #~print("leaves = {}".format(count))

    # --- Evaluate greedy model.
    #model.eval()
    #train_score, train_score_discrete = compute_score(model, train_set, cuda=args.cuda)
    #test_score, test_score_discrete = compute_score(model, test_set, cuda=args.cuda)

    #greedy_scores.append([len(model.trees),
    #                      train_score, train_score_discrete,
    #                      test_score, test_score_discrete])
    #print("Greedy tree scores at depth = {}:\n"
    #      "\t Probabilistic: Train score = {:.4}, Test score = {:.4} \n"
    #      "\t Deterministic: Train score = {:.4}, Test score = {:.4}".\
    #      format(greedy_scores[-1][0], greedy_scores[-1][1],
    #             greedy_scores[-1][3], greedy_scores[-1][2],
    #             greedy_scores[-1][4]))

    # --- Refine current model.
    if args.refine:
      print("Refining")
      ref_model = model

      ref_epoch = 0
      for f_loss in ref_model.refine(train_set, 3*args.epochs, args.algo,
                                     args.weightdecay, batch_size):
        ref_epoch += 1
        if ref_epoch % 5 == 0:
          print("Epoch {}".format(ref_epoch))
        if np.isnan(f_loss):
          print("failed!")
          break
        ref_model.set_steepness(inc_value=args.steepness_inc)

      # --- Evaluate the refined model.
      ref_model.eval()
      train_score, train_score_discrete = compute_score(ref_model, train_set, cuda=args.cuda)
      test_score, test_score_discrete = compute_score(ref_model, test_set, cuda=args.cuda)

      ref_scores[steepness].append([len(ref_model.trees),
                         train_score, train_score_discrete,
                         test_score, test_score_discrete])
      print("Refined random forest scores at {} trees:\n"
            "\t Probabilistic: Train score = {:.4}, Test score = {:.4} \n"
            "\t Deterministic: Train score = {:.4}, Test score = {:.4}".\
            format(ref_scores[steepness][-1][0], ref_scores[steepness][-1][1],
                   ref_scores[steepness][-1][3], ref_scores[steepness][-1][2],
                   ref_scores[steepness][-1][4]))

      model_filename = "{}_{}.pth".format(TIMESTAMP, argstr)
      print("Saving model to {}".format(model_filename))
      model.save(model_filename)
    else:
      model.flush_trees()

  total_nodes = model.count_nodes()
  leaf_nodes = model.count_leaf_nodes()
  print("Number of trees: {}".format(len(model.trees)))
  print("Total nodes: {}, {} per tree".
        format(total_nodes, float(total_nodes)/len(model.trees)))
  print("Leaf nodes: {}, {} per tree".
        format(leaf_nodes, float(leaf_nodes)/len(model.trees)))

  # --- Print summary of scores.
  print("\n--- Summary")
  print("\nScores for refined random forest at specific number of trees.")
  for s, scores in ref_scores.items():
    for score in scores:
      print("Trees = {}:\n"
            "\t Probabilistic: Train score = {:.4}, Test score = {:.4} \n"
            "\t Deterministic: Train score = {:.4}, Test score = {:.4}".\
            format(score[0], score[1], score[3], score[2], score[4]))

