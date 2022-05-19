#!/usr/bin/env python
"""
A command line interface to give an example how to use the end-to-end
decision tree module.
"""

import time
import argparse
import json
from copy import deepcopy

import numpy as np
np.set_printoptions(precision=4, linewidth=100, suppress=True)

from e2edt import DecisionTree

# CNN
from utils.TinyCNN import TinyCNN
# Regularizer
from utils.SpatialRegularizer import SpatialRegularizer

# Datasets
from utils.load_LIBSVM import load_LIBSVM
from utils.ISBI_dataset import load_ISBI
from utils.fashionMNIST import load_MNIST

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
                      choices=['MNIST','FashionMNIST','protein','SensIT',
                               'connect4'],#,'ISBI'],
                      help="Known data sets you can use.")
  parser.add_argument("--leaves", default=0, type=int,
                      help="Max number of leaves for best first training.")
  parser.add_argument("--depths", default=[2], type=int, nargs='+',
                      help="Max depth of tree. If given a list of increasing "
                           "depths, the script will fit a tree for a depth, "
                           "test it, refine it, test it again and do the same "
                           "for the next max depth starting from the current "
                           "non-refined tree.")
  parser.add_argument("-r", "--reg", default=0.0, type=float,
                      help="Regularization factor, if regularizer is given.")
  parser.add_argument("-w", "--weightdecay", default=0.0, type=float,
                      help="Optimizer weight decay (L2).")
  parser.add_argument("-s", "--steepness_inc", default=0.1, type=float,
                      help="Steepness increase per epoch.")
  #parser.add_argument("-k", "--kernel", default=9, type=int,
  #                    help="Kernel size for ISBI.")
  parser.add_argument("--epochs", default=20, type=int,
                      help="Max number of epochs per fit (i.e. each split in "
                           "greedy fitting and refinement).")
  parser.add_argument("--NN", default='none', type=str, 
                      choices=['none','TinyCNN'],
                      help="Optimization algorithm to use.")
  parser.add_argument("--batch_size", default=1000, type=int,
                      help="Batch used by all loaders. "
                           "0 means a single batch. "
                           "Small batches slow down refinement! "
                           "Should be positive (no checks).")
  parser.add_argument("--algo", default='EM', type=str, choices=['EM','alt'],
                      help="Optimization algorithm to use.")
  #parser.add_argument("-v","--visualize", action='store_true',
  #                    help="Visualize filters.")
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
  #elif args.data == 'ISBI':
  #  train_set, test_set, n_features, n_classes =\
  #    load_ISBI(kernel_size=args.kernel)
  elif args.data in ['protein', 'connect4', 'SensIT']:
    train_set, test_set, n_features, n_classes =\
      load_LIBSVM(args.data, use_validation)

  print("Data set properties: features = {}, classes = {}".\
        format(n_features, n_classes))

  # --- Model setup.
  non_linear_module = None
  if args.NN == 'TinyCNN':
    non_linear_module = TinyCNN
    if args.data != 'MNIST':
        raise ValueError("The layer dimensions are hardcoded for MNIST.")
    n_features = 50

  regularizer = None
  if args.reg > 0:
    regularizer = SpatialRegularizer(n_features, strength=args.reg, cuda=args.cuda)

  batch_size = args.batch_size
  if batch_size == 0:
    batch_size = len(train_set)
  args.batch_size = batch_size

  # steepness for CNN 25, for rest: 1
  model = DecisionTree(n_features, n_classes, initial_steepness=1.0, 
                       regularizer=regularizer,
                       non_linear_module=non_linear_module,
                       batch_size=batch_size)
  
  if args.cuda:
    model.cuda()

  #~# --- Code to start from a randomly initialized tree.
  #~model.build_balanced(args.depths[-1], 1.0)
  #~for f_loss in model.refine(train_loader, args.epochs, algo=args.algo):
  #~  print(f_loss)
  #~  print(compute_score(model, train_loader))
  #~  print(compute_score(model, test_loader))

  steepnesses = [1.0]
  greedy_scores = []
  ref_scores = {s: [] for s in steepnesses}
  # --- Iterate over different depths. Each time, first build a greedy tree up
  # to given depth, then refine model.
  for depth in args.depths:
    print("Current max tree depth = {}".format(depth))

    # --- greedy training
    model.fit_greedy(train_set, epochs=args.epochs, 
                     max_depth=depth, algo=args.algo, 
                     steepness_inc=args.steepness_inc,
                     weight_decay=args.weightdecay,
                     n_max_nodes=args.leaves)

    #~# Nasty hack to count leaves without implementing a function.
    #~count = [0]
    #~model.root.foreach(lambda n: count.__setitem__(0,count[0] + int(n._path_end)))
    #~print("leaves = {}".format(count))

    # --- Evaluate greedy model.
    model.eval()
    train_score, train_score_discrete = compute_score(model, train_set, cuda=args.cuda)
    test_score, test_score_discrete = compute_score(model, test_set, cuda=args.cuda)
    
    greedy_scores.append([depth,
                          train_score, train_score_discrete,
                          test_score, test_score_discrete])
    print("Greedy tree scores at depth = {}:\n"
          "\t Probabilistic: Train score = {:.4}, Test score = {:.4} \n"
          "\t Deterministic: Train score = {:.4}, Test score = {:.4}".\
          format(greedy_scores[-1][0], greedy_scores[-1][1],
                 greedy_scores[-1][3], greedy_scores[-1][2],
                 greedy_scores[-1][4]))

    # --- Refine current model. Not available with cuda enabled, due to
    # deepcopy.
    if args.refine:
      print("Refining")
      failed = True
      for steepness in steepnesses:
        #print("Steepness = {}".format(steepness))
        if len(args.depths) > 1:
          ref_model = deepcopy(model)
        else:
          ref_model = model

        ref_epoch = 0
        for f_loss in ref_model.refine(train_set, args.epochs, args.algo, 
                                       args.weightdecay):
          ref_epoch += 1
          if ref_epoch % 5 == 0:
            print("Epoch {}".format(ref_epoch))
          if np.isnan(f_loss):
            print("failed!")
            failed = True
            break
          ref_model.root.set_steepness(inc_value=args.steepness_inc)

        # --- Evaluate the refined model.
        ref_model.eval()
        train_score, train_score_discrete = compute_score(ref_model, train_set, cuda=args.cuda)
        test_score, test_score_discrete = compute_score(ref_model, test_set, cuda=args.cuda)
        
        ref_scores[steepness].append([depth,
                           train_score, train_score_discrete,
                           test_score, test_score_discrete])
        print("Refined tree scores at depth = {}:\n"
              "\t Probabilistic: Train score = {:.4}, Test score = {:.4} \n"
              "\t Deterministic: Train score = {:.4}, Test score = {:.4}".\
              format(ref_scores[steepness][-1][0], ref_scores[steepness][-1][1],
                     ref_scores[steepness][-1][3], ref_scores[steepness][-1][2],
                     ref_scores[steepness][-1][4]))

  # --- Print summary of scores.
  print("\n--- Summary")
  print("Scores for greedy trees at specific depth.")
  for score in greedy_scores:
    print("Depth = {}:\n"
          "\t Probabilistic: Train score = {:.4}, Test score = {:.4} \n"
          "\t Deterministic: Train score = {:.4}, Test score = {:.4}".\
          format(score[0], score[1], score[3], score[2], score[4]))
  print("\nScores for refined trees at specific depth.")
  for s, scores in ref_scores.items():
    for score in scores:
      print("Depth = {}:\n"
            "\t Probabilistic: Train score = {:.4}, Test score = {:.4} \n"
            "\t Deterministic: Train score = {:.4}, Test score = {:.4}".\
            format(score[0], score[1], score[3], score[2], score[4]))

  ## --- Visualize (not tested).
  #if args.visualize:
  #  visualize_decisions(model, PLOT_FOLDER)

  #if args.data == "ISBI":
  #  print("Visualize ISBI")
  #  visualizeSegmentation(model, test_set, 'plot/images/{}_{}_test'.format(TIMESTAMP, argstr), weighted=False)

