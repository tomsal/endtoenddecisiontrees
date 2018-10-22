#!/usr/bin/env python

import matplotlib
#matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.transforms import Bbox
import numpy as np
import json
import argparse
import os
import torch
import re
from torch.autograd import Variable
import seaborn as sns
#sns.set_style('dark')
#sns.set_context('poster',font_scale=0.5)

sns.set_style('ticks',  {'font.family': 'sans-serif'})
sns.set_context("paper",  font_scale=5.0) #2.3)

base_dir = ''
output_dir = 'tmp/'

#matplotlib.rcParams.update({'font.size': 62})
img_size = "0.5cm"

def export_tikz(tree, folder_prefix='', folder="plot/tikz"):
  root_node = tree.root
  leaf_ids = root_node.get_leaf_list() # assign leaf_ids
  scale_factor = 10.0 / root_node.n_train_samples
  # we need number of samples to compute node thickness
  #~ max_samples = count_samples(node)

  #plt.figure(figsize=(12,12))
  # create tikz node for root node
  #~ filename = visualize_node_pdf(root_node)
  #~ tikz_string = "\\node [DN] {{\includegraphics[width={}]{{{}}}}} "\
  #~               .format(img_size, filename)

  #~~ tikz_string = "\\node [DN] {{{}}} "\
  #~~               .format(root_node.node_id)
  #tikz_string = "["

  tikz_string = "\\begin{forest}\nfor tree=TREESTYLE"
  tikz_string += export_tikz_subtree(root_node, folder, scale_factor)#, max_samples)
  tikz_string += "\n\\end{forest}"

  if len(folder_prefix) > 0:
    tikz_string = re.sub('{{{}'.format(folder_prefix), '{', tikz_string)
  #tikz_string += "\n]\n"

  #if filename != '':
  #  with open(filename, 'w') as tikz_file:
  #    tikz_file.write(tikz_string)
  return tikz_string

leaf_count = 0
decision_count = 0
def export_tikz_subtree(node, folder, scale_factor=1.0): #, node_id, max_samples):
  global leaf_count, decision_count
  os.makedirs(folder, exist_ok=True)
  if not node._path_end:
    left_str = export_tikz_subtree(node.left_child, folder, scale_factor)
    right_str = export_tikz_subtree(node.right_child, folder, scale_factor)
    filename = visualize_decision(node, folder)
    factor = max(0.1, node.n_train_samples*scale_factor)
    return "\n[{{ \includegraphics[width=1cm]{{{}}} }},edge={{ SCALEEDGE={{{:.1f}}} }} {} {}\n]".\
           format(filename, factor, left_str, right_str)
  else:
    leaf_count += 1
    filename = visualize_leaf(node, folder)
    factor = max(0.1, node.n_train_samples*scale_factor)
    return "\n\t[{{ \includegraphics[width=1cm]{{{}}} }},edge={{ SCALEEDGE={{{:.1f}}} }}]".\
           format(filename, factor)

  return output_dir+filename

def visualize_decisions(tree, folder='plot'):
  os.makedirs(folder, exist_ok=True)
  tree.root.foreach(lambda n: visualize_decision(n, folder)\
                              if not n._path_end\
                              else False)

def visualize_decision(node, folder):
  node_id = node.node_id
  filename = "decision_id{}_d{}".format(node_id, node.depth)
  weights = np.array(node.gate.linear.weight.data.numpy(), copy=True).flatten()
  bias = float(node.gate.linear.bias.data.numpy()[0])
  
  with open(os.path.join(folder, filename+'.json'), 'w') as json_file:
    json.dump({'weights':weights.tolist(), 'bias':bias}, json_file)
  #normalization = 'biasstretched'
  node_id = node.node_id
  return visualize_params(weights, bias, os.path.join(folder, filename))

def visualize_params(weights, bias, filename):
  fig = plt.figure(figsize=(5,5))
  normalization = 'zeroone'
  if normalization == 'zeroone': # scale to [0,1]
    weights -= weights.min()
    weights /= weights.max()
  elif normalization == 'biasstretched_wrong':
    weights += bias
    abs_max = max(np.abs(weights.max()), np.abs(weights.min()))
    weights /= abs_max
    weights = (weights + 1.0) / 2.0
  elif normalization == 'biasstretched':
    #weights -= weights/np.linalg.norm(weights) * bias
    abs_max = max(np.abs(weights.max()), np.abs(weights.min()))
    weights /= abs_max
    weights = (weights + 1.0) / 2.0
    
  ax = plt.Axes(fig, [0., 0., 1., 1.])
  ax.set_axis_off()
  plt.gcf().add_axes(ax)
  width = np.sqrt(weights.shape[0]).astype(int)
  ax.imshow(weights.reshape((width,width)), cmap='gray',#'RdBu',#extent=[0,12,0,12],
            interpolation='none', aspect='equal', vmin=0.0, vmax=1.0)
  fig.savefig(filename+'.pdf')
  plt.close(fig)
  
  return filename+'.pdf'

def visualize_leaf(node, folder):
  node_id = node.leaf_id
  filename = "leaf_{}_d{}".format(node_id, node.depth)

  probs = node.leaf_predictions.data.numpy()
  
  with open(os.path.join(folder, filename+'.json'), 'w') as json_file:
    json.dump({'probs':probs.tolist()}, json_file)
  return visualize_probs(probs, os.path.join(folder, filename))

def visualize_probs(probs, filename):
  fig = plt.figure(figsize=(5,5))
  #ax = plt.Axes(fig, [0., 0., 1., 1.])
  #ax.set_axis_off()
  ax = plt.gca()
  ax.set_ylim((0.0,1.0))
  #plt.gcf().add_axes(ax)
  x = np.arange(probs.shape[0], dtype=int)
  ax.bar(x, probs, width=0.8)
  #x_ticks = x.tolist()
  #x_ticks[1::2] = [None]*5
  plt.xticks(x[::2], x[::2])
  plt.yticks([0,1],[0,1])
  #ax.imshow(weights.reshape((width,width)), cmap='gray',#'RdBu',#extent=[0,12,0,12],
  #          interpolation='none', aspect='equal', vmin=0.0, vmax=1.0)
  fig.savefig(filename+'.pdf', bbox_inches='tight', pad_inches=0)
  plt.close(fig)
  return filename+'.pdf'
#~~~~   if treevis:
#~~~~     ax.set_ylim([0,5500])
#~~~~   else:
#~~~~     pass
#~~~~     #ax.set_title(f"ID {node.node_id}, Depth {node.depth}")
#~~~~     #ax.set_ylabel('sample count')
#~~~~     #ax.set_xlabel('class')
#~~~~   ax.xaxis.set_ticks(np.arange(node.class_count.shape[0]))
#~~~~ 
#~~~~ def visualizeMNIST(tree, decisions=True, leaves=True, filter_width=28):
#~~~~   decision_nodes = []
#~~~~   leaf_nodes = []
#~~~~   if decisions:
#~~~~     decision_nodes = np.array([node for node in tree.nodes_ if node.is_decision_node()])
#~~~~     depths = [node.depth for node in decision_nodes]
#~~~~     depths, decision_nodes = zip(*sorted(zip(depths,decision_nodes), key=lambda x: x[0]))
#~~~~   if leaves:
#~~~~     leaf_nodes = np.array([node for node in tree.nodes_ if not node.is_decision_node()])
#~~~~   #decision_weights = tree
#~~~~   no_decisions = len(decision_nodes)
#~~~~   no_leaves = len(leaf_nodes)
#~~~~ 
#~~~~   if decisions:
#~~~~     cols = int(np.sqrt(no_decisions)) + 1
#~~~~   else:
#~~~~     cols = int(np.sqrt(no_leaves)) + 1
#~~~~   rows = no_decisions // cols
#~~~~   if no_decisions % cols > 0:
#~~~~     rows += 1
#~~~~   leaf_offset = rows * cols
#~~~~   
#~~~~   rows += no_leaves // cols
#~~~~   if no_leaves % cols > 0:
#~~~~     rows += 1
#~~~~   col_width  = 4
#~~~~   row_height = 4
#~~~~   fig, axs = plt.subplots(rows, cols, #sharex = True, sharey = True,
#~~~~                           figsize = (cols * col_width, rows * row_height))
#~~~~ 
#~~~~   if decisions:
#~~~~     for i, node in enumerate(decision_nodes):
#~~~~       ax = axs.flatten()[i]
#~~~~ 
#~~~~       visualize_decision(node, ax, width=filter_width)
#~~~~ 
#~~~~   if leaves:
#~~~~     for i, node in enumerate(leaf_nodes):
#~~~~       ax = axs.flatten()[leaf_offset+i]
#~~~~       
#~~~~       visualize_leaf(node, ax)
#~~~~ 
#~~~~   return fig
#~~~~ 
#~~~~ #if __name__ == '__main__':
#~~~~ #  visualizeMNIST(np.random.random((63,784)))

def visualizeSegmentation(model, dataset, filename, weighted=True):
  leaf_list = model.root.get_leaf_list()
  n_leaves = len(leaf_list)
  print("number of leaves: {}".format(n_leaves))

  pixels = len(dataset)
  # color look-up hash table
  colors = np.arange(0, 255, 255/n_leaves, dtype=int)
  #colors = {k:v for k,v in zip(leaf_dict['ordered leaves'],
  #                             np.arange(0, 255, 255/n_leaves, dtype=int))}

  sequential_loader = torch.utils.data.DataLoader(
                        dataset,
                        batch_size=1,
                        shuffle=False)

  if weighted:
    n_images = len(sequential_loader.dataset.dataset.images)
  else:
    n_images = len(sequential_loader.dataset.images)
  pixels = pixels // n_images
  width = int(np.sqrt(pixels).astype(int))
  img = np.zeros(pixels)
  target_img = np.zeros(pixels)
  prediction = np.zeros(pixels)
  for idx, iter_item in enumerate(sequential_loader):
    if len(iter_item) > 2:
      _, data, target = iter_item
    else:
      data, target = iter_item
    data = Variable(data)
    image_idx = idx // pixels
    pixel_idx = idx - image_idx * pixels
    leaf_id = model.root.get_leaf_id(data)
    img[pixel_idx] = colors[leaf_id]
    target_img[pixel_idx] = target[0]
    prediction[pixel_idx] = leaf_list[leaf_id].leaf_predictions.data[1]

    if image_idx != (idx+1) // pixels:
      print("Save image {} of {}".format(image_idx, n_images))
      save_img(img, width, filename, json_export=True)
      save_img(target_img, width, filename+str(image_idx)+'_target', json_export=True)
      save_img(prediction, width, filename+str(image_idx)+'_pred', json_export=True)
      img = np.zeros(pixels)
      target_img = np.zeros(pixels)
      prediction = np.zeros(pixels)
 
def save_img(img, width, filename, json_export=False):
  img = img.reshape((width, width))
  if json_export:
    with open(filename+'.json','w') as json_file:
      json.dump({'img':img.tolist()}, json_file)

  fig = plt.figure(figsize=(12,12))
  ax = plt.Axes(fig, [0., 0., 1., 1.])
  ax.set_axis_off()
  plt.gcf().add_axes(ax)
  ax.imshow(img, cmap='gray',#extent=[0,12,0,12],
            interpolation='none', aspect='equal')#, vmin=0.0, vmax=1.0)
  fig.savefig(filename+'.pdf')
  plt.close(fig)
 
if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="Plot json decision files.")
  parser.add_argument("FILES", type=str, nargs='+',
                      help="JSON files.")
  args = parser.parse_args()

  for filename in args.FILES:
    if filename[-5:] == '.json':
      print("Processing file {}".format(filename))
      with open(filename, 'r') as json_file:
        params = json.load(json_file)
      if 'weights' in params.keys():
        visualize_params(np.array(params['weights']), 
                         params['bias'], filename[:-5])
      if 'probs' in params.keys():
        visualize_probs(np.array(params['probs']), filename[:-5])
