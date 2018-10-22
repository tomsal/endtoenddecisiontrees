# End-to-end Learning of Deterministic Decision Trees

This repository contains an implementation of the algorithms described in the
paper "End-to-end Learning of Deterministic Decision Trees" by Thomas Hehn and
Fred Hamprecht (https://arxiv.org/abs/1712.02743).


## Installation
The code was tested for python 3.6 and pytorch 0.4. It is recommended to use
conda to install the required libraries.
```
conda create -y -n E2EDT python=3.6
conda install -y -n E2EDT pytorch=0.4 torchvision -c pytorch
conda install -y -n E2EDT scikit-learn h5py matplotlib seaborn
```


If you wish to install pytorch including cuda support, use e.g.:
```
conda install pytorch=0.4 torchvision cuda91 -c pytorch
```
Check www.pytorch.org for details which version you need.

## How to use
The file `run.py` is an example script to show how to use the e2edt package.
Run
```
python run.py --help
```
for further instructions.


If you want to test it on `MNIST` or `FashionMNIST`, pytorch will download the
data automatically. You can download the datasets `protein`, `SensIT` and
`Connect-4` from https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/.
Extract and save them to `data/LIBSVM`.


Example:
```
source activate E2EDT
python run.py --data MNIST --depth 4 6 --epochs 50 -s 0.01 --batch_size 1000 --algo EM --refine
```


## Intention
The code is published with the intention to support other researchers who want
to use the algorithms for their research. Efficiency and speed had
only minor priority during implementation. Instead it is intended to be
flexible and to offer possibilities for easy debugging.


## How to cite
If you use the code, please cite the corresponding paper.


## Contact
Feel free to contact the first author Thomas Hehn for questions or feedback who is now at TU Delft (http://intelligent-vehicles.org/people/).
