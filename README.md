# End-to-end Learning of Deterministic Decision Trees

This repository contains an implementation of the algorithms described in the
papers
"End-to-End Learning of Decision Trees and Forests"
written by Thomas Hehn, Julian Kooij and Fred Hamprecht
(https://link.springer.com/article/10.1007/s11263-019-01237-6)
as well as
"End-to-end Learning of Deterministic Decision Trees"
written by Thomas Hehn and Fred Hamprecht
(https://link.springer.com/chapter/10.1007/978-3-030-12939-2_42).


## Installation
The code was tested for python 3.6 and pytorch 1.0. It is recommended to use
conda to install the required libraries.
```
conda env create -f conda_env.yml
```


## How to use
The files `run.py` and `run_forest_run.py` are example scripts to show how to
use the e2edt package.
Run
```
python run.py --help
```
for further instructions.

Example to run the decision tree code:
```
source activate e2edt
python run.py --data MNIST --depth 4 6 --epochs 50 -s 0.01 --batch_size 1000 --algo EM --refine
```

Example to run the random forest code:
```
source activate e2edt
python run_forest_run.py --data USPS --depths 10 --epochs 15 -s 0.1 --refine
```

See [data/README.md](data/README.md) for details on the datasets.

## Intention
The code is published with the intention to support other researchers who want
to use the algorithms for their research. Efficiency and speed had
only minor priority during implementation. Instead it is intended to be
flexible and to offer possibilities for easy debugging.


## How to cite
If you use the code, please cite the corresponding IJCV paper:
```
@Article{Hehn2020,
  author={Hehn, Thomas M. and Kooij, Julian F. P. and Hamprecht, Fred A.},
  title={End-to-End Learning of Decision Trees and Forests},
  journal={International Journal of Computer Vision},
  year={2020},
  month={Apr},
  day={01},
  volume={128},
  number={4},
  pages={997-1011},
  issn={1573-1405},
  doi={10.1007/s11263-019-01237-6},
  url={https://doi.org/10.1007/s11263-019-01237-6}
}
```


## Contact
Feel free to contact the first author Thomas Hehn for questions or feedback who is now at TU Delft (http://intelligent-vehicles.org/people/).

## License

The code in this repository is published under the MIT License. See the file
`LICENSE` or https://mit-license.org/ for details.
