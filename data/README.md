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
Extract and save them to `data/LIBSVM`. For the dataset `USPS` you could
obtain the file `usps.h5` for example here:
https://www.kaggle.com/bistaumanga/usps-dataset/data


Example:
```
source activate E2EDT
python run.py --data MNIST --depth 4 6 --epochs 50 -s 0.01 --batch_size 1000 --algo EM --refine
```

