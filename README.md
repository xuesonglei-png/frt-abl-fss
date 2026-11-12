# FRT-ABL-FSS

This project implements a fuzzy-rough three-way abductive learning method for handwritten arithmetic expression recognition. It includes training and evaluation code, ablation variants, experiment analysis tools, and tests.

The implementation is in [`methods/frt_abl`](methods/frt_abl). Dataset files and experiment outputs are not included. Install the dependencies in `methods/frt_abl/requirements.txt`, then run the unit tests from the repository root:

```powershell
py -m unittest discover -s methods/frt_abl/tests
```

## Datasets and splits

- **MNIST:** [Official dataset page](https://yann.lecun.org/exdb/mnist/). Google-hosted IDX downloads: [training images](https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz), [training labels](https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz), [test images](https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz), and [test labels](https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz). We deterministically reserve 600 training images per digit for validation, leaving 54,000 training images; the original 10,000 test images remain held out. Nonoverlapping image pairs form 27,000 training, 3,000 validation, and 5,000 test addition examples.
- **HWF:** [Official ABLkit documentation](https://www.lamda.nju.edu.cn/abl_test/docs/build/html/Examples/HWF.html) and [dataset download used by its loader](https://drive.google.com/u/0/uc?id=1t52OE2Wdm5GdShX1jD2Wy8phCllk0r8I&export=download). We retain the original 2,000 test expressions and deterministically reserve 1,000 validation expressions from the 10,000 training expressions. The remaining training pool is partitioned into labeled anchor, calibration, and weakly supervised roles; expressions sharing images across roles are excluded.

MNIST training pairs are also partitioned into anchor, calibration, and weakly supervised roles. Role allocation is deterministic and independent of the training seed, with image-disjoint splits. The repository also includes annotation-budget split code and HWF generalization splits that train on expression lengths 1, 3, and 5, evaluate unseen length 7, and optionally hold out operator combinations. See `prepare.py`, `budget.py`, and `prepare_generalization.py` in `methods/frt_abl`.
