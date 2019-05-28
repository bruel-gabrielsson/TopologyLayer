# TopologyLayer

This repository contains a Python package that implements PyTorch-compatible persistent homology layers, as well as featurization of the output.

For an introduction to this topic, see our ArXiV paper. [TODO: link and cite]

# Get Started

## Environment Configuration (Conda)

First, create a conda environment
```bash
conda create -n toplayer python=2.7
source activate toplayer
```

Now, add dependencies
```bash
conda install numpy scipy matplotlib
conda install pytorch=1.0 torchvision -c pytorch
```

If you haven't already, clone the repository
```bash
git clone git@github.com:bruel-gabrielsson/TopologyLayer.git
```

You are now ready to compile extensions

## Compiling C++ Extensions

PyTorch tutorial [here](https://pytorch.org/tutorials/advanced/cpp_extension.html)

*Important*: in environment, it seems like using the pytorch conda channel is important
```bash
source activate toplayer
conda install pytorch=1.0 torchvision -c pytorch
```

Compilation uses python's `setuptools` module.

To complile:
```bash
source activate toplayer
python setup.py install --record files.txt
```
You should now have the module available in your environment.

To delete all installed files (from TopologyLayer home directory):
```bash
xargs rm -rf < files.txt # files.txt from setup
rm -rf build dist topologylayer.egg-info
rm files.txt
```


# High-Level Interface

For easiest use, high-level classes are provided for Pytorch compatibility.

The output of the diagram layers is not just a Pytorch tensor, but a tuple, which consists of
* A tuple (again) containing the persistence barcodes
* A flag indicating if the filtration was sub or super-levelset.

The recommended usage is to just pass the return type directly into a feature layer, which will take care of parsing.

## Barcode Return Types

The output of extensions will be a tuple of `torch.float` tensors (one tensor for each homology dimension), and a flag indicating whether computation was sub-level set persistence.

```python
dgms, issublevel = layer(x)
```

`dgms[k]` is the k-dimensional barcode, where `dgms[k][j][0]` is the birth time of bar `j` and `dgms[k][j][1]` is the death time of bar `j`.

All bars are returned (including bars of length 0).  It will be assumed that a featurization layer can choose to use or ignore these bars.

If you're unfamiliar with persistence, it is probably easiest to get started by just passing a barcode into a featurization layer.

## Persistence Layers

### LevelSetLayer

A `LevelSetLayer` takes in an image (of fixed dimension), and outputs a super-level set persistence diagram tensor.

```python
from topologylayer.nn import LevelSetLayer2D
import torch

layer = LevelSetLayer2D((3,3))
x = torch.tensor([[2, 1, 1],[1, 0.5, 1],[1, 1, 1]], dtype=torch.float)
dgm, issublevelset = layer(x.view(-1))
```
Note that we pass in `x.view(-1)` - this is currently necessary for backpropagation to work.

The above should give
`dgm[0] = tensor([[2., -inf]])` and `dgm[1] = tensor([[1.0000, 0.5000]])`
corresponding to the persistence diagrams


### RipsLayer

A `RipsLayer` takes in a point cloud (an $n\times d$ tensor), and outputs the persistence diagram of the Rips complex.

```python
from topologylayer.nn import RipsLayer

layer = RipsLayer(maxdim=1)
dgm, issublevelset = layer(x)
```

### AlphaLayer
An `AlphaLayer` takes in a point cloud (an $n\times d$ tensor), and outputs the persistence diagram of the weak Alpha complex.

```python
from topologylayer.nn import AlphaLayer

layer = AlphaLayer(maxdim=1)
dgm, issublevelset = layer(x)
```

The `AlphaLayer` is similar to the Rips layer, but potentially much faster for low-dimensions.

Note that a weak Alpha complex is not an Alpha complex.  It is better thought of as the restriction of the Rips complex to the Delaunay Triangulation of the space.

## Featurization Layers
Persistence diagrams are hard to work with directly in machine learning.  We implement some easy to work with featurizations.

### SumBarcodeLengths

A `SumBarcodeLengths` layer takes in a `dgminfo` object, and sums up the lengths of the persistence pairs, ignoring infinite bars, and handling dimension padding

```python
from topologylayer.nn import LevelSetLayer2D
from topologylayer.nn.features import SumBarcodeLengths
import torch

layer = LevelSetLayer2D((28,28), maxdim=1)
sumlayer = SumBarcodeLengths()

x = torch.rand(28,28)
dgminfo = layer(x)
dlen = sumlayer(dgminfo)
```

### TopKBarcodeLengths

A `TopKBarcodeLengths` layer takes in a `dgminfo` object, and returns the top k barcode lengths in a given homology dimension as a tensor, padding by 0 if necessary.  Parameters are `dim` and `k`


### BarcodePolyFeature

A `BarcodePolyFeature` layer takes in a `dgminfo` object, and returns a polynomial feature as in Adcock Carlsson and Carlsson.  Parameters are homology dimension `dim`, and exponents `a` and `b`


# Examples


# (Deprecated) Dionysus Driver

There are also functions that call Dionysus (https://mrzv.org/software/dionysus2/) for the persistence calculations.
There functions are superseded by the PyTorch Extensions, but may still be used.

```bash
source activate toplayer
pip install --verbose dionysus
```

The corresponding imports are
```python
from topologylayer.nn.levelset import LevelSetLayer1D
from topologylayer.nn.levelset import LevelSetLayer as LevelSetLayer2D
from topologylayer.nn.alpha import AlphaLayer
from topologylayer.nn.rips import RipsLayer
```
The return types should be the same as the extensions, but output may not be identical (zero-length bars are truncated).
