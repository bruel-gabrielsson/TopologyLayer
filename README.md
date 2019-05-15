# TopologyLayer

Requires Dionysus https://mrzv.org/software/dionysus2/


## Example Installation (Conda)

First, create a conda environment
```bash
conda create -n toplayer python=2.7
source activate toplayer
```

Now, add dependencies
```bash
pip install --verbose dionysus
conda install numpy scipy matplotlib
conda install pytorch=1.0 torchvision -c pytorch
```

If you haven't already, clone the repository
```bash
git clone git@github.com:bruel-gabrielsson/TopologyLayer.git
```

You should now be able to run the example file
```bash
cd <path_to_TopologyLayer>/Examples
python examples.py
```

## Compiling C++ Extensions

PyTorch tutorial [here](https://pytorch.org/tutorials/advanced/cpp_extension.html)

*Important*: in environment, it seems like using the pytorch conda channel is important
```bash
source activate toplayer
conda install pytorch=1.0 torchvision -c pytorch
```

Use python `setuptools`

To complile:
```bash
source activate toplayer
python setup.py install --record files.txt
```
You should now have the module available in your environment.

To delete (while testing):
```bash
xargs rm -rf < files.txt # files.txt from setup
rm -rf build dist topologylayer.egg-info
rm files.txt
```

## Barcode Return Types

The output of extensions will be a tuple of `torch.float` tensors (one tensor for each homology dimension), and a flag indicating whether computation was sub-level set persistence.

```python
dgms, issublevel = layer(x)
```

`dgms[k]` is the k-dimensional barcode, where `dgms[k][j][0]` is the birth time of bar `j` and `dgms[k][j][1]` is the death time of bar `j`

Note that this differs from the old method, where all diagrams had the same length (by adding padding), and were returned as a 3D tensor instead of a tuple of 2D tensors.

All bars are returned (including bars of length 0).  It will be assumed that a featurization layer can choose to use or ignore these bars.

In general, barcodes will be immediately passed to some featurization layer.




# High-Level Interface

For easiest use, high-level classes are provided for Pytorch compatibility.

The output of the diagram layers is not just a Pytorch tensor, but a `dgminfo` object, which currently consists of the diagram pytorch tensor and a boolean flag indicating the filtration direction, which is necessary for featurization

The recommended usage is to just pass `dgminfo` directly into a feature layer, which will take care of parsing, since the representation may change in the future.

### LevelSetLayer

A `LevelSetLayer` takes in an image (of fixed dimension), and outputs a super-level set persistence diagram tensor.

```python
from topologylayer.nn.levelset import LevelSetLayer
import torch

layer = LevelSetLayer((3,3))
x = torch.tensor([[2, 1, 1],[1, 0.5, 1],[1, 1, 1]], dtype=torch.float)
dgm, issublevelset = layer(x.view(-1))
```
Note that we pass in `x.view(-1)` - this is currently necessary for backpropagation to work.

The above should give
`dgm[0] = tensor([[2., -inf]])` and `dgm[1] = tensor([[1.0000, 0.5000]])`
corresponding to the persistence diagrams


### RipsLayer

A `RipsLayer` takes in a point cloud, and outputs the persistence diagram of the Rips complex.

```python
from topologylayer.nn import RipsLayer

layer = RipsLayer(maxdim=1, rmax=np.inf, verbose=True)
dgm, issublevelset = layer(x)
```

### AlphaLayer
An `AlphaLayer` takes in a point cloud, and outputs the persistence diagram of the Alpha complex.

```python
from topologylayer.nn import AlphaLayer

layer = AlphaLayer(maxdim=1, verbose=True)
dgm, issublevelset = layer(x)
```

The `AlphaLayer` is similar to the Rips layer, but potentially much faster for low-dimensions.
We use the convention that edges in the alpha complex have filtration value equal to the distance between the two points they connect.  This differs from the typical definition, where filtration values are halved.  We also use a flag complex on the 1-skeleton, which differs from the typical definition.

Notes
* 0-dimensional homology should agree with `RipsLayer`
* 1-dimensional homology is 2*Cech homology
* 2+ dimensional homology - no exact relation with Rips/Cech

### SumBarcodeLengths

A `SumBarcodeLengths` layer takes in a `dgminfo` object, and sums up the lengths of the persistence pairs, ignoring infinite bars, and handling dimension padding

```python
from topologylayer.nn.levelset import LevelSetLayer
from topologylayer.nn.features import SumBarcodeLengths
import torch

layer = LevelSetLayer((28,28), maxdim=1)
sumlayer = SumBarcodeLengths()

x = torch.rand(28,28)
dgminfo = layer(x)
dlen = sumlayer(dgminfo)
```

### TopKBarcodeLengths

A `TopKBarcodeLengths` layer takes in a `dgminfo` object, and returns the top k barcode lengths in a given homology dimension as a tensor, padding by 0 if necessary.  Parameters are `dim` and `k`


### BarcodePolyFeature

A `BarcodePolyFeature` layer takes in a `dgminfo` object, and returns a polynomial feature as in Adcock Carlsson and Carlsson.  Parameters are homology dimension `dim`, and exponents `a` and `b`


# Lower-Level Information

## Diagram Basics

The output of a `TopologyLayer` is a `pytorch.tensor` of size `d x ndp x 2` where `d` is the maximum homology dimension, `ndp` is the maximum number of diagram points in any homology dimension, and the remaining index is for birth and death times.  If a homology dimension has fewer than `ndp` pairs, then the remaining indices will be filled with `-1`.

This output tensor can be manipulated to construct various featurizations of persistent homology (note: do this using Pytorch functionality to be able to backprop).  Examples can be found in `top_utils.py`

### Internal Details

The output of `computePersistence` is a set of persistence diagrams, `dgm`, which is a dictionary of numpy arrays of dimension `n x 4`, indexed by homology dimension, where `n` is the number of persistence pairs, and the indices are explained in the following table
| index  | 0  |  1 |  2 | 3  |
|:-:|:-:|:-:|:-:|:-:|
| information  |  birth time |  death time | birth vertex  |  death vertex |

The birth and death times are used to construct the output of a `TopologyLayer`, and the birth and death vertices are cached in the `forward` method for gradient computations.

## Super-level set persistence layer

The `DiagramLayer` in the `DiagramlayerTopLevel` submodule is used for super-level set persistence.
```python
from DiagramlayerTopLevel import Diagramlayer as LevelSetLayer
```
The input to this layer will be a single-channel image, and a `filtration`, which encodes the space.

```python
# image grid
width, height = 28, 28
axis_x = np.arange(0, width)
axis_y = np.arange(0, height)
grid_axes = np.array(np.meshgrid(axis_x, axis_y))
grid_axes = np.transpose(grid_axes, (1, 2, 0))
# creation of a filtration
from scipy.spatial import Delaunay
tri = Delaunay(grid_axes.reshape([-1, 2]))
faces = tri.simplices.copy()
filtration = LevelSetLayer().init_filtration(faces)
```

To apply the `LevelSetLayer` to an image `x`, use
```python
toplayer = LevelSetLayer()
dgm = toplayer(x, filtration)
```
`dgm` will be a pytorch tensor described in the Diagram Basics section.
