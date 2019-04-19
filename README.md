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
conda install pytorch torchvision
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

# High-Level Interface

For easiest use, high-level classes are provided for Pytorch compatibility.

The output of the diagram layers is not just a Pytorch tensor, but a `dgminfo` object, which currently consists of the diagram pytorch tensor and a boolean flag indicating the filtration direction, which is necessary for featurization

The recommended usage is to just pass `dgminfo` directly into a feature layer, which will take care of parsing, since the representation may change in the future.

### LevelSetLayer

A `LevelSetLayer` takes in an image (of fixed dimension), and outputs a super-level set persistence diagram tensor.

```python
from levelset import LevelSetLayer
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
from rips import RipsLayer

layer = RipsLayer(maxdim=1, rmax=np.inf, verbose=True)
dgm, issublevelset = layer(x)
```

### SumBarcodeLengths

A `SumBarcodeLengths` layer takes in a `dgminfo` object, and sums up the lengths of the persistence pairs, ignoring infinite bars, and handling dimension padding

```python
from levelset import LevelSetLayer
from features import SumBarcodeLengths
import torch

layer = LevelSetLayer((28,28), maxdim=1)
sumlayer = SumBarcodeLengths()

x = torch.rand(28,28)
dgminfo = layer(x)
dlen = sumlayer(dgminfo)
```


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
