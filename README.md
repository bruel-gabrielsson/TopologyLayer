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
