# C++ library

To build, you need to have cloned the git repository with submodules so the pybind11 submodule is in this directory.

If you didn't do this, run
```bash
git submodule update --init
```

## Compiling python bindings

```bash
cd alternate_implementation
mkdir build
cd build
cmake -DPYBIND11_PYTHON_VERSION=2.7 ..
make
```
You can also compile for a different python version by modifying the above command.
