# Unit Tests for the topologylayer module

These tests assume that `topologylayer` has been put on the python path.

Unit tests use the [built-in python `unittest` module](https://docs.python.org/2.7/library/unittest.html)

## Run All tests

From this directory (`TopologyLayer/tests`), run
```bash
python -m unittest discover -p '*.py'
```

## Run individual tests

To run unit-tests, test `file.py` using
```bash
python -m unittest -v file # no *.py
```
for example
```bash
cd cpp
python -m unittest -v simplicial_complex
```
