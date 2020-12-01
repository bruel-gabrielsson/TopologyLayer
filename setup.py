# setup file to compile C++ library

from setuptools import setup
import torch, os
from torch.utils.cpp_extension import CppExtension, BuildExtension

this_dir = os.path.dirname(os.path.realpath(__file__))
include_dir = this_dir + '/topologylayer/functional/persistence/'
extra = {'cxx': ['-std=c++14']}

setup(name='topologylayer',
        packages=['topologylayer', 'topologylayer.functional',
        'topologylayer.nn', 'topologylayer.util'],
        ext_modules=[
                CppExtension('topologylayer.functional.persistence',
                        ['topologylayer/functional/persistence/pybind.cpp',
                        'topologylayer/functional/persistence/cohom.cpp',
                        'topologylayer/functional/persistence/hom.cpp',
                        'topologylayer/functional/persistence/complex.cpp',
                        'topologylayer/functional/persistence/cocycle.cpp'],
                        include_dirs=[include_dir],
                        extra_compile_args=extra['cxx']
                        )
        ],
        cmdclass={'build_ext': BuildExtension},
        zip_safe=False
)
