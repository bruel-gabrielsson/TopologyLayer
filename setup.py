# setup file to compile C++ library

from setuptools import setup
import torch, os
from torch.utils.cpp_extension import CppExtension, BuildExtension

this_dir = os.path.dirname(os.path.realpath(__file__))
include_dir = this_dir + '/topologylayer/functional/cohom_cpp/'
extra = {'cxx': ['-std=c++11']} #, '-D_GLIBCXX_USE_CXX11_ABI=1'

setup(name='topologylayer',
        packages=['topologylayer', 'topologylayer.functional',
        'topologylayer.nn', 'topologylayer.util'],
        ext_modules=[
                CppExtension('topologylayer.functional.cohom_cpp',
                        ['topologylayer/functional/cohom_cpp/pybind.cpp',
                        'topologylayer/functional/cohom_cpp/cohom.cpp',
                        'topologylayer/functional/cohom_cpp/complex.cpp',
                        'topologylayer/functional/cohom_cpp/cocycle.cpp'],
                        include_dirs=[include_dir],
                        extra_compile_args=extra['cxx']
                        )
        ],
        cmdclass={'build_ext': BuildExtension},
        zip_safe=False
)
