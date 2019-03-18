#from distutils.core import setup, Extension
from setuptools import setup, Extension,find_packages
from setuptools.command.build_ext import build_ext
from distutils.ccompiler import get_default_compiler

from Cython.Build import cythonize
#from Cython.Distutils import build_ext

import numpy
include_dirs = ['liblbfgs', numpy.get_include()]
ext_modules = cythonize(
        [Extension('confounded_glm.fitting._lowlevel',
                   ['confounded_glm/fitting/_lowlevel.pyx', 'liblbfgs/lbfgs.c'],
                   include_dirs=include_dirs),
        Extension('confounded_glm.fitting._fit',
                   ['confounded_glm/fitting/_fit.pyx'],
                   include_dirs = [numpy.get_include()] )
        ])


class custom_build_ext(build_ext):
    def finalize_options(self):
        build_ext.finalize_options(self)
        if self.compiler is None:
            compiler = get_default_compiler()
        else:
            compiler = self.compiler

        if compiler == 'msvc':
            include_dirs.append('compat/win32')

setup(
  name='confounded_glm',
  version='0.0',
  description='packages for confounded GLMs',
  #url='http://github.com/storborg/funniest',
  author='Weigutian Ou',
  license='MIT',
  #packages=['confounded_glm'],
  packages=find_packages(),
  ext_modules = ext_modules,
  cmdclass={"build_ext": custom_build_ext},
  author_email='weigutian.ou@gmail.com'
)
