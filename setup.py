from distutils.core import setup
from Cython.Build import cythonize
import numpy
from distutils.extension import Extension

setup(
    ext_modules=cythonize([Extension("em", ["em.pyx"], include_dirs=[numpy.get_include()],
                                        libraries=["m"],
                                        extra_compile_args = ["-O3", "-ffast-math", "-march=native", "-fopenmp"],
                                        extra_link_args=['-fopenmp']),
                             Extension("model.c.c_uv_exp", ["model/c/c_uv_exp.pyx"], include_dirs=[numpy.get_include()],
                                       libraries=["m"],
                                       language="c++",
                                       extra_compile_args=["-O3", "-ffast-math", "-march=native", "-fopenmp"],
                                       extra_link_args=['-fopenmp']),
                             Extension("model.c.c_uv_bayes", ["model/c/c_uv_bayes.pyx"], include_dirs=[numpy.get_include()],
                                       libraries=["m"],
                                       extra_compile_args=["-O3", "-ffast-math", "-march=native"])
                             ])
)

