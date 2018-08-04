from distutils.core import setup
from Cython.Build import cythonize
import numpy
from distutils.extension import Extension

ext_mods = cythonize([Extension("hawkeslib.model.c.c_uv_exp", ["hawkeslib/model/c/c_uv_exp.pyx"], include_dirs=[numpy.get_include()],
                                       libraries=["m"],
                                       language="c++",
                                       extra_compile_args=["-O3", "-ffast-math", "-march=native", "-fopenmp"],
                                       extra_link_args=['-fopenmp']),
                            Extension("hawkeslib.model.c.c_mv_exp", ["hawkeslib/model/c/c_mv_exp.pyx"], include_dirs=[numpy.get_include()],
                                       libraries=["m"],
                                       language="c++",
                                       extra_compile_args=["-O3", "-march=native", "-fopenmp"],
                                       extra_link_args=['-fopenmp']),
                             Extension("hawkeslib.model.c.c_uv_bayes", ["hawkeslib/model/c/c_uv_bayes.pyx"], include_dirs=[numpy.get_include()],
                                       libraries=["m"],
                                       extra_compile_args=["-O3", "-ffast-math", "-march=native"])
                             ])

setup(name="hawkeslib",
      version="0.1.1",
      description="parameter estimation for simple Hawkes (self-exciting) processes",
      author="Caner Turkmen",
      author_email="caner.turkmen@boun.edu.tr",
      url="http://hawkeslib.rtfd.io",
      ext_modules=ext_mods,
      packages=["hawkeslib", "hawkeslib.model", "hawkeslib.model.c"]
)

