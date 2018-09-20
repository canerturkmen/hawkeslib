import io
import os
from distutils.core import setup
from Cython.Build import cythonize
import numpy
from distutils.extension import Extension

REQUIRED = [
    "numpy>=1.14", "Cython>=0.28", "scipy>=1.1", "numdifftools>=0.9",
    "theano", "pymc3"
]

EXTRA_REQUIRED = ["mock", "nose", "Sphinx", "sphinx-rtd-theme>=0.4"]

DESCRIPTION = "parameter estimation for simple Hawkes (self-exciting) processes"

# make description
try:
    here = os.path.abspath(os.path.dirname(__file__))
    with io.open(os.path.join(here, 'README.markdown'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except:
    long_description = DESCRIPTION

# cythonize extensions
ext_mods = cythonize([Extension("hawkeslib.model.c.c_uv_exp", ["hawkeslib/model/c/c_uv_exp.pyx"], include_dirs=[numpy.get_include()],
                                       libraries=["m"],
                                       language="c++",
                                       extra_compile_args=["-O3", "-ffast-math", "-march=native"]),
                            Extension("hawkeslib.model.c.c_mv_exp", ["hawkeslib/model/c/c_mv_exp.pyx"], include_dirs=[numpy.get_include()],
                                       libraries=["m"],
                                       language="c++",
                                       extra_compile_args=["-O3", "-march=native"]),
                             Extension("hawkeslib.model.c.c_uv_bayes", ["hawkeslib/model/c/c_uv_bayes.pyx"], include_dirs=[numpy.get_include()],
                                       libraries=["m"],
                                       extra_compile_args=["-O3", "-ffast-math", "-march=native"])
                             ])

setup(name="hawkeslib",
      version="0.1.3",
      description=DESCRIPTION,
      author="Caner Turkmen",
      author_email="caner.turkmen@boun.edu.tr",
      url="http://hawkeslib.rtfd.io",
      ext_modules=ext_mods,
      packages=["hawkeslib", "hawkeslib.model", "hawkeslib.model.c"],
      install_required=REQUIRED,
      extras_require=EXTRA_REQUIRED,
      license="MIT",
      python_requires="2.7"
)

