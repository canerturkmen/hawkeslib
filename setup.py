import io
import os
try:
    from setuptools import setup, Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension

try:
    from Cython.Build import cythonize
    import numpy
except:
    raise SystemExit("Cython>=0.28 and numpy are required. Please install before proceeding")

REQUIRED = ["numpy>=1.14", "Cython>=0.28", "scipy>=1.1", "numdifftools>=0.9"]
EXTRA_REQUIRED = {"test": ["mock", "nose"], "docs": ["Sphinx", "sphinx-rtd-theme>=0.4"]}
DESCRIPTION = "parameter estimation for simple Hawkes (self-exciting) processes"
CLASSIFIERS = [
    'Programming Language :: Python',
    'Programming Language :: Python :: 2',
    'Programming Language :: Cython',
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: MIT License',
    'Topic :: Scientific/Engineering',
    'Topic :: Scientific/Engineering :: Information Analysis'
]

# make description
try:
    here = os.path.abspath(os.path.dirname(__file__))
    with io.open(os.path.join(here, 'README.markdown'), encoding='utf-8') as f:
        LONG_DESCRIPTION = '\n' + f.read()
except:
    LONG_DESCRIPTION = DESCRIPTION

# cythonize extensions
ext_mods = cythonize(
    [Extension("hawkeslib.model.c.c_uv_exp", ["hawkeslib/model/c/c_uv_exp.pyx"], include_dirs=[numpy.get_include()],
               libraries=["m"],
               language="c++",
               extra_compile_args=["-O3", "-march=native"]),
    Extension("hawkeslib.model.c.c_mv_exp", ["hawkeslib/model/c/c_mv_exp.pyx"], include_dirs=[numpy.get_include()],
               libraries=["m"],
               language="c++",
               extra_compile_args=["-O3", "-march=native"]),
    Extension("hawkeslib.model.c.c_uv_bayes", ["hawkeslib/model/c/c_uv_bayes.pyx"], include_dirs=[numpy.get_include()],
               libraries=["m"],
               extra_compile_args=["-O3", "-march=native"]),
    Extension("hawkeslib.model.c.c_mv_samp", ["hawkeslib/model/c/c_mv_samp.pyx"], include_dirs=[numpy.get_include()],
               libraries=["m"],
               language="c++",
               extra_compile_args=["-O3", "-march=native"])
    ])

setup(name="hawkeslib",
      version="0.2.2",
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      long_description_content_type="text/markdown",
      author="Caner Turkmen",
      author_email="caner.turkmen@boun.edu.tr",
      url="http://hawkeslib.rtfd.io",
      ext_modules=ext_mods,
      packages=["hawkeslib", "hawkeslib.model", "hawkeslib.model.c", "hawkeslib.util"],
      install_requires=REQUIRED,
      extras_require=EXTRA_REQUIRED,
      setup_requires=["numpy", "cython", "scipy"],
      license="MIT",
      python_requires=">=2.7.5",
      classifiers=CLASSIFIERS
)
