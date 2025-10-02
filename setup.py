"""Setup file for Cython compilation."""

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

extensions = [
    Extension(
        "rnd_extraction.cy_kernels",
        ["rnd_extraction/cy_kernels.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-march=native", "-ffast-math"],
        language="c",
    ),
    Extension(
        "mv_rnd.cbrapipe._fast",
        ["notebooks/mv_rnd/cbrapipe/_fast.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-march=native"],
        language="c",
    ),
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "initializedcheck": False,
        },
    ),
)
