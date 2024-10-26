from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name="Weather Data Collector",
    ext_modules=cythonize("weather_extractor/extractor.pyx"),
    include_dirs=[numpy.get_include()],
    install_requires=[
        'requests',
        'lxml',
        'pandas',
        'numpy',
        'cython'
    ]
)