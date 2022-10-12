import os
from setuptools import setup, find_packages

version_ns = {}
with open(os.path.join("magnetopy", "version.py")) as f:
    exec(f.read(), version_ns)
version = version_ns["__version__"]

setup(
    name="magnetopy",
    version=version,
    packages=find_packages(),
    description="Magnetism in Python",
    long_description=(
        """MagnetoPy is a collection of code to help with the analysis of 
        magnetometry data."""
    ),
    install_requires=[
        "numpy>=1.23.3",
        "pandas>=1.5.0",
        "scipy>=1.9.2",
        "matplotlib>=3.6.1",
    ],
    python_requires=">=3.10.1",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Chemistry",
    ],
    keywords=[
        "magnetism",
        "chemistry",
        "materials science",
        "data analysis",
        "utility",
    ],
    license="MIT License",
    url="https://github.com/pcb7445/MAGdb",
)
