from os.path import join, dirname, realpath
from setuptools import setup, __version__
import sys

assert sys.version_info.major == 3 and sys.version_info.minor >= 6, \
    "The Spinning Up repo is designed to work with Python 3.6 and greater." \
    + "Please install it before proceeding."

with open(join("spinup", "version.py")) as version_file:
    exec(version_file.read())

setup(
    name='spinup',
    py_modules=['spinup'],
    version=__version__,#'0.1',
    install_requires=[
        'cloudpickle==1.2.1',
        'gym[atari,box2d,classic_control]>=0.10.8',
        'ipython',
        'joblib',
        'matplotlib==3.1.1',
        'mpi4py',
        'numpy',
        'pandas',
        'pytest',
        'psutil',
        'scipy',
        'seaborn>=0.8.1',
        'tensorflow>=1.8.0,<2.0',
        'tqdm',
        'networkx',
        'tensorboard',
        'pyglet'
    ],
    description="Flexibility Design with Neural Reinforcement Learning developed based on Spinningup from OpenAI.",
    author="Lei Zhang",
)
