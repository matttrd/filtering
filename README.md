# filtering

This is a repository with minimal examples. We plan to provide a docker image for future releases.

The requirements of IGE are libcudf and NVIDIA Thrust.

The source code of the filtering algorithm is not provided as we are considering to patent it. However, we provide the shared library libgraph-image.so

Before installing the module please export the libs:

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{your-env-path}/lib

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{your-repo-path}/lib

The python module is created by wrapping the cpp code with pybind11.
To install the python module, run:

python setup.py install

In order to run all python code, please install the robustness library by Madry et al. via the commands:
pip install robustness

To folder "python" contains the following files:
- test_filtering.py: this is a minimal code example that filters a test image;
- adaptive_attacks.py: this file is used to attack pretrained models (at a given resolution);


The other .py files are used by the aforementioned files.
