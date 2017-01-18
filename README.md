# SPN-Z-SL
Structure Learning for Continuous Variables in SPN-Z

Used for all experiments in Online Structure Learning for Sum-Product Networks
Created by Wilson Hsu and Agastya Kalra.

#### Dependencies:
- Scipy

- Numpy

- Python 3.3+

#### Usage:

To run any of the experiments in the dataset first create a folder in the library directory called SPN-Z-SL/data/dataset_name/.

Then create 10 files of the dataset as follows:

dataset_name.1.data ... dataset_name.10.data.

then add ~/path/to/SPN-Z-SL/ to your PYTHONPATH.

finally in experiments/read_data.py, add the line to run an experiment. The file explains the parameters.

then in the libraries main directory type:

python3 experiment/real_data.py


REAL_NVP Experiments: https://github.com/KalraA/real-nvp/
