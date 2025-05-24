# DRL_Final
Final project for Deep Reinforcement Learning 2025@NTU

## install
pip install -r requirements.txt -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
sudo apt-get install doxygen
cd to abc_py
把pybind11 拉到abc_py下
sudo apt-get update
sudo apt-get install libboost-all-dev
mkdir build 
cd build 
cmake .. with path of your abc
example cmake .. -DABC_DIR=/home/dereklin1205/University/DRL/DRL_Final/abc_rl/abc_rl_dependency/abc


## HOW TO USE PYTHON RL ENVIRONMENT
My code is organized as follows
├── abc
├── abc.history
├── helper_functions.py
├── requirements.txt
├── rl_environment.py
├── runtime_results
│   └── current.aig
└── testbenches
    ├── 2025_IWLS_Contest_Benchmarks_020425
    └── EPFL

You will need the abc library that can be found here https://github.com/berkeley-abc/abc
Please install requirements.txt
Only use rl_environment.py. Please do not touch/use helper_functions.py. There are still errors.

The main function in rl_environment.py has a simple demo of how to use the code.






