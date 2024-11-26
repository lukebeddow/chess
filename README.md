# Chess Engine

<img src="https://github.com/lukebeddow/gifs-and-resources/blob/main/chess.gif" width="20%" height="20%"/>

This repository implements a traditional chess engine which I have written in C++, as well as a deep learning framework to train a neural network evaluator function. The engine was a passion project, and so I decided I did not want to look up any advice or use established methods/algorithms beforehand. The final version of the engine is strong enough to beat me, which was my original aim for the project.

## Python

The ```python/``` folder contains the main script ```Pygame_interface.py``` which can spawn a GUI to play against. The ```modules``` folder contains compiled C++ code, and the ```testing``` contains testing code. This includes an implementation of the engine written entirely in Python.

## C++

The ```src/``` folder contains C++ implementation of the engine, which can be compiled into a terminal application, or into a Python module using Pybind11. Simply configure paths in the Makefile and then run ```make```, with the options of ```make cpp``` and ```make py``` to specify between C++ targets and compilation of Python modules.

## Learning

The ```src/``` folder contains a wrapper around Stockfish which can generate board evaluations to use as training data. Then the ```python/``` folder contains scripts such as ```assemble_data.py``` which creates and saves training data, and ```train_nn_evaluator.py``` which trains a neural network evaluator function.

## Building locally and playing the engines

To build the project on your local machine:
```bash
git clone https://github.com/lukebeddow/chess                                    # clone project to your preference
cd chess                                                                         # enter cloned repository
sudo apt-get update && apt-get install build-essential make g++ libboost-all-dev # install system dependencies
source /path/to/venv/bin/activate                                                # recommended: activate python virtual environment
pip install --no-cache-dir -r requirements.txt                                   # install python dependencies
make torch auto                                                                  # full build, traditional engine and nn evaluator
make all auto                                                                    # traditional engine build only
```

Now, you can play against the engines with:
```bash
python python/Pygame_interface.py                                                # spawns GUI to play chess in
bin/play_terminal                                                                # c++ only, terminal application
```

Further explanation of project dependencies:
* Build tools: ```apt-get update && apt-get install build-essential make g++ libboost-all-dev```
  * Make - used for building the project
  * g++ - used for compiling
  * boost - compilation dependency for c++ code
* Python (recommended to create a new virtual environment, and activate it first): ```pip install --no-cache-dir -r requirements.txt```
  * dill - for improved python pickling (serialising)
  * numpy & scipy - data handling and functionality
  * py_lz4framed - for efficient file compression
  * pygame - for chess GUI in python
  * torch - for learning in python and compilation dependency for c++
  * pybind11 - compilation dependency for c++
