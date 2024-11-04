# Chess Engine

<img src="https://github.com/lukebeddow/gifs-and-resources/blob/main/chess.gif" width="20%" height="20%"/>

This repository implements a traditional chess engine which I have written in C++, as well as a deep learning framework to train a neural network evaluator function. The engine was a passion project, and so I decided I did not want to look up any advice or use established methods/algorithms beforehand. The final version of the engine is strong enough to beat me, which was my original aim for the project.

## Python

The ```python/``` folder contains the main script ```Pygame_interface.py``` which can spawn a GUI to play against. The ```modules``` folder contains compiled C++ code, and the ```testing``` contains testing code. This includes an implementation of the engine written entirely in Python.

## C++

The ```src/``` folder contains C++ implementation of the engine, which can be compiled into a terminal application, or into a Python module using Pybind11. Simply configure paths in the Makefile and then run ```make```, with the options of ```make cpp``` and ```make py``` to specify between C++ targets and compilation of Python modules.

## Learning

The ```src/``` folder contains a wrapper around Stockfish which can generate board evaluations to use as training data. Then the ```python/``` folder contains scripts such as ```assemble_data.py``` which creates and saves training data, and ```train_nn_evaluator.py``` which trains a neural network evaluator function.
