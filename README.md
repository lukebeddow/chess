# Chess Engine

This repository implements a chess engine which I have written, in both C++ and Python. This was a passion project, and so I decided I did not want to look up any advice or use established methods/algorithms beforehand.

## Python

The ```python/``` folder contains the main script ```Pygame_interface.py``` which can spawn a GUI to play against. The ```modules``` folder contains compiled C++ code, and the ```testing``` contains testing code. This includes an implementation of the engine written entirely in Python.

## C++

The ```src/``` folder contains C++ implementation of the engine, which can be compiled into a terminal application, or into a Python module using Pybind11. Simply configure paths in the Makefile and then run ```make```, with the options of ```make cpp``` and ```make py``` to specify between C++ targets and compilation of Python modules.
