# ----- compiling on lukes laptop ----- #

ifeq ($(findstring luke, $(MAKECMDGOALS)), luke)

# set this command goal as a phony target (important)
.PHONY: luke

# define python location (if no source in venv, use: ln -s /usr/include/python3.8/ /home/path/to/env/name/include/python3.8)
PYTHON = /home/luke/pyenv/py38_general
PYTHON_EXE = $(PYTHON)/bin/python
PYTHON_INCLUDE = $(PYTHON)/include/python3.8
PYBIND_PATH = /home/luke/repo/pybind11
PYTORCH_PATH = /home/luke/repo/libtorch

endif

# ----- compiling on the lab PC ----- #

ifeq ($(filter lab, $(MAKECMDGOALS)), lab)

# set this command goal as a phony target (important)
.PHONY: lab

# define python location (if no source in venv, use: ln -s /usr/include/python3.8/ /home/path/to/env/name/include/python3.8)
PYTHON = /home/luke/pyenv/py38_general
PYTHON_EXE = $(PYTHON)/bin/python
PYTHON_INCLUDE = $(PYTHON)/include/python3.8
PYBIND_PATH = /home/luke/libs/pybind11
PYTORCH_PATH = /home/luke/repo/libtorch

endif