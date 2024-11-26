# ----- special case: automatic compilation ----- #

ifeq ($(filter auto, $(MAKECMDGOALS)), auto)

# set this command goal as a phony target (important)
.PHONY: auto

# now assemble the compiler flags for includes (-I), library paths (-L), and libraries (-libxxx)

# pybind should return the python headers and the pybind headers (requires pip install pybind11)
NORMAL_INCLUDES = $(shell python3 -m pybind11 --includes)
NORMAL_LIBRARIES = 
NORMAL_LIB_FILES = 

# torch depends on the python installation (pip install torch)
PYTORCH_PATH = $(shell python3 -c "import torch; import os; print(os.path.dirname(torch.__file__))")
TORCH_INCLUDES = -I$(PYTORCH_PATH)/include \
								 -I$(PYTORCH_PATH)/include/torch/csrc/api/include
TORCH_LIBRARIES = -L$(PYTORCH_PATH)/lib
TORCH_LIB_FILES = -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda

DEFAULT_VARIABLES = -DLUKE_ROOTPATH='"$(shell python3 -c "import os; print(os.getcwd())")"'

# ----- all other cases ----- #
else

# ----- compiling on lukes laptop ----- #

ifeq ($(findstring luke, $(MAKECMDGOALS)), luke)

# set this command goal as a phony target (important)
.PHONY: luke

# define python location (if no source in venv, use: ln -s /usr/include/python3.8/ /home/path/to/env/name/include/python3.8)
PATH_HERE = /home/luke/chess
PYTHON = /home/luke/pyenv/py38_general
PYTHON_EXE = $(PYTHON)/bin/python
PYTHON_INCLUDE = $(PYTHON)/include/python3.8
PYBIND_PATH = /home/luke/repo/pybind11
# PYTORCH_PATH = /home/luke/repo/libtorch
PYTORCH_PATH = $(PYTHON)/lib/python3.8/site-packages/torch

# ----- compiling on the lab PC ----- #

else ifeq ($(filter lab, $(MAKECMDGOALS)), lab)

# set this command goal as a phony target (important)
.PHONY: lab

# define python location (if no source in venv, use: ln -s /usr/include/python3.8/ /home/path/to/env/name/include/python3.8)
PATH_HERE = /home/luke/chess
PYTHON = /home/luke/pyenv/py38_general
PYTHON_EXE = $(PYTHON)/bin/python
PYTHON_INCLUDE = $(PYTHON)/include/python3.8
PYBIND_PATH = /home/luke/libs/pybind11
# PYTORCH_PATH = /home/luke/repo/libtorch
PYTORCH_PATH = $(PYTHON)/lib/python3.8/site-packages/torch

endif

# ----- for normal cases (excluding only docker), now assemble the standard compiler flags ----- #

# now assemble the compiler flags for includes (-I), library paths (-L), and libraries (-libxxx)
NORMAL_INCLUDES = -I$(PYTHON_INCLUDE) \
									-I$(PYBIND_PATH)/include
NORMAL_LIBRARIES = 
NORMAL_LIB_FILES = 

# torch depends on the python installation (pip install torch)
TORCH_INCLUDES = -I$(PYTORCH_PATH)/include \
								 -I$(PYTORCH_PATH)/include/torch/csrc/api/include
TORCH_LIBRARIES = -L$(PYTORCH_PATH)/lib
TORCH_LIB_FILES = -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda

DEFAULT_VARIABLES = -DLUKE_ROOTPATH='"$(PATH_HERE)"'

endif

# ----- now finalise compiler flags ----- #

ifeq ($(filter torch, $(MAKECMDGOALS)), torch)

# ----- are we compiling with pytorch ----- #

INCLUDES = $(NORMAL_INCLUDES) $(TORCH_INCLUDES)
LIBRARIES = $(NORMAL_LIBRARIES) $(TORCH_LIBRARIES)
LIB_FILES = $(NORMAL_LIB_FILES) $(TORCH_LIB_FILES)
VARIABLES = $(DEFAULT_VARIABLES) -DLUKE_PYTORCH -D_GLIBCXX_USE_CXX11_ABI=0
RPATH = $$ORIGIN:$(PYTORCH_PATH)/lib

else

# ----- or normal, lightweight, no pytorch build ----- #

INCLUDES = $(NORMAL_INCLUDES)
LIBRARIES = $(NORMAL_LIBRARIES)
LIB_FILES = $(NORMAL_LIB_FILES)
VARIABLES = $(DEFAULT_VARIABLES)
RPATH = $$ORIGIN

endif