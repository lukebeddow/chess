# ----- description ----- #

# This Makefile compiles code which interfaces with mujoco physics simulator.
# There are two key targets, firstly a c++ compilation and secondly a python
# compilation. The c++ compilation results in an executable aimed at testing.
# The python compilation results in a python module that can be imported and
# used from within python.

# Useful resources:
#		https://www.gnu.org/software/make/manual/html_node/Automatic-Variables.html
# 	https://www.gnu.org/software/make/manual/html_node/Static-Usage.html#Static-Usage
# 	https://www.hiroom2.com/2016/09/03/makefile-header-dependencies/
# 	https://stackoverflow.com/questions/16924333/makefile-compiling-from-directory-to-another-directory

# ----- user defined variables ----- #

# define the targets (must compile from a .cpp file with same name in SOURCEDIR)
TARGET_LIST_CPP := play_terminal dr_memory_build test_terminal stockfish
TARGET_LIST_PY := board_module tree_module stockfish_module

# define directory structure
SOURCEDIR := src
BUILDDIR := build
BUILDPY := $(BUILDDIR)/py
BUILDCPP := $(BUILDDIR)/cpp
BUILDDEP := $(BUILDDIR)/depends
OUTCPP := bin
OUTPY := python/modules

# are we compiling in debug mode
ifeq ($(filter debug, $(MAKECMDGOALS)), debug)
DEBUG = -O0 -g
else
DEBUG = -O2
endif

# define python location (if no source in venv, use: ln -s /usr/include/python3.8/ /home/path/to/env/name/include/python3.8)
PYTHON = /home/luke/pyenv/py38_general
PYTHON_EXE = $(PYTHON)/bin/python
PYTHON_INCLUDE = $(PYTHON)/include/python3.8
PYBIND_PATH = /home/luke/repo/pybind11

# define compiler flags and libraries
COMMON = $(DEBUG) -std=c++14 -mavx -pthread -I$(PYTHON_INCLUDE) \
	-I$(PYBIND_PATH)/include \
	-Wl,-rpath,'$$ORIGIN'
PYBIND = $(COMMON) -fPIC -Wall -shared -DLUKE_PYBIND

# extra flags for make -jN => use N parallel cores
MAKEFLAGS += -j8

# ----- automatically generated variables ----- #

# get every source file and each corresponding dependecy file
SOURCES := $(wildcard $(SOURCEDIR)/*.cpp)
DEPENDS := $(patsubst $(SOURCEDIR)/%.cpp, $(BUILDDEP)/%.d, $(SOURCES))

# define targets in the output directory
CPPTARGETS := $(patsubst %, $(OUTCPP)/%, $(TARGET_LIST_CPP))
PYTARGETS := $(patsubst %, $(OUTPY)/%.so, $(TARGET_LIST_PY))

# seperate source files for targets and those for shared objects
CPPTARGETSRC := $(patsubst %, $(SOURCEDIR)/%.cpp, $(TARGET_LIST_CPP))
PYTARGETSRC := $(patsubst %, $(SOURCEDIR)/%.cpp, $(TARGET_LIST_PY))
SOURCES := $(filter-out $(CPPTARGETSRC), $(SOURCES))
SOURCES := $(filter-out $(PYTARGETSRC), $(SOURCES))

# define target object files and shared object files
CPPTARGETOBJ := $(patsubst %, $(BUILDDIR)/%.o, $(TARGET_LIST_CPP))
PYTARGETOBJ := $(patsubst %, $(BUILDDIR)/%.o, $(TARGET_LIST_PY))
CPPSHAREDOBJ := $(patsubst $(SOURCEDIR)/%.cpp, $(BUILDCPP)/%.o, $(SOURCES))
PYSHAREDOBJ := $(patsubst $(SOURCEDIR)/%.cpp, $(BUILDPY)/%.o, $(SOURCES))

# create the directories if they don't exist
DIRS := $(BUILDDIR) $(BUILDPY) $(BUILDCPP) $(BUILDDEP) $(OUTCPP) $(OUTPY)
$(info $(shell mkdir -p $(DIRS)))

# ----- start of make ----- #

all: cpp py
cpp: $(CPPTARGETS) $(DEPENDS)
py: $(PYTARGETS) $(DEPENDS)

.PHONY: test
test:
	echo python targets are $(PYTARGETS)
	echo c++ targets are $(CPPTARGETS)

.PHONY: debug
debug: cpp

# build object files
$(CPPSHAREDOBJ): $(BUILDCPP)/%.o : $(SOURCEDIR)/%.cpp
	g++ $(COMMON) -c $< -o $@
$(PYSHAREDOBJ): $(BUILDPY)/%.o : $(SOURCEDIR)/%.cpp
	g++ $(COMMON) -c -fPIC $< -o $@
$(CPPTARGETOBJ): $(BUILDDIR)/%.o : $(SOURCEDIR)/%.cpp
	g++ $(COMMON) -c $< -o $@
$(PYTARGETOBJ): $(BUILDDIR)/%.o : $(SOURCEDIR)/%.cpp
	g++ $(PYBIND) -c $< -o $@

# build targets
$(CPPTARGETS): $(OUTCPP)% : $(BUILDDIR)%.o $(CPPSHAREDOBJ)
	g++ $(COMMON) $^ -o $@
$(PYTARGETS): $(OUTPY)%.so : $(BUILDDIR)%.o $(PYSHAREDOBJ)
	g++ $(PYBIND) $^ -o $@

# if not cleaning, declare the dependencies of each object file (headers and source)
ifneq ($(filter clean, $(MAKECMDGOALS)), clean)
include $(DEPENDS)
endif

# generate dependency files (-M for all, -MM for exclude system dependencies)
$(BUILDDEP)/%.d: $(SOURCEDIR)/%.cpp
	@set -e; rm -f $@; \
		g++ -MM $(COMMON) $< > $@.$$$$; \
		sed 's,\($*\)\.o[ :]*,$(BUILDDIR)/\1.o $(BUILDCPP)/\1.o $(BUILDPY)/\1.o $@ : ,g' \
			< $@.$$$$ > $@;
		rm -f $@.*

clean:
	rm -f $(BUILDDIR)/*.o 
	rm -f $(BUILDCPP)/*.o 
	rm -f $(BUILDPY)/*.o 
	rm -f $(BUILDDIR)/*.d
	rm -f $(CPPTARGETS)
	rm -f $(PYTARGETS)
