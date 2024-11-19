# ----- description ----- #

# This Makefile compiles my chess engine into c++ executables and python modules

# Useful resources:
#		https://www.gnu.org/software/make/manual/html_node/Automatic-Variables.html
# 	https://www.gnu.org/software/make/manual/html_node/Static-Usage.html#Static-Usage
# 	https://www.hiroom2.com/2016/09/03/makefile-header-dependencies/
# 	https://stackoverflow.com/questions/16924333/makefile-compiling-from-directory-to-another-directory

# ----- user defined variables ----- #

# define the targets (must compile from a .cpp file with same name in SOURCEDIR)
TARGET_LIST_CPP := play_terminal dr_memory_build test_terminal
TARGET_LIST_PY := board_module tree_module stockfish_module
# TARGET_LIST_PY := torch_module

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

ifeq ($(filter profile, $(MAKECMDGOALS)), profile)
DEBUG = -O0 -pg
else
DEBUG = -O2
endif

# define library locations - this file contains user specified options
include buildsettings.mk

# define compiler flags and libraries
COMMON = $(DEBUG) -std=c++17 -mavx -pthread -I$(PYTHON_INCLUDE) \
	-I$(PYBIND_PATH)/include \
	-I$(PYTORCH_PATH)/include \
	-I$(PYTORCH_PATH)/include/torch/csrc/api/include \
	-L$(PYTORCH_PATH)/lib \
	-DLUKE_PYTORCH \
	-D_GLIBCXX_USE_CXX11_ABI=0 \
	-Wl,-rpath,$$ORIGIN:$(PYTORCH_PATH)/lib
PYBIND = $(COMMON) -fPIC -Wall -shared -DLUKE_PYBIND

# COMMON = $(DEBUG) -std=c++17 -mavx -pthread -I$(PYTHON_INCLUDE)\
# 	-I$(PYBIND_PATH)/include \
# 	-I$(PYTORCH_PATH)/include \
# 	-I$(PYTORCH_PATH)/include/torch/csrc/api/include \
# 	-L$(PYTORCH_PATH)/lib \
# 	-DLUKE_PYTORCH \
# 	-Wl,-rpath,'$$ORIGIN'
# PYBIND = $(COMMON) -fPIC -Wall -shared -DLUKE_PYBIND
# LIBS = -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda

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

.PHONY: profile
profile: cpp

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
	g++ $(COMMON) $^ -o $@ $(LIBS) 
$(PYTARGETS): $(OUTPY)%.so : $(BUILDDIR)%.o $(PYSHAREDOBJ)
	g++ $(PYBIND) $^ -o $@ $(LIBS)

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
