# Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at the
# Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights reserved.
# See file COPYRIGHT for details.
#
# This file is part of the MFEM library. For more information and source code
# availability see http://mfem.org.
#
# MFEM is free software; you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.

# Use the MFEM build directory.

MFEM_DIR ?= ../../builds/MFEM
CONFIG_MK = $(MFEM_DIR)/share/mfem/config.mk

MFEM_LIB_FILE = mfem_is_not_built
-include $(CONFIG_MK)

# Enumerate code directories in logical order.

PRO_DIR = systems
BUI_DIR = build
MET_DIR = methods
AUX_DIR = auxiliaries
OUT_DIR = output

DIRS = $(PRO_DIR) $(MET_DIR) $(AUX_DIR) $(FP_DIR) #$(OP_DIR) $(INT_DIR)

AUX_FILES = $(BUI_DIR)/dofs.o
MET_FILES = $(BUI_DIR)/feevol.met $(BUI_DIR)/loworder.met $(BUI_DIR)/mcl.met
PRO_FILES = $(BUI_DIR)/system.pro $(BUI_DIR)/m1.pro
# List scalar problems.

# PRO_FILES += $(BUI_DIR)/advection.pro $(BUI_DIR)/buckleyleverett.pro $(BUI_DIR)/burgers.pro $(BUI_DIR)/kpp.pro

# List systems.

#PRO_FILES += $(BUI_DIR)/euler.pro $(BUI_DIR)/advection.pro

MAIN_FILES = ex_moments

# ifeq ($(MFEM_USE_MPI), YES)
   #MAIN_FILES += pimpcg
   #PAUX_FILES += $(BUI_DIR)/pdofs.o
   #PAUX_FILES += $(BUI_DIR)/ptools.o
   #PMET_FILES += $(BUI_DIR)/pfixedpointiteration.met
#endif

# Setup valgrind test.

PROBLEM = 0
VALGRIND-CONFIG = -tf 0.001 -dt 0.001 -p $(PROBLEM) -e 0

## Remember some makefile rules.

# List keywords that are not associated with files (by default, all are).

.PHONY: all library impcg clean valgrind-test grid-convergence-test style # pimpcg inzufügen hinter impcg

# Delete the default suffixes.

.SUFFIXES:

# Define suffixes.

.SUFFIXES: .c .cpp

# Replace the default implicit rule for *.cpp files.

%: %.cpp $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) $< -ggdb -o $@ $(AUX_FILES) $(PRO_FILES) $(MFEM_LIBS)

%.o: ../$(AUX_DIR)/%.cpp
	$(MFEM_CXX) $(MFEM_FLAGS) -c $^ -ggdb -o $@

%.met: ../$(MET_DIR)/%.cpp
	$(MFEM_CXX) $(MFEM_FLAGS) -c $^ -ggdb -o $@

%.pro: ../$(PRO_DIR)/%.cpp
	$(MFEM_CXX) $(MFEM_FLAGS) -c $^ -ggdb -o $@

%.int: ../$(INT_DIR)/%.cpp
	$(MFEM_CXX) $(MFEM_FLAGS) -c $^ -ggdb -o $@

%.fp: ../$(FP_DIR)/%.cpp
	$(MFEM_CXX) $(MFEM_FLAGS) -c $^ -ggdb -o $@

%.op: ../$(OP_DIR)/%.cpp
	$(MFEM_CXX) $(MFEM_FLAGS) -c $^ -ggdb -o $@

all: $(MAIN_FILES)

library: $(AUX_FILES) $(PAUX_FILES) $(MET_FILES) $(PMET_FILES) $(PRO_FILES)
ex_moments: library
	$(MFEM_CXX) $(MFEM_FLAGS) ex_moments.cpp -ggdb -o $@ $(AUX_FILES) $(MET_FILES) $(PRO_FILES) $(MFEM_LIBS)

clean:
	@rm -f ex_moments errors.txt *gf* grid* $(BUI_DIR)/*

# TODO: Does not work yet.

#valgrind-parallel:
#	# mpirun -np 2 valgrind --leak-check=yes ./par-dgstab -tf 0.1 -r 0
#	# valgrind --gen-suppressions=yes pdgstab $(VALGRIND-CONFIG)
#	# valgrind --suppressions=$(MPI_HOME)/share/openmpi/openmpi-valgrind.supp pdgstab $(VALGRIND-CONFIG)
#	valgrind --suppressions=./my-mpi-suppressions.supp pdgstab $(VALGRIND-CONFIG)

# Generate an error message if the MFEM library is not built and exit.

$(MFEM_LIB_FILE):
	$(error The MFEM library is not built)

ASTYLE = astyle --options=$(MFEM_DIR)/config/mfem.astylerc
FORMAT_FILES = $(foreach dir,$(DIRS),"$(dir)/*.?pp")
FORMAT_FILES += ex_moments.cpp

style:
	@if ! $(ASTYLE) $(FORMAT_FILES) | grep Formatted; then\
	   echo "No source files were changed.";\
	fi
