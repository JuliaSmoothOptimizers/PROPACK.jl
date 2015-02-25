ifeq ($(OS), Windows_NT)
    SLIB = dll
else
    UNAME := $(shell uname)
    ifeq ($(UNAME), Darwin)
        SLIB = dylib
    else
        SLIB = so
    endif
endif

MGS = mgs.risc
FC = gfortran
FFLAG = -xf77-cpp-input
BLAS = -lblas
LAPACK = -llapack
LDFLAGS = -fPIC $(LAPACK) $(BLAS)
