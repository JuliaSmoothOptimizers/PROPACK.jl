ifeq ($(OS), Windows_NT)
    SLIB = dll
    BLAS = -lopenblas
    LAPACK = -lopenblas
else
    UNAME := $(shell uname)
    ifeq ($(UNAME), Darwin)
        SLIB = dylib
    else
        SLIB = so
    endif
    BLAS = -lblas
    LAPACK = -llapack
endif

MGS = mgs.risc
FC = gfortran
FFLAG = -xf77-cpp-input
LDFLAGS = -fPIC $(LAPACK) $(BLAS)
