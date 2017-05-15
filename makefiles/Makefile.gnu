OBJECTS = \
	qml/math/fcho_solve.so \
	qml/distance/fdistance.so \
	qml/representations/frepresentations.so \
	qml/kernels/farad_kernels.so \
	qml/kernels/fkernels.so

# Flags for GCC compilers and system BLAS/LAPACK
COMPILER_FLAGS = --opt='-O3 -fopenmp -m64 -march=native' --f90flags=''
LINKER_FLAGS = -lgomp -lpthread -lm -ldl
MATH_LINKER_FLAGS = -lblas -llapack

# F2PY executable
F2PY_EXEC = f2py

all: $(OBJECTS)

qml/math/fcho_solve.so: qml/math/fcho_solve.f90
	$(F2PY_EXEC) -c -m fcho_solve qml/math/fcho_solve.f90 $(COMPILER_FLAGS) $(LINKER_FLAGS) $(MATH_LINKER_FLAGS)
	mv fcho_solve*.so qml/math/fcho_solve.so

qml/distance/fdistance.so: qml/distance/fdistance.f90
	$(F2PY_EXEC) -c -m fdistance qml/distance/fdistance.f90 $(COMPILER_FLAGS) $(LINKER_FLAGS)
	mv fdistance*.so qml/distance/fdistance.so

qml/representations/frepresentations.so: qml/representations/frepresentations.f90
	$(F2PY_EXEC) -c -m frepresentations qml/representations/frepresentations.f90 $(COMPILER_FLAGS) $(LINKER_FLAGS)
	mv frepresentations*.so qml/representations/frepresentations.so

qml/kernels/fkernels.so: qml/kernels/fkernels.f90
	$(F2PY_EXEC) -c -m fkernels qml/kernels/fkernels.f90 $(COMPILER_FLAGS) $(LINKER_FLAGS)
	mv fkernels*.so qml/kernels/fkernels.so

qml/kernels/farad_kernels.so: qml/kernels/farad_kernels.f90
	$(F2PY_EXEC) -c -m farad_kernels qml/kernels/farad_kernels.f90 $(COMPILER_FLAGS) $(LINKER_FLAGS)  $(MATH_LINKER_FLAGS)
	mv farad_kernels*.so qml/kernels/farad_kernels.so

clean:
	rm -f qml/*.pyc
	rm -f qml/math/*.so
	rm -f qml/math/*.pyc
	rm -f qml/distance/*.so
	rm -f qml/distance/*.pyc
	rm -f qml/kernels/*.so
	rm -f qml/kernels/*.pyc
	rm -f qml/representations/*.so
	rm -f qml/representations/*.pyc
