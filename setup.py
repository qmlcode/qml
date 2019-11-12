import sys
from numpy.distutils.core import Extension, setup

from mkldiscover import mkl_exists

__author__ = "Anders S. Christensen"
__copyright__ = "Copyright 2016"
__credits__ = ["Anders S. Christensen et al. (2016) https://github.com/qmlcode/qml"]
__license__ = "MIT"
__version__ = "0.4.0.12"
__maintainer__ = "Anders S. Christensen"
__email__ = "andersbiceps@gmail.com"
__status__ = "Beta"
__description__ = "Quantum Machine Learning"
__url__ = "https://github.com/qmlcode/qml"


FORTRAN = "f90"

# GNU (default)
COMPILER_FLAGS = ["-O3", "-fopenmp", "-m64", "-march=native", "-fPIC",
                    "-Wno-maybe-uninitialized", "-Wno-unused-function", "-Wno-cpp"]
LINKER_FLAGS = ["-lgomp"]
# MATH_LINKER_FLAGS = ["-lblas", "-llapack", "-latlas", "-fopenmp"]
MATH_LINKER_FLAGS = ["-lblas", "-llapack"]

# UNCOMMENT TO FORCE LINKING TO MKL with GNU compilers:
if mkl_exists(verbose=True):
    LINKER_FLAGS = ["-lgomp", "-lpthread", "-lm", "-ldl"]
    MATH_LINKER_FLAGS = ["-L${MKLROOT}/lib/intel64", "-lmkl_rt"]

# For clang without OpenMP: (i.e. most Apple/mac system)
if sys.platform == "darwin" and all(["gnu" not in arg for arg in sys.argv]):
    COMPILER_FLAGS = ["-O3", "-m64", "-march=native", "-fPIC"]
    LINKER_FLAGS = []
    # MATH_LINKER_FLAGS = ["-lblas", "-llapack", "-latlas", "-fopenmp"]
    MATH_LINKER_FLAGS = ["-lblas", "-llapack"]


# Intel
if any(["intelem" in arg for arg in sys.argv]):
    COMPILER_FLAGS = ["-xHost", "-O3", "-axAVX", "-qopenmp"]
    LINKER_FLAGS = ["-liomp5", "-lpthread", "-lm", "-ldl"]
    MATH_LINKER_FLAGS = ["-L${MKLROOT}/lib/intel64", "-lmkl_rt"]


ext_fkernels = Extension(name = '.kernels.fkernels',
                          sources = [
                          'qml/kernels/fkernels.f90',
                          'qml/kernels/fkpca.f90',
                              ],
                          extra_f90_compile_args = COMPILER_FLAGS,
                          extra_f77_compile_args = COMPILER_FLAGS,
                          extra_compile_args = COMPILER_FLAGS,
                          extra_link_args = LINKER_FLAGS + MATH_LINKER_FLAGS,
                          language = FORTRAN,
                          f2py_options=['--quiet'])

ext_fgradient_kernels = Extension(name = '.kernels.fgradient_kernels',
                          sources = ['qml/kernels/fgradient_kernels.f90'],
                          extra_f90_compile_args = COMPILER_FLAGS,
                          extra_f77_compile_args = COMPILER_FLAGS,
                          extra_compile_args = COMPILER_FLAGS,
                          extra_link_args = LINKER_FLAGS + MATH_LINKER_FLAGS,
                          language = FORTRAN,
                          f2py_options=['--quiet'])

ext_ffchl_module = Extension(name = '.fchl.ffchl_module',
                          sources = [
                              'qml/fchl/ffchl_module.f90',
                              'qml/fchl/ffchl_kernel_types.f90',
                              'qml/fchl/ffchl_kernels.f90',
                              'qml/fchl/ffchl_scalar_kernels.f90',
                              'qml/fchl/ffchl_force_kernels.f90',
                              'qml/fchl/ffchl_electric_field_kernels.f90',
                              ],
                          extra_f90_compile_args = COMPILER_FLAGS,
                          extra_f77_compile_args = COMPILER_FLAGS,
                          extra_compile_args = COMPILER_FLAGS ,
                          extra_link_args = LINKER_FLAGS + MATH_LINKER_FLAGS,
                          language = FORTRAN,
                          f2py_options=['--quiet'])

ext_farad_kernels = Extension(name = '.arad.farad_kernels',
                          sources = ['qml/arad/farad_kernels.f90'],
                          extra_f90_compile_args = COMPILER_FLAGS,
                          extra_f77_compile_args = COMPILER_FLAGS,
                          extra_compile_args = COMPILER_FLAGS,
                          extra_link_args = LINKER_FLAGS,
                          language = FORTRAN,
                          f2py_options=['--quiet'])

ext_frepresentations = Extension(name = '.representations.frepresentations',
                          sources = ['qml/representations/frepresentations.f90'],
                          extra_f90_compile_args = COMPILER_FLAGS,
                          extra_f77_compile_args = COMPILER_FLAGS,
                          extra_compile_args = COMPILER_FLAGS,
                          extra_link_args = MATH_LINKER_FLAGS + LINKER_FLAGS,
                          language = FORTRAN,
                          f2py_options=['--quiet'])

ext_fdistance = Extension(name = '.kernels.fdistance',
                          sources = ['qml/kernels/fdistance.f90'],
                          extra_f90_compile_args = COMPILER_FLAGS,
                          extra_f77_compile_args = COMPILER_FLAGS,
                          extra_compile_args = COMPILER_FLAGS,
                          extra_link_args = LINKER_FLAGS,
                          language = FORTRAN,
                          f2py_options=['--quiet'])

ext_fslatm = Extension(name = '.representations.fslatm',
                          sources = ['qml/representations/fslatm.f90'],
                          extra_f90_compile_args = COMPILER_FLAGS,
                          extra_f77_compile_args = COMPILER_FLAGS,
                          extra_compile_args = COMPILER_FLAGS,
                          extra_link_args = LINKER_FLAGS,
                          language = FORTRAN,
                          f2py_options=['--quiet'])

ext_fsolvers = Extension(name = '.math.fsolvers',
                          sources = ['qml/math/fsolvers.f90'],
                          extra_f90_compile_args = COMPILER_FLAGS,
                          extra_f77_compile_args = COMPILER_FLAGS,
                          extra_compile_args = COMPILER_FLAGS,
                          extra_link_args = MATH_LINKER_FLAGS + LINKER_FLAGS,
                          language = FORTRAN,
                          f2py_options=['--quiet'])

ext_facsf = Extension(name = '.representations.facsf',
                          sources = ['qml/representations/facsf.f90'],
                          extra_f90_compile_args = COMPILER_FLAGS,
                          extra_f77_compile_args = COMPILER_FLAGS,
                          extra_compile_args = COMPILER_FLAGS,
                          extra_link_args = LINKER_FLAGS,
                          language = FORTRAN,
                          f2py_options=['--quiet'])

def requirements():
    with open('requirements.txt') as f:
        return [line.rstrip() for line in f]


# use README.md as long description
def readme():
    with open('README.rst') as f:
        return f.read()


def setup_qml():

    setup(

        name="qml",
        packages=[
            'qml',
            'qml.aglaia',
            'qml.arad',
            'qml.fchl',
            'qml.kernels',
            'qml.math',
            'qml.representations',
            'qml.qmlearn',
            'qml.utils',
            'qml.models'
            ],

        # metadata
        version=__version__,
        author=__author__,
        author_email=__email__,
        platforms = 'Any',
        description = __description__,
        long_description = readme(),
        keywords = ['Machine Learning', 'Quantum Chemistry'],
        classifiers = [],
        url = __url__,
        install_requires = requirements(),

        # set up package contents

        ext_package = 'qml',
        ext_modules = [
              ext_ffchl_module,
              ext_fkernels,
              ext_fgradient_kernels,
              ext_frepresentations,
              ext_fslatm,
              ext_fsolvers,
              ext_facsf,
              ext_fdistance,
              ext_farad_kernels,
        ],
)


if __name__ == '__main__':

    setup_qml()
