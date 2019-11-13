module ffchl_kernel_types
    implicit none

    public

    ! define kernel enums
    integer, parameter :: GAUSSIAN = 1
    integer, parameter :: LINEAR = 2
    integer, parameter :: POLYNOMIAL = 3
    integer, parameter :: SIGMOID = 4
    integer, parameter :: MULTIQUADRATIC = 5
    integer, parameter :: INV_MULTIQUADRATIC = 6
    integer, parameter :: BESSEL = 7
    integer, parameter :: L2 = 8
    integer, parameter :: MATERN = 9
    integer, parameter :: CAUCHY = 10
    integer, parameter :: POLYNOMIAL2 = 11

end module ffchl_kernel_types