! MIT License
!
! Copyright (c) 2018 Anders Steen Christensen and Felix A. Faber
!
! Permission is hereby granted, free of charge, to any person obtaining a copy
! of this software and associated documentation files (the "Software"), to deal
! in the Software without restriction, including without limitation the rights
! to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
! copies of the Software, and to permit persons to whom the Software is
! furnished to do so, subject to the following conditions:
!
! The above copyright notice and this permission notice shall be included in all
! copies or substantial portions of the Software.
!
! THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
! IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
! FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
! AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
! LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
! OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
! SOFTWARE.

! Inspiration from:
! http://crsouza.com/2010/03/17/kernel-functions-for-machine-learning-applications/#kernel_functions
module ffchl_kernels

    implicit none

    public :: kernel

contains


subroutine gaussian_kernel(s11, s22, s12, parameters, k)

    implicit none

    double precision, intent(in) :: s11
    double precision, intent(in) :: s22
    double precision, intent(in) :: s12
    double precision, intent(in), dimension(:,:) :: parameters

    double precision, intent(out), dimension(:) :: k

    integer :: i
    double precision :: l2

    l2 = s11 + s22 - 2.0d0*s12 

    do i = 1, size(k)
        k(i) = exp(l2 * parameters(i,1)) 
    enddo

end subroutine gaussian_kernel 


subroutine linear_kernel(s12, parameters, k)

    implicit none

    double precision, intent(in) :: s12
    double precision, intent(in), dimension(:,:) :: parameters
    double precision, intent(out), dimension(:) :: k

    integer :: i

    do i = 1, size(k)
        k(i) = s12 + parameters(i,1)
    enddo

end subroutine linear_kernel 


subroutine polynomial_kernel(s12, parameters, k)

    implicit none

    double precision, intent(in) :: s12
    double precision, intent(in), dimension(:,:) :: parameters

    double precision, intent(out), dimension(:) :: k

    integer :: i

    do i = 1, size(k)
        k(i) = (parameters(i,1) * s12 + parameters(i,2))**parameters(i,3)
    enddo

end subroutine polynomial_kernel 


subroutine sigmoid_kernel(s12, parameters, k)

    implicit none
    double precision, intent(in) :: s12
    double precision, intent(in), dimension(:,:) :: parameters

    double precision, intent(out), dimension(:) :: k

    integer :: i

    do i = 1, size(k)
        k(i) = tanh(parameters(i,1) * s12 + parameters(i,2))
    enddo

end subroutine sigmoid_kernel 


subroutine multiquadratic_kernel(s11, s22, s12, parameters, k)

    implicit none

    double precision, intent(in) :: s11
    double precision, intent(in) :: s22
    double precision, intent(in) :: s12
    double precision, intent(in), dimension(:,:) :: parameters

    double precision, intent(out), dimension(:) :: k

    integer :: i
    double precision :: l2
    
    l2 = s11 + s22 - 2.0d0*s12 

    do i = 1, size(k)
        k(i) = sqrt(l2 + parameters(i,1)**2)
    enddo

end subroutine multiquadratic_kernel 


subroutine inverse_multiquadratic_kernel(s11, s22, s12, parameters, k)

    implicit none

    double precision, intent(in) :: s11
    double precision, intent(in) :: s22
    double precision, intent(in) :: s12
    double precision, intent(in), dimension(:,:) :: parameters

    double precision, intent(out), dimension(:) :: k

    integer :: i
    double precision :: l2
    
    l2 = s11 + s22 - 2.0d0*s12 

    do i = 1, size(k)
        k(i) = 1.0d0 / sqrt(l2 + parameters(i,1)**2)
    enddo

end subroutine inverse_multiquadratic_kernel 


subroutine bessel_kernel(s12, parameters, k)

    implicit none

    double precision, intent(in) :: s12
    double precision, intent(in), dimension(:,:) :: parameters

    double precision, intent(out), dimension(:) :: k

    integer :: i
    
    do i = 1, size(k)
        k(i) = BESSEL_JN(int(parameters(i,2)), parameters(i,1) *s12) & 
            & / (s12**(-parameters(i,3)*(parameters(i,2) + 1)))
    enddo

end subroutine bessel_kernel 

subroutine l2_kernel(s11, s22, s12, parameters, k)

    implicit none

    double precision, intent(in) :: s11
    double precision, intent(in) :: s22
    double precision, intent(in) :: s12
    double precision, intent(in), dimension(:,:) :: parameters

    double precision, intent(out), dimension(:) :: k

    integer :: i
    double precision :: l2
    
    l2 = s11 + s22 - 2.0d0*s12 

    do i = 1, size(k)
        k(i) = l2*parameters(i,1) + parameters(i,2)
    enddo

end subroutine l2_kernel 


subroutine matern_kernel(s11, s22, s12, parameters, kernel)

    implicit none

    double precision, intent(in) :: s11
    double precision, intent(in) :: s22
    double precision, intent(in) :: s12
    double precision, intent(in), dimension(:,:) :: parameters

    double precision, intent(out), dimension(:) :: kernel
    
    double precision :: l2

    double precision :: rho

    integer :: i, k, n
    double precision:: v, fact

    l2 = sqrt(s11 + s22 - 2.0d0*s12)

    kernel(:) = 0.0d0    

    do i = 1, size(kernel)
        n = int(parameters(i,2))
        v = n + 0.5d0

        rho = 2.0d0 * sqrt(2.0d0 * v) * l2  / parameters(i,1) 

        do k = 0, n

            fact = parameters(i,3+k)

            kernel(i) = kernel(i) + exp(-0.5d0 * rho) * fact * rho**(n-k)

        enddo
    enddo

end subroutine matern_kernel 


subroutine cauchy_kernel(s11, s22, s12, parameters, k)

    implicit none

    double precision, intent(in) :: s11
    double precision, intent(in) :: s22
    double precision, intent(in) :: s12
    double precision, intent(in), dimension(:,:) :: parameters

    double precision, intent(out), dimension(:) :: k

    integer :: i
    double precision :: l2
    
    l2 = s11 + s22 - 2.0d0*s12 

    do i = 1, size(k)
        k(i) = 1.0d0 /(1.0d0 + l2/parameters(i,1)**2)
    enddo

end subroutine cauchy_kernel 

subroutine polynomial2_kernel(s12, parameters, k)

    implicit none

    double precision, intent(in) :: s12
    double precision, intent(in), dimension(:,:) :: parameters

    double precision, intent(out), dimension(:) :: k

    integer :: i
    
    do i = 1, size(k)
        k(i) = parameters(i,  1) &
           & + parameters(i,  2) * s12 &
           & + parameters(i,  3) * s12**2 &
           & + parameters(i,  4) * s12**3 &
           & + parameters(i,  5) * s12**4 &
           & + parameters(i,  6) * s12**5 &
           & + parameters(i,  7) * s12**6 &
           & + parameters(i,  8) * s12**7 &
           & + parameters(i,  9) * s12**8 &
           & + parameters(i, 10) * s12**9
    enddo
    
    
end subroutine polynomial2_kernel

function kernel(s11, s22, s12, kernel_idx, parameters) result(k)

    implicit none

    double precision, intent(in) :: s11
    double precision, intent(in) :: s22
    double precision, intent(in) :: s12
    integer, intent(in) :: kernel_idx
    double precision, intent(in), dimension(:,:) :: parameters

    integer :: n
    double precision, allocatable, dimension(:) :: k

    n = size(parameters, dim=1)
    allocate(k(n))

    if (kernel_idx == 1) then
        call gaussian_kernel(s11, s22, s12, parameters, k)

    else if (kernel_idx == 2) then
        call linear_kernel(s12, parameters, k)
    
    else if (kernel_idx == 3) then
        call polynomial_kernel(s12, parameters, k)

    else if (kernel_idx == 4) then
        call sigmoid_kernel(s12, parameters, k)

    else if (kernel_idx == 5) then
        call multiquadratic_kernel(s11, s22, s12, parameters, k)

    else if (kernel_idx == 6) then
        call inverse_multiquadratic_kernel(s11, s22, s12, parameters, k)

    else if (kernel_idx == 7) then
        call bessel_kernel(s12, parameters, k)
    
    else if (kernel_idx == 8) then
        call l2_kernel(s11, s22, s12, parameters, k)
    
    else if (kernel_idx == 9) then
        call matern_kernel(s11, s22, s12, parameters, k)
    
    else if (kernel_idx == 10) then
        call cauchy_kernel(s11, s22, s12, parameters, k)
    
    else if (kernel_idx == 11) then
        call polynomial2_kernel(s12, parameters, k)

    else
        write (*,*) "QML ERROR: Unknown kernel function requested:", kernel_idx
        stop
    endif


end function kernel

end module ffchl_kernels
