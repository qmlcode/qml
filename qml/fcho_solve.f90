! MIT License
!
! Copyright (c) 2016 Anders Steen Christensen
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

subroutine fcho_solve(A,y,x)

    implicit none

    double precision, dimension(:,:), intent(in) :: A
    double precision, dimension(:), intent(in) :: y
    double precision, dimension(:), intent(inout) :: x
    
    integer :: info, na

    na = size(A, dim=1)

    call dpotrf("U", na, A, na, info)
    if (info > 0) then
        write (*,*) "WARNING: Error in LAPACK Cholesky decomposition DPOTRF()."
        write (*,*) "WARNING: The", info, "-th leading order is not positive definite."
    else if (info < 0) then
        write (*,*) "WARNING: Error in LAPACK Cholesky decomposition DPOTRF()."
        write (*,*) "WARNING: The", -info, "-th argument had an illegal value."
    endif

    x(:na) = y(:na)

    call dpotrs("U", na, 1, A, na, x, na, info)
    if (info < 0) then
        write (*,*) "WARNING: Error in LAPACK Cholesky solver DPOTRS()."
        write (*,*) "WARNING: The", -info, "-th argument had an illegal value."
    endif

end subroutine fcho_solve

! subroutine fcho_invert(A)
! 
!     implicit none
! 
!     double precision, dimension(:,:), intent(inout) :: A
!     integer :: info, na
! 
!     na = size(A, dim=1)
! 
!     call dpotrf("L", na, A , na, info)
!     if (info > 0) then
!         write (*,*) "WARNING: Cholesky decomposition DPOTRF() exited with error code:", info
!     endif
! 
!     call dpotri("L", na, A , na, info )
!     if (info > 0) then
!         write (*,*) "WARNING: Cholesky inversion DPOTRI() exited with error code:", info
!     endif
! 
! end subroutine fcho_invert
