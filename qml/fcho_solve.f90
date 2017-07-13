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
        write (*,*) "WARNING: Cholesky decomposition DPOTRF() exited with error code:", info
    endif

    x(:na) = y(:na)

    call dpotrs("U", na, 1, A, na, x, na, info)
    if (info > 0) then
        write (*,*) "WARNING: Cholesky solve DPOTRS() exited with error code:", info
    endif

end subroutine fcho_solve

subroutine fcho_invert(A)

    implicit none

    double precision, dimension(:,:), intent(inout) :: A
    integer :: info, na

    na = size(A, dim=1)

    call dpotrf("L", na, A , na, info)
    if (info > 0) then
        write (*,*) "WARNING: Cholesky decomposition DPOTRF() exited with error code:", info
    endif

    call dpotri("L", na, A , na, info )
    if (info > 0) then
        write (*,*) "WARNING: Cholesky inversion DPOTRI() exited with error code:", info
    endif

end subroutine fcho_invert


subroutine fbkf_invert(A)

    implicit none

    double precision, dimension(:,:), intent(inout) :: A
    integer :: info, na, nb

    integer, dimension(size(A,1)) :: ipiv   ! pivot indices
    integer :: ilaenv
    
    integer :: lwork

    double precision, allocatable, dimension(:) :: work

    na = size(A, dim=1)
    
    nb = ilaenv( 1, 'DSYTRF', "L", na, -1, -1, -1 )

    lwork = na*nb

    allocate(work(lwork))

    ! call dpotrf("L", na, A , na, info)
    call dsytrf("L", na, A, na, ipiv, work, lwork, info)
    if (info > 0) then
        write (*,*) "WARNING: Bunch-Kaufman factorization DSYTRI() exited with error code:", info
    endif

    ! call dpotri("L", na, A , na, info )
    call dsytri( "L", na, a, na, ipiv, work, info )
    if (info > 0) then
        write (*,*) "WARNING: BKF inversion DPOTRI() exited with error code:", info
    endif

    deallocate(work)

end subroutine fbkf_invert


subroutine fbkf_solve(A,y,x)

    implicit none

    double precision, dimension(:,:), intent(in) :: A
    double precision, dimension(:), intent(in) :: y
    double precision, dimension(:), intent(inout) :: x

    double precision, allocatable, dimension(:) :: work
    integer :: ilaenv

    integer, dimension(size(A,1)) :: ipiv   ! pivot indices
    integer :: info, na, nb, lwork

    na = size(A, dim=1)

    nb = ilaenv( 1, 'DSYTRF', "L", na, -1, -1, -1 )

    lwork = na*nb
    allocate(work(lwork))

    call dsytrf("L", na, A, na, ipiv, work, lwork, info)
    if (info > 0) then
        write (*,*) "WARNING: Bunch-Kaufman factorization DSYTRI() exited with error code:", info
    endif

    x(:na) = y(:na)

    call dsytrs("L", na, 1, A, na, ipiv, x, na, info )

    if (info > 0) then
        write (*,*) "WARNING: Bunch-Kaufman solver DSYTRS() exited with error code:", info
    endif

    deallocate(work)
end subroutine fbkf_solve
