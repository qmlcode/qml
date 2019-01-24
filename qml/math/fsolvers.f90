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


subroutine fqrlq_solve(A, y, la, x)

    implicit none

    double precision, dimension(:,:), intent(inout) :: A
    double precision, dimension(:), intent(inout):: y
    integer, intent(in):: la

    double precision, allocatable, dimension(:,:) :: b
    double precision, dimension(la), intent(out) :: x

    integer :: m, n, nrhs, lda, ldb, info

    integer :: lwork
    double precision, dimension(:), allocatable :: work

    m = size(A, dim=1)
    n = size(A, dim=2)


    nrhs = 1
    lda = m
    ldb = max(m,n)
    
    allocate(b(ldb,1))
    b = 0.0d0
    b(:m,1) = y(:m)

    lwork = (min(m,n) + max(m,n)) * 10
    allocate(work(lwork))

    ! write (*,*) info, lda, ldb, m, n
    call dgels("N", m, n, nrhs, a, lda, b, ldb, work, lwork, info)

    if (info < 0) then
        write (*,*) "QML WARNING: Could not perform QRLQ solver DGELS: info =", info
    else if (info > 0) then
        write (*,*) "QML WARNING: QRLQ solver (DGELS) the", -info, "th"
        write (*,*) "diagonal element of the triangular factor of A is zero,"
        write (*,*) "so that A does not have full rank; the least squares"
        write (*,*) "solution could not be computed."
    endif
    
    x(:n) = b(:n,1)    
    
end subroutine fqrlq_solve


subroutine fsvd_solve(A, y, la, rcond, x)
    
    implicit none

    double precision, dimension(:,:), intent(inout) :: A
    double precision, dimension(:), intent(inout):: y
    integer, intent(in):: la
    double precision, intent(in) :: rcond 

    double precision, allocatable, dimension(:,:) :: b
    double precision, dimension(la), intent(out) :: x

    integer :: m, n, nrhs, lda, ldb, info

    integer :: lwork
    integer :: liwork
    double precision, dimension(:), allocatable :: work

    integer, dimension(:), allocatable :: iwork

    double precision, dimension(:), allocatable :: s
    integer :: rank

    m = size(A, dim=1)
    n = size(A, dim=2)

    nrhs = 1
    lda = m
    ldb = max(m,n)

    allocate(b(ldb,1))
    b = 0.0d0
    b(:m,1) = y(:m)


    allocate(s(ldb))
    ! rcond = 0.0d0

    allocate(work(1))
    allocate(iwork(1))

    lwork = -1
    call dgelsd(m, n, nrhs, a, lda, b, ldb, s, rcond, rank, work, lwork, iwork, info)

    liwork = int(iwork(1))
    lwork  = int(work(1))

    deallocate(work)
    deallocate(iwork)
    allocate(work(lwork))
    allocate(iwork(liwork))

    call dgelsd(m, n, nrhs, a, lda, b, ldb, s, rcond, rank, work, lwork, iwork, info)

    ! write (*,*) "DGELSD(): matrix rank =", rank, "/", n
    ! write (*,*) "DGELSD(): matrix rcond =", rcond
    ! write (*,*) "DGELSD(): matrix cond =", 1.0d0/rcond

    ! allocate(x(m))
    x(:n) = b(:n,1)

end subroutine fsvd_solve


subroutine fcond(A, rcond)

    implicit none

    double precision, dimension(:,:), intent(inout) :: A
    double precision, intent(out) :: rcond

    double precision :: anorm
    character, parameter :: norm = "1"
    character, parameter :: uplo = "U"

    double precision, allocatable, dimension(:) :: work
    integer, allocatable, dimension(:) :: iwork


    double precision, allocatable, dimension(:) :: A_diag

    integer :: info, na, nb
    integer :: n, lda
    integer :: i

    double precision :: dlansy

    na = size(A, dim=1)
    nb = size(A, dim=2)

    n = na
    lda = na

    ! Save diagonal
    allocate(a_diag(n))
    do i = 1, na
        a_diag(i) = a(i,i)
    enddo

    allocate(work(n))
    anorm = dlansy(norm, uplo, n, a, lda, work )
    deallocate(work)


    ! Cholesky factorization

    call dpotrf("U", n, A , lda, info)
    if (info > 0) then
        write (*,*) "WARNING: Cholesky decompositon failed because A is not positive definite. info = ", info
    else if (info < 0) then
        write (*,*) "WARNING: Cholesky decompositon DPOTRF() failed. info = ", info
    endif


    ! Condition number from Cholesky factorization

    allocate(work(n*3))
    allocate(iwork(n))

    call dpocon(uplo, n, a, lda, anorm, rcond, work, iwork, info)
    if (info < 0) then
        write (*,*) "WARNING: Calculating condition number DPOCON() failed. info = ", info
    endif

    deallocate(work)
    deallocate(iwork)

    ! Restore lower triangle and diagonal
    do i = 1, na
        a(i,i) = a_diag(i)
        a(i,i+1:) = a(i+1:,i)
    enddo

    deallocate(a_diag)

    rcond = 1.0d0 / rcond

end subroutine fcond


subroutine fcond_ge(K, rcond)

    implicit none

    double precision, dimension(:,:), intent(in) :: K
    double precision, intent(out) :: rcond

    double precision :: anorm
    character, parameter :: norm = "1"
    ! character, parameter :: uplo = "U"

    double precision, allocatable, dimension(:) :: work
    double precision, allocatable, dimension(:,:) :: A
    integer, allocatable, dimension(:) :: iwork
    integer, allocatable, dimension(:) :: ipiv

    integer :: info
    integer :: n, m, lda

    double precision :: dlange

    m = size(K, dim=1)
    n = size(K, dim=2)

    allocate(A(m,n))

    A(:,:) = K(:,:)

    lda = n

    allocate(work(max(m,n)))
    anorm = dlange( norm, m, n, a, lda, work )
    deallocate(work)

    allocate(ipiv(min(m,n)))
    call dgetrf( m, n, a, lda, ipiv, info )
    deallocate(ipiv)

    if (info > 0) then
        write (*,*) "WARNING: LU-decompositon failed because A is exactly singular. info = ", info
    else if (info < 0) then
        write (*,*) "WARNING: LU-decompositon DGETRF() failed. info = ", info
    endif

    allocate(work(n*4))
    allocate(iwork(n))
    call dgecon( norm, n, a, lda, anorm, rcond, work, iwork, info )

     if (info < 0) then
        write (*,*) "WARNING: Calculating condition number DGECON() failed. info = ", info
    endif

    deallocate(work)
    deallocate(iwork)
    deallocate(a)

    rcond = 1.0d0 / rcond

end subroutine fcond_ge

