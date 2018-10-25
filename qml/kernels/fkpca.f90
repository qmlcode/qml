! MIT License
!
! Copyright (c) 2018 Anders Steen Christensen
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


subroutine fkpca(k, n, centering, kpca)

    implicit none

    double precision, dimension(:,:), intent(in) :: k
    integer, intent(in) :: n
    logical, intent(in) :: centering
    double precision, dimension(n,n), intent(out) :: kpca

    ! Eigenvalues
    double precision, dimension(n) :: eigenvals

    double precision, allocatable, dimension(:) :: work

    integer :: lwork
    integer :: info

    integer :: i

    double precision :: inv_n
    double precision, allocatable, dimension(:) :: temp
    double precision :: temp_sum

    kpca(:,:) = k(:,:)

    ! This first part centers the matrix,
    ! basically Kpca = K - G@K - K@G + G@K@G, with G = 1/n
    ! It is a bit hard to follow, sry, but it is very fast
    ! and requires very little memory overhead.

    if (centering) then

        inv_n = 1.0d0 / n

        allocate(temp(n))
        temp(:) = 0.0d0

        !$OMP PARALLEL DO
        do i = 1, n
            temp(i) = sum(k(i,:)) * inv_n
        enddo
        !$OMP END PARALLEL DO

        temp_sum = sum(temp(:)) * inv_n

        !$OMP PARALLEL DO
        do i = 1, n
            kpca(i,:) = kpca(i,:) + temp_sum
        enddo
        !$OMP END PARALLEL DO

        !$OMP PARALLEL DO
        do i = 1, n
            kpca(:,i) = kpca(:,i) - temp(i)
        enddo
        !$OMP END PARALLEL DO

        !$OMP PARALLEL DO
        do i = 1, n
            kpca(i,:) = kpca(i,:) - temp(i)
        enddo
        !$OMP END PARALLEL DO

        deallocate(temp)

    endif

    ! This 2nd part solves the eigenvalue problem with the least
    ! memory intensive solver, namely DSYEV(). DSYEVD() is twice
    ! as fast, but requires a lot more memory, which quickly
    ! becomes prohibitive.

    ! Dry run which returns the optimal "lwork"
    allocate(work(1))
    call dsyev("V", "U", n, kpca, n, eigenvals, work, -1, info)
    lwork = nint(work(1)) + 1
    deallocate(work)

    ! Get eigenvectors
    allocate(work(lwork))
    call dsyev("V", "U", n, kpca, n, eigenvals, work, lwork, info)
    deallocate(work)

    if (info < 0) then

        write (*,*) "ERROR: The ", -info, "-th argument to DSYEV() had an illegal value."

    else if (info > 0) then

        write (*,*) "ERROR: DSYEV() failed to compute an eigenvalue."

    end if

    ! This 3rd part sorts the kernel PCA matrix such that the first PCA is kpca(1)
    kpca = kpca(:,n:1:-1)
    kpca = transpose(kpca)

    !$OMP PARALLEL DO
    do i = 1, n
        kpca(i,:) = kpca(i,:) * sqrt(eigenvals(n - i + 1))
    enddo
    !$OMP END PARALLEL DO

end subroutine fkpca
