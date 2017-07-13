! MIT License
!
! Copyright (c) 2016 Anders Steen Christensen, Lars A. Bratholm, Felix A. Faber
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

subroutine fget_local_kernels_gaussian(q1, q2, n1, n2, sigmas, &
        & nm1, nm2, nsigmas, kernels)

    implicit none

    double precision, dimension(:,:), intent(in) :: q1
    double precision, dimension(:,:), intent(in) :: q2

    ! List of numbers of atoms in each molecule
    integer, dimension(:), intent(in) :: n1
    integer, dimension(:), intent(in) :: n2

    ! Sigma in the Gaussian kernel
    double precision, dimension(:), intent(in) :: sigmas

    ! Number of molecules
    integer, intent(in) :: nm1
    integer, intent(in) :: nm2

    ! Number of sigmas
    integer, intent(in) :: nsigmas

    ! -1.0 / sigma^2 for use in the kernel
    double precision, dimension(nsigmas) :: inv_sigma2

    ! Resulting alpha vector
    double precision, dimension(nsigmas,nm1,nm2), intent(out) :: kernels

    ! Internal counters
    integer :: a, b, i, j, k, ni, nj

    ! Temporary variables necessary for parallelization
    double precision, allocatable, dimension(:,:) :: atomic_distance

    integer, allocatable, dimension(:) :: i_starts
    integer, allocatable, dimension(:) :: j_starts

    allocate(i_starts(nm1))
    allocate(j_starts(nm2))

    !$OMP PARALLEL DO
    do i = 1, nm1
        i_starts(i) = sum(n1(:i)) - n1(i)
    enddo
    !$OMP END PARALLEL DO

    !$OMP PARALLEL DO
    do j = 1, nm2
        j_starts(j) = sum(n2(:j)) - n2(j)
    enddo
    !$OMP END PARALLEL DO

    inv_sigma2(:) = -0.5d0 / (sigmas(:))**2
    kernels(:,:,:) = 0.0d0

    allocate(atomic_distance(maxval(n1), maxval(n2)))
    atomic_distance(:,:) = 0.0d0

    !$OMP PARALLEL DO PRIVATE(atomic_distance,ni,nj)
     do a = 1, nm1
        ni = n1(a)
        do b = 1, nm2
            nj = n2(b)

            atomic_distance(:,:) = 0.0d0
            do i = 1, ni
                do j = 1, nj

                    atomic_distance(i, j) = sum((q1(:,i + i_starts(a)) - q2(:,j + j_starts(b)))**2)

                enddo
            enddo

            do k = 1, nsigmas
                kernels(k, a, b) =  sum(exp(atomic_distance(:ni,:nj) * inv_sigma2(k)))
            enddo

        enddo
    enddo
    !$OMP END PARALLEL DO

    deallocate(atomic_distance)
    deallocate(i_starts)
    deallocate(j_starts)

end subroutine fget_local_kernels_gaussian

subroutine fget_local_kernels_laplacian(q1, q2, n1, n2, sigmas, &
        & nm1, nm2, nsigmas, kernels)

    implicit none

    double precision, dimension(:,:), intent(in) :: q1
    double precision, dimension(:,:), intent(in) :: q2

    ! List of numbers of atoms in each molecule
    integer, dimension(:), intent(in) :: n1
    integer, dimension(:), intent(in) :: n2

    ! Sigma in the Gaussian kernel
    double precision, dimension(:), intent(in) :: sigmas

    ! Number of molecules
    integer, intent(in) :: nm1
    integer, intent(in) :: nm2

    ! Number of sigmas
    integer, intent(in) :: nsigmas

    ! -1.0 / sigma^2 for use in the kernel
    double precision, dimension(nsigmas) :: inv_sigma2

    ! Resulting alpha vector
    double precision, dimension(nsigmas,nm1,nm2), intent(out) :: kernels

    ! Internal counters
    integer :: a, b, i, j, k, ni, nj

    ! Temporary variables necessary for parallelization
    double precision, allocatable, dimension(:,:) :: atomic_distance

    integer, allocatable, dimension(:) :: i_starts
    integer, allocatable, dimension(:) :: j_starts

    allocate(i_starts(nm1))
    allocate(j_starts(nm2))

    !$OMP PARALLEL DO
    do i = 1, nm1
        i_starts(i) = sum(n1(:i)) - n1(i)
    enddo
    !$OMP END PARALLEL DO

    !$OMP PARALLEL DO
    do j = 1, nm2
        j_starts(j) = sum(n2(:j)) - n2(j)
    enddo
    !$OMP END PARALLEL DO

    inv_sigma2(:) = -1.0d0 / sigmas(:)
    kernels(:,:,:) = 0.0d0

    allocate(atomic_distance(maxval(n1), maxval(n2)))
    atomic_distance(:,:) = 0.0d0

    !$OMP PARALLEL DO PRIVATE(atomic_distance,ni,nj)
     do a = 1, nm1
        ni = n1(a)
        do b = 1, nm2
            nj = n2(b)

            atomic_distance(:,:) = 0.0d0
            do i = 1, ni
                do j = 1, nj

                    atomic_distance(i, j) = sum(abs(q1(:,i + i_starts(a)) - q2(:,j + j_starts(b))))

                enddo
            enddo


            do k = 1, nsigmas
                kernels(k, a, b) =  sum(exp(atomic_distance(:ni,:nj) * inv_sigma2(k)))
            enddo

        enddo
    enddo
    !$OMP END PARALLEL DO

    deallocate(atomic_distance)
    deallocate(i_starts)
    deallocate(j_starts)

end subroutine fget_local_kernels_laplacian

subroutine fget_vector_kernels_laplacian(q1, q2, n1, n2, sigmas, &
        & nm1, nm2, nsigmas, kernels)

    implicit none

    ! Descriptors for the training set
    double precision, dimension(:,:,:), intent(in) :: q1
    double precision, dimension(:,:,:), intent(in) :: q2

    ! List of numbers of atoms in each molecule
    integer, dimension(:), intent(in) :: n1
    integer, dimension(:), intent(in) :: n2

    ! Sigma in the Gaussian kernel
    double precision, dimension(:), intent(in) :: sigmas

    ! Number of molecules
    integer, intent(in) :: nm1
    integer, intent(in) :: nm2

    ! Number of sigmas
    integer, intent(in) :: nsigmas

    ! -1.0 / sigma^2 for use in the kernel
    double precision, dimension(nsigmas) :: inv_sigma

    ! Resulting alpha vector
    double precision, dimension(nsigmas,nm1,nm2), intent(out) :: kernels

    ! Internal counters
    integer :: i, j, k, ni, nj, ia, ja

    ! Temporary variables necessary for parallelization
    double precision, allocatable, dimension(:,:) :: atomic_distance

    inv_sigma(:) = -1.0d0 / sigmas(:)

    kernels(:,:,:) = 0.0d0

    allocate(atomic_distance(maxval(n1), maxval(n2)))
    atomic_distance(:,:) = 0.0d0

    !$OMP PARALLEL DO PRIVATE(atomic_distance,ni,nj)
    do j = 1, nm2
        nj = n2(j)
        do i = 1, nm1
            ni = n1(i)

            atomic_distance(:,:) = 0.0d0

            do ja = 1, nj
                do ia = 1, ni

                    atomic_distance(ia,ja) = sum(abs(q1(:,ia,i) - q2(:,ja,j)))

                enddo
            enddo

            do k = 1, nsigmas
                kernels(k, i, j) =  sum(exp(atomic_distance(:ni,:nj) * inv_sigma(k)))
            enddo

        enddo
    enddo
    !$OMP END PARALLEL DO

    deallocate(atomic_distance)

end subroutine fget_vector_kernels_laplacian

subroutine fget_vector_kernels_gaussian(q1, q2, n1, n2, sigmas, &
        & nm1, nm2, nsigmas, kernels)

    implicit none

    ! ARAD descriptors for the training set, format (i,j_1,5,m_1)
    double precision, dimension(:,:,:), intent(in) :: q1
    double precision, dimension(:,:,:), intent(in) :: q2

    ! List of numbers of atoms in each molecule
    integer, dimension(:), intent(in) :: n1
    integer, dimension(:), intent(in) :: n2

    ! Sigma in the Gaussian kernel
    double precision, dimension(:), intent(in) :: sigmas

    ! Number of molecules
    integer, intent(in) :: nm1
    integer, intent(in) :: nm2

    ! Number of sigmas
    integer, intent(in) :: nsigmas

    ! -1.0 / sigma^2 for use in the kernel
    double precision, dimension(nsigmas) :: inv_sigma2

    ! Resulting alpha vector
    double precision, dimension(nsigmas,nm1,nm2), intent(out) :: kernels

    ! Internal counters
    integer :: i, j, k, ni, nj, ia, ja

    ! Temporary variables necessary for parallelization
    double precision, allocatable, dimension(:,:) :: atomic_distance

    inv_sigma2(:) = -0.5d0 / (sigmas(:))**2


    kernels(:,:,:) = 0.0d0

    allocate(atomic_distance(maxval(n1), maxval(n2)))
    atomic_distance(:,:) = 0.0d0

    !$OMP PARALLEL DO PRIVATE(atomic_distance,ni,nj,ja,ia)
    do j = 1, nm2
        nj = n2(j)
        do i = 1, nm1
            ni = n1(i)

            atomic_distance(:,:) = 0.0d0

            do ja = 1, nj
                do ia = 1, ni

                    atomic_distance(ia,ja) = sum((q1(:,ia,i) - q2(:,ja,j))**2)

                enddo
            enddo

            do k = 1, nsigmas
                kernels(k, i, j) =  sum(exp(atomic_distance(:ni,:nj) * inv_sigma2(k)))
            enddo

        enddo
    enddo
    !$OMP END PARALLEL DO

    deallocate(atomic_distance)

end subroutine fget_vector_kernels_gaussian


subroutine fgaussian_kernel(a, na, b, nb, k, sigma)

    implicit none

    double precision, dimension(:,:), intent(in) :: a
    double precision, dimension(:,:), intent(in) :: b

    integer, intent(in) :: na, nb

    double precision, dimension(:,:), intent(inout) :: k
    double precision, intent(in) :: sigma

    double precision, allocatable, dimension(:) :: temp

    double precision :: inv_sigma
    integer :: i, j

    inv_sigma = -0.5d0 / (sigma*sigma)

    allocate(temp(size(a, dim=1)))

!$OMP PARALLEL DO PRIVATE(temp)
    do i = 1, nb
        do j = 1, na
            temp(:) = a(:,j) - b(:,i)
            k(j,i) = exp(inv_sigma * sum(temp*temp))
        enddo
    enddo
!$OMP END PARALLEL DO

    deallocate(temp)

end subroutine fgaussian_kernel

subroutine flaplacian_kernel(a, na, b, nb, k, sigma)

    implicit none

    double precision, dimension(:,:), intent(in) :: a
    double precision, dimension(:,:), intent(in) :: b

    integer, intent(in) :: na, nb

    double precision, dimension(:,:), intent(inout) :: k
    double precision, intent(in) :: sigma

    double precision :: inv_sigma

    integer :: i, j

    inv_sigma = -1.0d0 / sigma

!$OMP PARALLEL DO
    do i = 1, nb
        do j = 1, na
            k(j,i) = exp(inv_sigma * sum(abs(a(:,j) - b(:,i))))
        enddo
    enddo
!$OMP END PARALLEL DO

end subroutine flaplacian_kernel


subroutine flinear_kernel(a, na, b, nb, k)

    implicit none

    double precision, dimension(:,:), intent(in) :: a
    double precision, dimension(:,:), intent(in) :: b

    integer, intent(in) :: na, nb

    double precision, dimension(:,:), intent(inout) :: k

    integer :: i, j

!$OMP PARALLEL DO
    do i = 1, nb
        do j = 1, na
            k(j,i) = dot_product(a(:,j), b(:,i))
        enddo
    enddo
!$OMP END PARALLEL DO

end subroutine flinear_kernel


subroutine fmatern_kernel_l2(a, na, b, nb, k, sigma, order)

    implicit none

    double precision, dimension(:,:), intent(in) :: a
    double precision, dimension(:,:), intent(in) :: b

    integer, intent(in) :: na, nb

    double precision, dimension(:,:), intent(inout) :: k
    double precision, intent(in) :: sigma
    integer, intent(in) :: order

    double precision, allocatable, dimension(:) :: temp

    double precision :: inv_sigma, inv_sigma2, d, d2
    integer :: i, j

    allocate(temp(size(a, dim=1)))

    if (order == 0) then
        inv_sigma = - 1.0d0 / sigma

        !$OMP PARALLEL DO PRIVATE(temp)
            do i = 1, nb
                do j = 1, na
                    temp(:) = a(:,j) - b(:,i)
                    k(j,i) = exp(inv_sigma * sqrt(sum(temp*temp)))
                enddo
            enddo
        !$OMP END PARALLEL DO
    else if (order == 1) then
        inv_sigma = - sqrt(3.0d0) / sigma

        !$OMP PARALLEL DO PRIVATE(temp, d)
            do i = 1, nb
                do j = 1, na
                    temp(:) = a(:,j) - b(:,i)
                    d = sqrt(sum(temp*temp))
                    k(j,i) = exp(inv_sigma * d) * (1.0d0 - inv_sigma * d)
                enddo
            enddo
        !$OMP END PARALLEL DO
    else
        inv_sigma = - sqrt(5.0d0) / sigma
        inv_sigma2 = 5.0d0 / (3.0d0 * sigma * sigma)

        !$OMP PARALLEL DO PRIVATE(temp, d, d2)
            do i = 1, nb
                do j = 1, na
                    temp(:) = a(:,j) - b(:,i)
                    d2 = sum(temp*temp)
                    d = sqrt(d2)
                    k(j,i) = exp(inv_sigma * d) * (1.0d0 - inv_sigma * d + inv_sigma2 * d2)
                enddo
            enddo
        !$OMP END PARALLEL DO
    end if

    deallocate(temp)

end subroutine fmatern_kernel_l2


subroutine fsargan_kernel(a, na, b, nb, k, sigma, gammas, ng)

    implicit none

    double precision, dimension(:,:), intent(in) :: a
    double precision, dimension(:,:), intent(in) :: b
    double precision, dimension(:), intent(in) :: gammas

    integer, intent(in) :: na, nb, ng

    double precision, dimension(:,:), intent(inout) :: k
    double precision, intent(in) :: sigma

    double precision, allocatable, dimension(:) :: prefactor
    double precision :: inv_sigma
    double precision :: d

    integer :: i, j, m

    inv_sigma = -1.0d0 / sigma

    ! Allocate temporary
    allocate(prefactor(ng))


    !$OMP PARALLEL DO PRIVATE(d, prefactor)
        do i = 1, nb
            do j = 1, na
                d = sum(abs(a(:,j) - b(:,i)))
                do m = 1, ng
                    prefactor(m) = gammas(m) * (- inv_sigma * d) ** m
                enddo
                k(j,i) = exp(inv_sigma * d) * (1 + sum(prefactor(:)))
            enddo
        enddo
    !$OMP END PARALLEL DO

    ! Clean up
    deallocate(prefactor)

end subroutine fsargan_kernel
