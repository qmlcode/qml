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


subroutine fget_kernels_fchl(x1, x2, n1, n2, nneigh1, nneigh2, &
       & sigmas, nm1, nm2, nsigmas, &
       & t_width, d_width, cut_start, cut_distance, order, pd, &
       & distance_scale, angular_scale, alchemy, two_body_power, three_body_power, kernels)

    use ffchl_module, only: scalar, get_threebody_fourier, get_twobody_weights

    implicit none

    double precision, allocatable, dimension(:,:,:,:) :: fourier

    ! fchl descriptors for the training set, format (i,maxatoms,5,maxneighbors)
    double precision, dimension(:,:,:,:), intent(in) :: x1
    double precision, dimension(:,:,:,:), intent(in) :: x2

    ! List of numbers of atoms in each molecule
    integer, dimension(:), intent(in) :: n1
    integer, dimension(:), intent(in) :: n2

    ! Number of neighbors for each atom in each compound
    integer, dimension(:,:), intent(in) :: nneigh1
    integer, dimension(:,:), intent(in) :: nneigh2

    ! Sigma in the Gaussian kernel
    double precision, dimension(:), intent(in) :: sigmas

    ! Number of molecules
    integer, intent(in) :: nm1
    integer, intent(in) :: nm2

    ! Number of sigmas
    integer, intent(in) :: nsigmas

    double precision, intent(in) :: two_body_power
    double precision, intent(in) :: three_body_power

    double precision, intent(in) :: t_width
    double precision, intent(in) :: d_width
    double precision, intent(in) :: cut_start
    double precision, intent(in) :: cut_distance
    integer, intent(in) :: order
    double precision, intent(in) :: distance_scale
    double precision, intent(in) :: angular_scale

    ! -1.0 / sigma^2 for use in the kernel
    double precision, dimension(nsigmas) :: inv_sigma2

    double precision, dimension(:,:), intent(in) :: pd

    ! Resulting alpha vector
    double precision, dimension(nsigmas,nm1,nm2), intent(out) :: kernels

    ! Internal counters
    integer :: i, j, k! , l
    integer :: ni, nj
    integer :: a, b, n

    ! Temporary variables necessary for parallelization
    double precision :: l2dist
    double precision, allocatable, dimension(:,:) :: atomic_distance

    ! Pre-computed terms in the full distance matrix
    double precision, allocatable, dimension(:,:) :: self_scalar1
    double precision, allocatable, dimension(:,:) :: self_scalar2

    ! Pre-computed terms
    double precision, allocatable, dimension(:,:,:) :: ksi1
    double precision, allocatable, dimension(:,:,:) :: ksi2

    double precision, allocatable, dimension(:,:,:,:,:) :: sinp1
    double precision, allocatable, dimension(:,:,:,:,:) :: sinp2
    double precision, allocatable, dimension(:,:,:,:,:) :: cosp1
    double precision, allocatable, dimension(:,:,:,:,:) :: cosp2

    logical, intent(in) :: alchemy

    ! Value of PI at full FORTRAN precision.
    double precision, parameter :: pi = 4.0d0 * atan(1.0d0)

    ! counter for periodic distance
    integer :: pmax1
    integer :: pmax2
    ! integer :: nneighi

    double precision :: ang_norm2

    integer :: maxneigh1
    integer :: maxneigh2

    maxneigh1 = maxval(nneigh1)
    maxneigh2 = maxval(nneigh2)

    ang_norm2 = 0.0d0

    do n = -10000, 10000
        ang_norm2 = ang_norm2 + exp(-((t_width * n)**2)) &
            & * (2.0d0 - 2.0d0 * cos(n * pi))
    end do

    ang_norm2 = sqrt(ang_norm2 * pi) * 2.0d0

    pmax1 = 0
    pmax2 = 0

    do a = 1, nm1
        pmax1 = max(pmax1, int(maxval(x1(a,1,2,:n1(a)))))
    enddo
    do a = 1, nm2
        pmax2 = max(pmax2, int(maxval(x2(a,1,2,:n2(a)))))
    enddo

    inv_sigma2(:) = -0.5d0 / (sigmas(:))**2

    allocate(ksi1(nm1, maxval(n1), maxneigh1))
    allocate(ksi2(nm2, maxval(n2), maxneigh2))

    ksi1 = 0.0d0
    ksi2 = 0.0d0


    !$OMP PARALLEL DO PRIVATE(ni)
    do a = 1, nm1
        ni = n1(a)
        do i = 1, ni
            ksi1(a, i, :) = get_twobody_weights(x1(a,i,:,:), nneigh1(a, i), &
                & two_body_power, cut_start, cut_distance, maxneigh1)
        enddo
    enddo
    !$OMP END PARALLEL do

    !$OMP PARALLEL DO PRIVATE(ni)
    do a = 1, nm2
        ni = n2(a)
        do i = 1, ni
            ksi2(a, i, :) = get_twobody_weights(x2(a,i,:,:), nneigh2(a, i), &
               & two_body_power, cut_start, cut_distance, maxneigh2)
        enddo
    enddo
    !$OMP END PARALLEL do


    allocate(cosp1(nm1, maxval(n1), pmax1, order, maxval(nneigh1)))
    allocate(sinp1(nm1, maxval(n1), pmax1, order, maxval(nneigh1)))

    cosp1 = 0.0d0
    sinp1 = 0.0d0

    !$OMP PARALLEL DO PRIVATE(ni, fourier) schedule(dynamic)
    do a = 1, nm1
        ni = n1(a)
        do i = 1, ni

            fourier = get_threebody_fourier(x1(a,i,:,:), &
                & nneigh1(a, i), order, three_body_power, cut_start, cut_distance, pmax1, order, maxneigh1)

            cosp1(a,i,:,:,:) = fourier(1,:,:,:)
            sinp1(a,i,:,:,:) = fourier(2,:,:,:)

        enddo
    enddo
    !$OMP END PARALLEL DO

    allocate(cosp2(nm2, maxval(n2), pmax2, order, maxval(nneigh2)))
    allocate(sinp2(nm2, maxval(n2), pmax2, order, maxval(nneigh2)))

    cosp2 = 0.0d0
    sinp2 = 0.0d0

    !$OMP PARALLEL DO PRIVATE(ni, fourier) schedule(dynamic)
    do a = 1, nm2
        ni = n2(a)
        do i = 1, ni

            fourier = get_threebody_fourier(x2(a,i,:,:), &
                & nneigh2(a, i), order, three_body_power, cut_start, cut_distance, pmax2, order, maxneigh2)

            cosp2(a,i,:,:,:) = fourier(1,:,:,:)
            sinp2(a,i,:,:,:) = fourier(2,:,:,:)

        enddo
    enddo
    !$OMP END PARALLEL DO

    allocate(self_scalar1(nm1, maxval(n1)))
    allocate(self_scalar2(nm2, maxval(n2)))

    !$OMP PARALLEL DO PRIVATE(ni)
    do a = 1, nm1
        ni = n1(a)
        do i = 1, ni
            self_scalar1(a,i) = scalar(x1(a,i,:,:), x1(a,i,:,:), &
                & nneigh1(a,i), nneigh1(a,i), ksi1(a,i,:), ksi1(a,i,:), &
                & sinp1(a,i,:,:,:), sinp1(a,i,:,:,:), &
                & cosp1(a,i,:,:,:), cosp1(a,i,:,:,:), &
                & t_width, d_width, cut_distance, order, &
                & pd, ang_norm2,distance_scale, angular_scale, alchemy)
        enddo
    enddo
    !$OMP END PARALLEL DO

    !$OMP PARALLEL DO PRIVATE(ni)
    do a = 1, nm2
        ni = n2(a)
        do i = 1, ni
            self_scalar2(a,i) = scalar(x2(a,i,:,:), x2(a,i,:,:), &
                & nneigh2(a,i), nneigh2(a,i), ksi2(a,i,:), ksi2(a,i,:), &
                & sinp2(a,i,:,:,:), sinp2(a,i,:,:,:), &
                & cosp2(a,i,:,:,:), cosp2(a,i,:,:,:), &
                & t_width, d_width, cut_distance, order, &
                & pd, ang_norm2, distance_scale, angular_scale, alchemy)
        enddo
    enddo
    !$OMP END PARALLEL DO


    allocate(atomic_distance(maxval(n1), maxval(n2)))

    kernels(:,:,:) = 0.0d0
    atomic_distance(:,:) = 0.0d0

    !$OMP PARALLEL DO schedule(dynamic) PRIVATE(l2dist,atomic_distance,ni,nj)
    do b = 1, nm2
        nj = n2(b)
        do a = 1, nm1
            ni = n1(a)

            atomic_distance(:,:) = 0.0d0

            do i = 1, ni
                do j = 1, nj

                    l2dist = scalar(x1(a,i,:,:), x2(b,j,:,:), &
                        & nneigh1(a,i), nneigh2(b,j), ksi1(a,i,:), ksi2(b,j,:), &
                        & sinp1(a,i,:,:,:), sinp2(b,j,:,:,:), &
                        & cosp1(a,i,:,:,:), cosp2(b,j,:,:,:), &
                        & t_width, d_width, cut_distance, order, &
                        & pd, ang_norm2, distance_scale, angular_scale, alchemy)

                    l2dist = self_scalar1(a,i) + self_scalar2(b,j) - 2.0d0 * l2dist
                    atomic_distance(i,j) = l2dist

                enddo
            enddo

            do k = 1, nsigmas
                kernels(k, a, b) = sum(exp(atomic_distance(:ni,:nj) &
                    & * inv_sigma2(k)))
            enddo

        enddo
    enddo
    !$OMP END PARALLEL DO

    deallocate(atomic_distance)
    deallocate(self_scalar1)
    deallocate(self_scalar2)
    deallocate(ksi1)
    deallocate(ksi2)
    deallocate(cosp1)
    deallocate(cosp2)
    deallocate(sinp1)
    deallocate(sinp2)

end subroutine fget_kernels_fchl


subroutine fget_symmetric_kernels_fchl(x1, n1, nneigh1, sigmas, nm1, nsigmas, &
       & t_width, d_width, cut_start, cut_distance, order, pd, &
       & distance_scale, angular_scale, alchemy, two_body_power, three_body_power, kernels)

    use ffchl_module, only: scalar, get_threebody_fourier, get_twobody_weights

    implicit none

    double precision, allocatable, dimension(:,:,:,:) :: fourier

    ! FCHL descriptors for the training set, format (i,j_1,5,m_1)
    double precision, dimension(:,:,:,:), intent(in) :: x1

    ! List of numbers of atoms in each molecule
    integer, dimension(:), intent(in) :: n1

    ! Number of neighbors for each atom in each compound
    integer, dimension(:,:), intent(in) :: nneigh1

    ! Sigma in the Gaussian kernel
    double precision, dimension(:), intent(in) :: sigmas

    ! Number of molecules
    integer, intent(in) :: nm1

    ! Number of sigmas
    integer, intent(in) :: nsigmas

    double precision, intent(in) :: two_body_power
    double precision, intent(in) :: three_body_power

    double precision, intent(in) :: t_width
    double precision, intent(in) :: d_width
    double precision, intent(in) :: cut_start
    double precision, intent(in) :: cut_distance
    integer, intent(in) :: order
    double precision, intent(in) :: distance_scale
    double precision, intent(in) :: angular_scale

    logical, intent(in) :: alchemy
    ! -1.0 / sigma^2 for use in the kernel
    double precision, dimension(nsigmas) :: inv_sigma2

    double precision, dimension(:,:), intent(in) :: pd

    ! Resulting alpha vector
    double precision, dimension(nsigmas,nm1,nm1), intent(out) :: kernels

    ! Internal counters
    integer :: i, j, k, ni, nj
    integer :: a, b, n

    ! Temporary variables necessary for parallelization
    double precision :: l2dist
    double precision, allocatable, dimension(:,:) :: atomic_distance

    ! Pre-computed terms in the full distance matrix
    double precision, allocatable, dimension(:,:) :: self_scalar1

    ! Pre-computed terms
    double precision, allocatable, dimension(:,:,:) :: ksi1

    double precision, allocatable, dimension(:,:,:,:,:) :: sinp1
    double precision, allocatable, dimension(:,:,:,:,:) :: cosp1

    ! Value of PI at full FORTRAN precision.
    double precision, parameter :: pi = 4.0d0 * atan(1.0d0)

    ! counter for periodic distance
    integer :: pmax1
    ! integer :: nneighi

    double precision :: ang_norm2

    integer :: maxneigh1

    maxneigh1 = maxval(nneigh1)

    ang_norm2 = 0.0d0

    do n = -10000, 10000
        ang_norm2 = ang_norm2 + exp(-((t_width * n)**2)) &
            & * (2.0d0 - 2.0d0 * cos(n * pi))
    end do

    ang_norm2 = sqrt(ang_norm2 * pi) * 2.0d0

    pmax1 = 0

    do a = 1, nm1
        pmax1 = max(pmax1, int(maxval(x1(a,1,2,:n1(a)))))
    enddo

    inv_sigma2(:) = -0.5d0 / (sigmas(:))**2

    allocate(ksi1(nm1, maxval(n1), maxval(nneigh1)))

    ksi1 = 0.0d0

    !$OMP PARALLEL DO PRIVATE(ni)
    do a = 1, nm1
        ni = n1(a)
        do i = 1, ni
            ksi1(a, i, :) = get_twobody_weights(x1(a,i,:,:), nneigh1(a, i), &
               & two_body_power, cut_start, cut_distance, maxneigh1)
        enddo
    enddo
    !$OMP END PARALLEL do

    allocate(cosp1(nm1, maxval(n1), pmax1, order, maxval(nneigh1)))
    allocate(sinp1(nm1, maxval(n1), pmax1, order, maxval(nneigh1)))

    cosp1 = 0.0d0
    sinp1 = 0.0d0

    !$OMP PARALLEL DO PRIVATE(ni, fourier)
    do a = 1, nm1
        ni = n1(a)
        do i = 1, ni

            fourier = get_threebody_fourier(x1(a,i,:,:), &
                & nneigh1(a, i), order, three_body_power, cut_start, cut_distance, pmax1, order, maxval(nneigh1))

            cosp1(a,i,:,:,:) = fourier(1,:,:,:)
            sinp1(a,i,:,:,:) = fourier(2,:,:,:)

        enddo
    enddo
    !$OMP END PARALLEL DO

    allocate(self_scalar1(nm1, maxval(n1)))

    !$OMP PARALLEL DO PRIVATE(ni)
    do a = 1, nm1
        ni = n1(a)
        do i = 1, ni
            self_scalar1(a,i) = scalar(x1(a,i,:,:), x1(a,i,:,:), &
                & nneigh1(a,i), nneigh1(a,i), ksi1(a,i,:), ksi1(a,i,:), &
                & sinp1(a,i,:,:,:), sinp1(a,i,:,:,:), &
                & cosp1(a,i,:,:,:), cosp1(a,i,:,:,:), &
                & t_width, d_width, cut_distance, order, &
                & pd, ang_norm2,distance_scale, angular_scale, alchemy)
        enddo
    enddo
    !$OMP END PARALLEL DO

    allocate(atomic_distance(maxval(n1), maxval(n1)))

    kernels(:,:,:) = 0.0d0
    atomic_distance(:,:) = 0.0d0

    !$OMP PARALLEL DO schedule(dynamic) PRIVATE(l2dist,atomic_distance,ni,nj)
    do b = 1, nm1
        nj = n1(b)
        do a = b, nm1
            ni = n1(a)

            atomic_distance(:,:) = 0.0d0

            do i = 1, ni
                do j = 1, nj

                    l2dist = scalar(x1(a,i,:,:), x1(b,j,:,:), &
                        & nneigh1(a,i), nneigh1(b,j), ksi1(a,i,:), ksi1(b,j,:), &
                        & sinp1(a,i,:,:,:), sinp1(b,j,:,:,:), &
                        & cosp1(a,i,:,:,:), cosp1(b,j,:,:,:), &
                        & t_width, d_width, cut_distance, order, &
                        & pd, ang_norm2, distance_scale, angular_scale, alchemy)

                    l2dist = self_scalar1(a,i) + self_scalar1(b,j) - 2.0d0 * l2dist
                    atomic_distance(i,j) = l2dist

                enddo
            enddo

            do k = 1, nsigmas
                kernels(k, a, b) =  sum(exp(atomic_distance(:ni,:nj) &
                    & * inv_sigma2(k)))
                kernels(k, b, a) = kernels(k, a, b)
            enddo

        enddo
    enddo
    !$OMP END PARALLEL DO

    deallocate(atomic_distance)
    deallocate(self_scalar1)
    deallocate(ksi1)
    deallocate(cosp1)
    deallocate(sinp1)

end subroutine fget_symmetric_kernels_fchl


subroutine fget_global_symmetric_kernels_fchl(x1, n1, nneigh1, sigmas, nm1, nsigmas, &
       & t_width, d_width, cut_start, cut_distance, order, pd, &
       & distance_scale, angular_scale, alchemy, two_body_power, three_body_power, kernels)

    use ffchl_module, only: scalar, get_threebody_fourier, get_twobody_weights

    implicit none

    double precision, allocatable, dimension(:,:,:,:) :: fourier

    ! FCHL descriptors for the training set, format (i,j_1,5,m_1)
    double precision, dimension(:,:,:,:), intent(in) :: x1

    ! List of numbers of atoms in each molecule
    integer, dimension(:), intent(in) :: n1

    ! Number of neighbors for each atom in each compound
    integer, dimension(:,:), intent(in) :: nneigh1

    ! Sigma in the Gaussian kernel
    double precision, dimension(:), intent(in) :: sigmas

    ! Number of molecules
    integer, intent(in) :: nm1

    ! Number of sigmas
    integer, intent(in) :: nsigmas

    double precision, intent(in) :: two_body_power
    double precision, intent(in) :: three_body_power

    double precision, intent(in) :: t_width
    double precision, intent(in) :: d_width
    double precision, intent(in) :: cut_start
    double precision, intent(in) :: cut_distance
    integer, intent(in) :: order
    double precision, intent(in) :: distance_scale
    double precision, intent(in) :: angular_scale
    logical, intent(in) :: alchemy

    ! -1.0 / sigma^2 for use in the kernel
    double precision, dimension(nsigmas) :: inv_sigma2

    double precision, dimension(:,:), intent(in) :: pd

    ! Resulting alpha vector
    double precision, dimension(nsigmas,nm1,nm1), intent(out) :: kernels

    ! Internal counters
    integer :: i, j, k, ni, nj
    integer :: a, b, n

    ! Temporary variables necessary for parallelization
    double precision :: l2dist
    double precision, allocatable, dimension(:,:) :: atomic_distance

    ! Pre-computed terms in the full distance matrix
    double precision, allocatable, dimension(:) :: self_scalar1

    ! Pre-computed terms
    double precision, allocatable, dimension(:,:,:) :: ksi1

    double precision, allocatable, dimension(:,:,:,:,:) :: sinp1
    double precision, allocatable, dimension(:,:,:,:,:) :: cosp1

    ! Value of PI at full FORTRAN precision.
    double precision, parameter :: pi = 4.0d0 * atan(1.0d0)

    ! counter for periodic distance
    integer :: pmax1
    ! integer :: nneighi

    double precision :: ang_norm2

    double precision :: mol_dist

    integer :: maxneigh1

    maxneigh1 = maxval(nneigh1)

    ang_norm2 = 0.0d0

    do n = -10000, 10000
        ang_norm2 = ang_norm2 + exp(-((t_width * n)**2)) &
            & * (2.0d0 - 2.0d0 * cos(n * pi))
    end do

    ang_norm2 = sqrt(ang_norm2 * pi) * 2.0d0

    pmax1 = 0

    do a = 1, nm1
        pmax1 = max(pmax1, int(maxval(x1(a,1,2,:n1(a)))))
    enddo

    inv_sigma2(:) = -0.5d0 / (sigmas(:))**2

    allocate(ksi1(nm1, maxval(n1), maxval(nneigh1)))

    ksi1 = 0.0d0

    !$OMP PARALLEL DO PRIVATE(ni)
    do a = 1, nm1
        ni = n1(a)
        do i = 1, ni
            ksi1(a, i, :) = get_twobody_weights(x1(a,i,:,:), nneigh1(a, i), &
               & two_body_power, cut_start, cut_distance, maxneigh1)
        enddo
    enddo
    !$OMP END PARALLEL do

    allocate(cosp1(nm1, maxval(n1), pmax1, order, maxneigh1))
    allocate(sinp1(nm1, maxval(n1), pmax1, order, maxneigh1))

    cosp1 = 0.0d0
    sinp1 = 0.0d0

    !$OMP PARALLEL DO PRIVATE(ni, fourier)
    do a = 1, nm1
        ni = n1(a)
        do i = 1, ni

            fourier = get_threebody_fourier(x1(a,i,:,:), &
                & nneigh1(a, i), order, three_body_power, cut_start, cut_distance, pmax1, order, maxneigh1)

            cosp1(a,i,:,:,:) = fourier(1,:,:,:)
            sinp1(a,i,:,:,:) = fourier(2,:,:,:)

        enddo
    enddo
    !$OMP END PARALLEL DO

    allocate(self_scalar1(nm1))

    self_scalar1 = 0.0d0

    !$OMP PARALLEL DO PRIVATE(ni) REDUCTION(+:self_scalar1)
    do a = 1, nm1
        ni = n1(a)
        do i = 1, ni
            do j = 1, ni

            self_scalar1(a) = self_scalar1(a) + scalar(x1(a,i,:,:), x1(a,j,:,:), &
                & nneigh1(a,i), nneigh1(a,j), ksi1(a,i,:), ksi1(a,j,:), &
                & sinp1(a,i,:,:,:), sinp1(a,j,:,:,:), &
                & cosp1(a,i,:,:,:), cosp1(a,j,:,:,:), &
                & t_width, d_width, cut_distance, order, &
                & pd, ang_norm2,distance_scale, angular_scale, alchemy)
            enddo
        enddo
    enddo
    !$OMP END PARALLEL DO

    allocate(atomic_distance(maxval(n1), maxval(n1)))

    kernels(:,:,:) = 0.0d0
    atomic_distance(:,:) = 0.0d0

    !$OMP PARALLEL DO schedule(dynamic) PRIVATE(l2dist,atomic_distance,ni,nj)
    do b = 1, nm1
        nj = n1(b)
        do a = b, nm1
            ni = n1(a)

            atomic_distance(:,:) = 0.0d0

            do i = 1, ni
                do j = 1, nj

                    l2dist = scalar(x1(a,i,:,:), x1(b,j,:,:), &
                        & nneigh1(a,i), nneigh1(b,j), ksi1(a,i,:), ksi1(b,j,:), &
                        & sinp1(a,i,:,:,:), sinp1(b,j,:,:,:), &
                        & cosp1(a,i,:,:,:), cosp1(b,j,:,:,:), &
                        & t_width, d_width, cut_distance, order, &
                        & pd, ang_norm2, distance_scale, angular_scale, alchemy)

                    atomic_distance(i,j) = l2dist

                enddo
            enddo

            mol_dist = self_scalar1(a) + self_scalar1(b) - 2.0d0 * sum(atomic_distance(:ni,:nj))

            do k = 1, nsigmas
                kernels(k, a, b) = exp(mol_dist * inv_sigma2(k))
                kernels(k, b, a) = kernels(k, a, b)
            enddo
        enddo
    enddo
    !$OMP END PARALLEL DO

    deallocate(atomic_distance)
    deallocate(self_scalar1)
    deallocate(ksi1)
    deallocate(cosp1)
    deallocate(sinp1)

end subroutine fget_global_symmetric_kernels_fchl


subroutine fget_global_kernels_fchl(x1, x2, n1, n2, nneigh1, nneigh2, &
       & sigmas, nm1, nm2, nsigmas, &
       & t_width, d_width, cut_start, cut_distance, order, pd, &
       & distance_scale, angular_scale, alchemy, two_body_power, three_body_power, kernels)

    use ffchl_module, only: scalar, get_threebody_fourier, get_twobody_weights

    implicit none

    double precision, allocatable, dimension(:,:,:,:) :: fourier

    ! fchl descriptors for the training set, format (i,maxatoms,5,maxneighbors)
    double precision, dimension(:,:,:,:), intent(in) :: x1
    double precision, dimension(:,:,:,:), intent(in) :: x2

    ! List of numbers of atoms in each molecule
    integer, dimension(:), intent(in) :: n1
    integer, dimension(:), intent(in) :: n2

    ! Number of neighbors for each atom in each compound
    integer, dimension(:,:), intent(in) :: nneigh1
    integer, dimension(:,:), intent(in) :: nneigh2

    ! Sigma in the Gaussian kernel
    double precision, dimension(:), intent(in) :: sigmas

    ! Number of molecules
    integer, intent(in) :: nm1
    integer, intent(in) :: nm2

    ! Number of sigmas
    integer, intent(in) :: nsigmas

    double precision, intent(in) :: two_body_power
    double precision, intent(in) :: three_body_power

    double precision, intent(in) :: t_width
    double precision, intent(in) :: d_width
    double precision, intent(in) :: cut_start
    double precision, intent(in) :: cut_distance
    integer, intent(in) :: order
    double precision, intent(in) :: distance_scale
    double precision, intent(in) :: angular_scale
    logical, intent(in) :: alchemy

    ! -1.0 / sigma^2 for use in the kernel
    double precision, dimension(nsigmas) :: inv_sigma2

    double precision, dimension(:,:), intent(in) :: pd

    ! Resulting alpha vector
    double precision, dimension(nsigmas,nm1,nm2), intent(out) :: kernels

    ! Internal counters
    integer :: i, j, k
    integer :: ni, nj
    integer :: a, b, n

    ! Temporary variables necessary for parallelization
    double precision :: l2dist
    double precision, allocatable, dimension(:,:) :: atomic_distance

    ! Pre-computed terms in the full distance matrix
    double precision, allocatable, dimension(:) :: self_scalar1
    double precision, allocatable, dimension(:) :: self_scalar2

    ! Pre-computed terms
    double precision, allocatable, dimension(:,:,:) :: ksi1
    double precision, allocatable, dimension(:,:,:) :: ksi2

    double precision, allocatable, dimension(:,:,:,:,:) :: sinp1
    double precision, allocatable, dimension(:,:,:,:,:) :: sinp2
    double precision, allocatable, dimension(:,:,:,:,:) :: cosp1
    double precision, allocatable, dimension(:,:,:,:,:) :: cosp2

    ! Value of PI at full FORTRAN precision.
    double precision, parameter :: pi = 4.0d0 * atan(1.0d0)

    ! counter for periodic distance
    integer :: pmax1
    integer :: pmax2
    ! integer :: nneighi
    double precision :: ang_norm2

    double precision :: mol_dist

    integer :: maxneigh1
    integer :: maxneigh2

    maxneigh1 = maxval(nneigh1)
    maxneigh2 = maxval(nneigh2)

    ang_norm2 = 0.0d0

    do n = -10000, 10000
        ang_norm2 = ang_norm2 + exp(-((t_width * n)**2)) &
            & * (2.0d0 - 2.0d0 * cos(n * pi))
    end do

    ang_norm2 = sqrt(ang_norm2 * pi) * 2.0d0

    pmax1 = 0
    pmax2 = 0

    do a = 1, nm1
        pmax1 = max(pmax1, int(maxval(x1(a,1,2,:n1(a)))))
    enddo
    do a = 1, nm2
        pmax2 = max(pmax2, int(maxval(x2(a,1,2,:n2(a)))))
    enddo

    inv_sigma2(:) = -0.5d0 / (sigmas(:))**2

    allocate(ksi1(nm1, maxval(n1), maxval(nneigh1)))
    allocate(ksi2(nm2, maxval(n2), maxval(nneigh2)))

    ksi1 = 0.0d0
    ksi2 = 0.0d0

    !$OMP PARALLEL DO PRIVATE(ni)
    do a = 1, nm1
        ni = n1(a)
        do i = 1, ni
            ksi1(a, i, :) = get_twobody_weights(x1(a,i,:,:), nneigh1(a, i), &
               & two_body_power, cut_start, cut_distance, maxneigh1)
        enddo
    enddo
    !$OMP END PARALLEL do

    !$OMP PARALLEL DO PRIVATE(ni)
    do a = 1, nm2
        ni = n2(a)
        do i = 1, ni
            ksi2(a, i, :) = get_twobody_weights(x2(a,i,:,:), nneigh2(a, i), &
               & two_body_power, cut_start, cut_distance, maxneigh2)
        enddo
    enddo
    !$OMP END PARALLEL do

    allocate(cosp1(nm1, maxval(n1), pmax1, order, maxval(nneigh1)))
    allocate(sinp1(nm1, maxval(n1), pmax1, order, maxval(nneigh1)))

    cosp1 = 0.0d0
    sinp1 = 0.0d0

    !$OMP PARALLEL DO PRIVATE(ni, fourier)
    do a = 1, nm1
        ni = n1(a)
        do i = 1, ni

            fourier = get_threebody_fourier(x1(a,i,:,:), &
                & nneigh1(a, i), order, three_body_power, cut_start, cut_distance, pmax1, order, maxneigh1)

            cosp1(a,i,:,:,:) = fourier(1,:,:,:)
            sinp1(a,i,:,:,:) = fourier(2,:,:,:)

        enddo
    enddo
    !$OMP END PARALLEL DO

    allocate(cosp2(nm2, maxval(n2), pmax2, order, maxval(nneigh2)))
    allocate(sinp2(nm2, maxval(n2), pmax2, order, maxval(nneigh2)))

    cosp2 = 0.0d0
    sinp2 = 0.0d0

    !$OMP PARALLEL DO PRIVATE(ni, fourier)
    do a = 1, nm2
        ni = n2(a)
        do i = 1, ni

            fourier = get_threebody_fourier(x2(a,i,:,:), &
                & nneigh2(a, i), order, three_body_power, cut_start, cut_distance, pmax2, order, maxval(nneigh2))

            cosp2(a,i,:,:,:) = fourier(1,:,:,:)
            sinp2(a,i,:,:,:) = fourier(2,:,:,:)

        enddo
    enddo
    !$OMP END PARALLEL DO

    allocate(self_scalar1(nm1))
    allocate(self_scalar2(nm2))

    self_scalar1 = 0.0d0
    self_scalar2 = 0.0d0

    !$OMP PARALLEL DO PRIVATE(ni) REDUCTION(+:self_scalar1)
    do a = 1, nm1
        ni = n1(a)
        do i = 1, ni
            do j= 1, ni

            self_scalar1(a) = self_scalar1(a) + scalar(x1(a,i,:,:), x1(a,j,:,:), &
                & nneigh1(a,i), nneigh1(a,j), ksi1(a,i,:), ksi1(a,j,:), &
                & sinp1(a,i,:,:,:), sinp1(a,j,:,:,:), &
                & cosp1(a,i,:,:,:), cosp1(a,j,:,:,:), &
                & t_width, d_width, cut_distance, order, &
                & pd, ang_norm2,distance_scale, angular_scale, alchemy)
            enddo
        enddo
    enddo
    !$OMP END PARALLEL DO

    !$OMP PARALLEL DO PRIVATE(ni) REDUCTION(+:self_scalar2)
    do a = 1, nm2
        ni = n2(a)
        do i = 1, ni
            do j= 1, ni
            self_scalar2(a) = self_scalar2(a) + scalar(x2(a,i,:,:), x2(a,j,:,:), &
                & nneigh2(a,i), nneigh2(a,j), ksi2(a,i,:), ksi2(a,j,:), &
                & sinp2(a,i,:,:,:), sinp2(a,j,:,:,:), &
                & cosp2(a,i,:,:,:), cosp2(a,j,:,:,:), &
                & t_width, d_width, cut_distance, order, &
                & pd, ang_norm2, distance_scale, angular_scale, alchemy)
            enddo
        enddo
    enddo
    !$OMP END PARALLEL DO


    allocate(atomic_distance(maxval(n1), maxval(n2)))

    kernels(:,:,:) = 0.0d0
    atomic_distance(:,:) = 0.0d0

    !$OMP PARALLEL DO schedule(dynamic) PRIVATE(l2dist,atomic_distance,ni,nj)
    do b = 1, nm2
        nj = n2(b)
        do a = 1, nm1
            ni = n1(a)

            atomic_distance(:,:) = 0.0d0

            do i = 1, ni
                do j = 1, nj

                    l2dist = scalar(x1(a,i,:,:), x2(b,j,:,:), &
                        & nneigh1(a,i), nneigh2(b,j), ksi1(a,i,:), ksi2(b,j,:), &
                        & sinp1(a,i,:,:,:), sinp2(b,j,:,:,:), &
                        & cosp1(a,i,:,:,:), cosp2(b,j,:,:,:), &
                        & t_width, d_width, cut_distance, order, &
                        & pd, ang_norm2, distance_scale, angular_scale, alchemy)

                    atomic_distance(i,j) = l2dist

                enddo
            enddo

            mol_dist = self_scalar1(a) + self_scalar2(b) - 2.0d0 * sum(atomic_distance(:ni,:nj))

            do k = 1, nsigmas
                kernels(k, a, b) = exp(mol_dist * inv_sigma2(k))
            enddo

        enddo
    enddo
    !$OMP END PARALLEL DO

    deallocate(atomic_distance)
    deallocate(self_scalar1)
    deallocate(self_scalar2)
    deallocate(ksi1)
    deallocate(ksi2)
    deallocate(cosp1)
    deallocate(cosp2)
    deallocate(sinp1)
    deallocate(sinp2)

end subroutine fget_global_kernels_fchl


subroutine fget_atomic_kernels_fchl(x1, x2, nneigh1, nneigh2, &
       & sigmas, na1, na2, nsigmas, &
       & t_width, d_width, cut_start, cut_distance, order, pd, &
       & distance_scale, angular_scale, alchemy, two_body_power, three_body_power, kernels)

    use ffchl_module, only: scalar, get_threebody_fourier, get_twobody_weights

    implicit none

    double precision, allocatable, dimension(:,:,:,:) :: fourier

    ! fchl descriptors for the training set, format (i,maxatoms,5,maxneighbors)
    double precision, dimension(:,:,:), intent(in) :: x1
    double precision, dimension(:,:,:), intent(in) :: x2

    ! Number of neighbors for each atom in each compound
    integer, dimension(:), intent(in) :: nneigh1
    integer, dimension(:), intent(in) :: nneigh2

    ! Sigma in the Gaussian kernel
    double precision, dimension(:), intent(in) :: sigmas

    ! Number of molecules
    integer, intent(in) :: na1
    integer, intent(in) :: na2

    ! Number of sigmas
    integer, intent(in) :: nsigmas

    double precision, intent(in) :: two_body_power
    double precision, intent(in) :: three_body_power

    double precision, intent(in) :: t_width
    double precision, intent(in) :: d_width
    double precision, intent(in) :: cut_start
    double precision, intent(in) :: cut_distance
    integer, intent(in) :: order
    double precision, intent(in) :: distance_scale
    double precision, intent(in) :: angular_scale
    logical, intent(in) :: alchemy

    ! -1.0 / sigma^2 for use in the kernel
    double precision, dimension(nsigmas) :: inv_sigma2

    double precision, dimension(:,:), intent(in) :: pd

    ! Resulting alpha vector
    double precision, dimension(nsigmas,na1,na2), intent(out) :: kernels

    ! Internal counters
    integer :: i, j
    ! integer :: ni, nj
    integer :: a, n

    ! Temporary variables necessary for parallelization
    double precision :: l2dist

    ! Pre-computed terms in the full distance matrix
    double precision, allocatable, dimension(:) :: self_scalar1
    double precision, allocatable, dimension(:) :: self_scalar2

    ! Pre-computed terms
    double precision, allocatable, dimension(:,:) :: ksi1
    double precision, allocatable, dimension(:,:) :: ksi2

    double precision, allocatable, dimension(:,:,:,:) :: sinp1
    double precision, allocatable, dimension(:,:,:,:) :: sinp2
    double precision, allocatable, dimension(:,:,:,:) :: cosp1
    double precision, allocatable, dimension(:,:,:,:) :: cosp2

    ! Value of PI at full FORTRAN precision.
    double precision, parameter :: pi = 4.0d0 * atan(1.0d0)

    ! counter for periodic distance
    integer :: pmax1
    integer :: pmax2
    ! integer :: nneighi
    double precision :: ang_norm2

    ! double precision :: mol_dist

    integer :: maxneigh1
    integer :: maxneigh2

    maxneigh1 = maxval(nneigh1)
    maxneigh2 = maxval(nneigh2)

    ang_norm2 = 0.0d0

    do n = -10000, 10000
        ang_norm2 = ang_norm2 + exp(-((t_width * n)**2)) &
            & * (2.0d0 - 2.0d0 * cos(n * pi))
    end do

    ang_norm2 = sqrt(ang_norm2 * pi) * 2.0d0

    pmax1 = 0
    pmax2 = 0

    do a = 1, na1
        pmax1 = max(pmax1, int(maxval(x1(a,2,:nneigh1(a)))))
    enddo
    do a = 1, na2
        pmax2 = max(pmax2, int(maxval(x2(a,2,:nneigh2(a)))))
    enddo

    inv_sigma2(:) = -0.5d0 / (sigmas(:))**2

    allocate(ksi1(na1, maxval(nneigh1)))
    allocate(ksi2(na2, maxval(nneigh2)))

    ksi1 = 0.0d0
    ksi2 = 0.0d0

    !$OMP PARALLEL DO
    do i = 1, na1
        ksi1(i, :) = get_twobody_weights(x1(i,:,:), nneigh1(i), &
            & two_body_power, cut_start, cut_distance, maxneigh1)
    enddo
    !$OMP END PARALLEL do

    !$OMP PARALLEL DO
    do i = 1, na2
        ksi2(i, :) = get_twobody_weights(x2(i,:,:), nneigh2(i), &
            & two_body_power, cut_start, cut_distance, maxneigh2)
    enddo
    !$OMP END PARALLEL do

    allocate(cosp1(na1, pmax1, order, maxneigh1))
    allocate(sinp1(na1, pmax1, order, maxneigh1))

    cosp1 = 0.0d0
    sinp1 = 0.0d0

    !$OMP PARALLEL DO PRIVATE(fourier)
    do i = 1, na1

        fourier = get_threebody_fourier(x1(i,:,:), &
            & nneigh1(i), order, three_body_power, cut_start, cut_distance, pmax1, order, maxneigh1)

        cosp1(i,:,:,:) = fourier(1,:,:,:)
        sinp1(i,:,:,:) = fourier(2,:,:,:)

    enddo
    !$OMP END PARALLEL DO

    allocate(cosp2(na2, pmax2, order, maxneigh2))
    allocate(sinp2(na2, pmax2, order, maxneigh2))

    cosp2 = 0.0d0
    sinp2 = 0.0d0

    !$OMP PARALLEL DO PRIVATE(fourier)
    do i = 1, na2

        fourier = get_threebody_fourier(x2(i,:,:), &
            & nneigh2(i), order, three_body_power, cut_start, cut_distance, pmax2, order, maxneigh2)

        cosp2(i,:,:,:) = fourier(1,:,:,:)
        sinp2(i,:,:,:) = fourier(2,:,:,:)

    enddo
    !$OMP END PARALLEL DO

    allocate(self_scalar1(na1))
    allocate(self_scalar2(na2))

    self_scalar1 = 0.0d0
    self_scalar2 = 0.0d0

    !$OMP PARALLEL DO
    do i = 1, na1
        self_scalar1(i) = scalar(x1(i,:,:), x1(i,:,:), &
            & nneigh1(i), nneigh1(i), ksi1(i,:), ksi1(i,:), &
            & sinp1(i,:,:,:), sinp1(i,:,:,:), &
            & cosp1(i,:,:,:), cosp1(i,:,:,:), &
            & t_width, d_width, cut_distance, order, &
            & pd, ang_norm2,distance_scale, angular_scale, alchemy)
    enddo
    !$OMP END PARALLEL DO

    !$OMP PARALLEL DO
    do i = 1, na2
        self_scalar2(i) = scalar(x2(i,:,:), x2(i,:,:), &
            & nneigh2(i), nneigh2(i), ksi2(i,:), ksi2(i,:), &
            & sinp2(i,:,:,:), sinp2(i,:,:,:), &
            & cosp2(i,:,:,:), cosp2(i,:,:,:), &
            & t_width, d_width, cut_distance, order, &
            & pd, ang_norm2,distance_scale, angular_scale, alchemy)
    enddo
    !$OMP END PARALLEL DO

    kernels(:,:,:) = 0.0d0

    !$OMP PARALLEL DO schedule(dynamic) PRIVATE(l2dist)
    do i = 1, na1
        do j = 1, na2

            l2dist = self_scalar1(i) + self_scalar2(j) - 2.0d0 * scalar(x1(i,:,:), x2(j,:,:), &
                & nneigh1(i), nneigh2(j), ksi1(i,:), ksi2(j,:), &
                & sinp1(i,:,:,:), sinp2(j,:,:,:), &
                & cosp1(i,:,:,:), cosp2(j,:,:,:), &
                & t_width, d_width, cut_distance, order, &
                & pd, ang_norm2, distance_scale, angular_scale, alchemy)

            kernels(:, i, j) = exp(l2dist * inv_sigma2(:))

        enddo
    enddo
    !$OMP END PARALLEL DO

    deallocate(self_scalar1)
    deallocate(self_scalar2)
    deallocate(ksi1)
    deallocate(ksi2)
    deallocate(cosp1)
    deallocate(cosp2)
    deallocate(sinp1)
    deallocate(sinp2)

end subroutine fget_atomic_kernels_fchl


subroutine fget_atomic_symmetric_kernels_fchl(x1, nneigh1, &
       & sigmas, na1, nsigmas, &
       & t_width, d_width, cut_start, cut_distance, order, pd, &
       & distance_scale, angular_scale, alchemy, two_body_power, three_body_power, kernels)

    use ffchl_module, only: scalar, get_threebody_fourier, get_twobody_weights

    implicit none

    double precision, allocatable, dimension(:,:,:,:) :: fourier

    ! fchl descriptors for the training set, format (i,maxatoms,5,maxneighbors)
    double precision, dimension(:,:,:), intent(in) :: x1

    ! Number of neighbors for each atom in each compound
    integer, dimension(:), intent(in) :: nneigh1

    ! Sigma in the Gaussian kernel
    double precision, dimension(:), intent(in) :: sigmas

    ! Number of molecules
    integer, intent(in) :: na1

    ! Number of sigmas
    integer, intent(in) :: nsigmas

    double precision, intent(in) :: two_body_power
    double precision, intent(in) :: three_body_power

    double precision, intent(in) :: t_width
    double precision, intent(in) :: d_width
    double precision, intent(in) :: cut_start
    double precision, intent(in) :: cut_distance
    integer, intent(in) :: order
    double precision, intent(in) :: distance_scale
    double precision, intent(in) :: angular_scale
    logical, intent(in) :: alchemy

    ! -1.0 / sigma^2 for use in the kernel
    double precision, dimension(nsigmas) :: inv_sigma2

    double precision, dimension(:,:), intent(in) :: pd

    ! Resulting alpha vector
    double precision, dimension(nsigmas,na1,na1), intent(out) :: kernels

    ! Internal counters
    integer :: i, j
    ! integer :: ni, nj
    integer :: a, n

    ! Temporary variables necessary for parallelization
    double precision :: l2dist

    ! Pre-computed terms in the full distance matrix
    double precision, allocatable, dimension(:) :: self_scalar1

    ! Pre-computed terms
    double precision, allocatable, dimension(:,:) :: ksi1

    double precision, allocatable, dimension(:,:,:,:) :: sinp1
    double precision, allocatable, dimension(:,:,:,:) :: cosp1

    ! Value of PI at full FORTRAN precision.
    double precision, parameter :: pi = 4.0d0 * atan(1.0d0)

    ! counter for periodic distance
    integer :: pmax1
    ! integer :: nneighi
    double precision :: ang_norm2

    integer :: maxneigh1

    maxneigh1 = maxval(nneigh1)

    ang_norm2 = 0.0d0

    do n = -10000, 10000
        ang_norm2 = ang_norm2 + exp(-((t_width * n)**2)) &
            & * (2.0d0 - 2.0d0 * cos(n * pi))
    end do

    ang_norm2 = sqrt(ang_norm2 * pi) * 2.0d0

    pmax1 = 0

    do a = 1, na1
        pmax1 = max(pmax1, int(maxval(x1(a,2,:nneigh1(a)))))
    enddo

    inv_sigma2(:) = -0.5d0 / (sigmas(:))**2

    allocate(ksi1(na1, maxval(nneigh1)))

    ksi1 = 0.0d0

    !$OMP PARALLEL DO
    do i = 1, na1
        ksi1(i, :) = get_twobody_weights(x1(i,:,:), nneigh1(i), &
            & two_body_power, cut_start, cut_distance, maxneigh1)
    enddo
    !$OMP END PARALLEL do

    allocate(cosp1(na1, pmax1, order, maxneigh1))
    allocate(sinp1(na1, pmax1, order, maxneigh1))

    cosp1 = 0.0d0
    sinp1 = 0.0d0

    !$OMP PARALLEL DO PRIVATE(fourier)
    do i = 1, na1

        fourier = get_threebody_fourier(x1(i,:,:), &
            & nneigh1(i), order, three_body_power, cut_start, cut_distance, pmax1, order, maxneigh1)

        cosp1(i,:,:,:) = fourier(1,:,:,:)
        sinp1(i,:,:,:) = fourier(2,:,:,:)

    enddo
    !$OMP END PARALLEL DO

    allocate(self_scalar1(na1))

    self_scalar1 = 0.0d0

    !$OMP PARALLEL DO
    do i = 1, na1
        self_scalar1(i) = scalar(x1(i,:,:), x1(i,:,:), &
            & nneigh1(i), nneigh1(i), ksi1(i,:), ksi1(i,:), &
            & sinp1(i,:,:,:), sinp1(i,:,:,:), &
            & cosp1(i,:,:,:), cosp1(i,:,:,:), &
            & t_width, d_width, cut_distance, order, &
            & pd, ang_norm2,distance_scale, angular_scale, alchemy)
    enddo
    !$OMP END PARALLEL DO

    kernels(:,:,:) = 0.0d0

    !$OMP PARALLEL DO schedule(dynamic) PRIVATE(l2dist)
    do i = 1, na1
        do j = 1, na1

            l2dist = self_scalar1(i) + self_scalar1(j) - 2.0d0 * scalar(x1(i,:,:), x1(j,:,:), &
                & nneigh1(i), nneigh1(j), ksi1(i,:), ksi1(j,:), &
                & sinp1(i,:,:,:), sinp1(j,:,:,:), &
                & cosp1(i,:,:,:), cosp1(j,:,:,:), &
                & t_width, d_width, cut_distance, order, &
                & pd, ang_norm2, distance_scale, angular_scale, alchemy)

            kernels(:, i, j) = exp(l2dist * inv_sigma2(:))

        enddo
    enddo
    !$OMP END PARALLEL DO

    deallocate(self_scalar1)
    deallocate(ksi1)
    deallocate(cosp1)
    deallocate(sinp1)

end subroutine fget_atomic_symmetric_kernels_fchl
