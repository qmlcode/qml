! MIT License
!
! Copyright (c) 2016 Anders Steen Christensen, Felix Faber
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

module arad

    implicit none

contains

function atomic_distl2(X1, X2, N1, N2, sin1, sin2, width, cut_distance, r_width, c_width) result(aadist)

    implicit none

    double precision, dimension(:,:), intent(in) :: X1
    double precision, dimension(:,:), intent(in) :: X2

    integer, intent(in) :: N1
    integer, intent(in) :: N2

    double precision, dimension(:), intent(in) :: sin1
    double precision, dimension(:), intent(in) :: sin2

    double precision, intent(in) :: width
    double precision, intent(in) :: cut_distance
    double precision, intent(in) :: r_width
    double precision, intent(in) :: c_width

    double precision :: aadist

    double precision :: d

    integer :: m_1, m_2

    double precision :: maxgausdist2

    double precision :: inv_width
    double precision :: c_width2, r_width2, r2

    inv_width = -1.0d0 / (4.0d0 * width**2)

    maxgausdist2 = (8.0d0 * width)**2
    r_width2 = r_width**2
    c_width2 = c_width**2

    aadist = 0.0d0

    do m_1 = 1, N1

        if (X1(1, m_1) > cut_distance) exit

        do m_2 = 1, N2

            if (X2(1, m_2) > cut_distance) exit

            r2 = (X2(1,m_2) - X1(1,m_1))**2

            if (r2 < maxgausdist2) then

                d = exp(r2 * inv_width )  * sin1(m_1) * sin2(m_2)

                d = d * (r_width2/(r_width2 + (x1(2,m_1) - x2(2,m_2))**2) * &
                    & c_width2/(c_width2 + (x1(3,m_1) - x2(3,m_2))**2))

                aadist = aadist + d * (1.0d0 + x1(4,m_1)*x2(4,m_2) + x1(5,m_1)*x2(5,m_2))

            end if
        end do
    end do

end function atomic_distl2

end module arad


subroutine fget_local_kernels_arad(q1, q2, z1, z2, n1, n2, sigmas, nm1, nm2, nsigmas, &
        & width, cut_distance, r_width, c_width, kernels)

    use arad, only: atomic_distl2

    implicit none

    ! ARAD descriptors for the training set, format (i,j_1,5,m_1)
    double precision, dimension(:,:,:,:), intent(in) :: q1
    double precision, dimension(:,:,:,:), intent(in) :: q2

    ! ARAD atom-types for each atom in each molecule, format (i, j_1, 2)
    double precision, dimension(:,:,:), intent(in) :: z1
    double precision, dimension(:,:,:), intent(in) :: z2

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

    ! ARAD parameters
    double precision, intent(in) :: width
    double precision, intent(in) :: cut_distance
    double precision, intent(in) :: r_width
    double precision, intent(in) :: c_width

    ! Resulting alpha vector
    double precision, dimension(nsigmas,nm1,nm2), intent(out) :: kernels

    ! Internal counters
    integer :: i, j, k, ni, nj
    integer :: m_1, i_1, j_1

    ! Pre-computed constants
    double precision :: r_width2
    double precision :: c_width2
    double precision :: inv_cut

    ! Temporary variables necessary for parallelization
    double precision :: l2dist
    double precision, allocatable, dimension(:,:) :: atomic_distance

    ! Pre-computed terms in the full distance matrix
    double precision, allocatable, dimension(:,:) :: selfl21
    double precision, allocatable, dimension(:,:) :: selfl22

    ! Pre-computed sine terms
    double precision, allocatable, dimension(:,:,:) :: sin1
    double precision, allocatable, dimension(:,:,:) :: sin2

    ! Value of PI at full FORTRAN precision.
    double precision, parameter :: pi = 4.0d0 * atan(1.0d0)

    r_width2 = r_width**2
    c_width2 = c_width**2

    inv_cut = pi / (2.0d0 * cut_distance)
    inv_sigma2(:) = -1.0d0 / (sigmas(:))**2

    allocate(sin1(nm1, maxval(n1), maxval(n1)))
    allocate(sin2(nm2, maxval(n2), maxval(n2)))

    sin1 = 0.0d0
    sin2 = 0.0d0

    !$OMP PARALLEL DO PRIVATE(ni)
    do i = 1, nm1
        ni = n1(i)
        do m_1 = 1, ni
            do i_1 = 1, ni
                if (q1(i,i_1,1,m_1) < cut_distance) then
                    sin1(i, i_1, m_1) = 1.0d0 - sin(q1(i,i_1,1,m_1) * inv_cut)
                endif
            enddo
        enddo
    enddo
    !$OMP END PARALLEL DO

    !$OMP PARALLEL DO PRIVATE(ni)
    do i = 1, nm2
        ni = n2(i)
        do m_1 = 1, ni
            do i_1 = 1, ni
                if (q2(i,i_1,1,m_1) < cut_distance) then
                    sin2(i, i_1, m_1) = 1.0d0 - sin(q2(i,i_1,1,m_1) * inv_cut)
                endif
            enddo
        enddo
    enddo
    !$OMP END PARALLEL DO

    allocate(selfl21(nm1, maxval(n1)))
    allocate(selfl22(nm2, maxval(n2)))

    !$OMP PARALLEL DO PRIVATE(ni)
    do i = 1, nm1
        ni = n1(i)
        do i_1 = 1, ni
            selfl21(i,i_1) = atomic_distl2(q1(i,i_1,:,:), q1(i,i_1,:,:), n1(i), n1(i), &
                & sin1(i,i_1,:), sin1(i,i_1,:), width, cut_distance, r_width, c_width)
        enddo
    enddo
    !$OMP END PARALLEL DO

    !$OMP PARALLEL DO PRIVATE(ni)
    do i = 1, nm2
        ni = n2(i)
        do i_1 = 1, ni
            selfl22(i,i_1) = atomic_distl2(q2(i,i_1,:,:), q2(i,i_1,:,:), n2(i), n2(i), &
                & sin2(i,i_1,:), sin2(i,i_1,:), width, cut_distance, r_width, c_width)
        enddo
    enddo
    !$OMP END PARALLEL DO


    allocate(atomic_distance(maxval(n1), maxval(n2)))

    kernels(:,:,:) = 0.0d0
    atomic_distance(:,:) = 0.0d0

    !$OMP PARALLEL DO PRIVATE(l2dist,atomic_distance,ni,nj) schedule(dynamic)
    do j = 1, nm2
        nj = n2(j)
        do i = 1, nm1
            ni = n1(i)

            atomic_distance(:,:) = 0.0d0

            do i_1 = 1, ni
                do j_1 = 1, nj

                    l2dist = atomic_distl2(q1(i,i_1,:,:), q2(j,j_1,:,:), n1(i), n2(j), &
                        & sin1(i,i_1,:), sin2(j,j_1,:), width, cut_distance, r_width, c_width)

                    l2dist = selfl21(i,i_1) + selfl22(j,j_1) - 2.0d0 * l2dist &
                        & * (r_width2/(r_width2 + (z1(i,i_1,1) - z2(j,j_1,1))**2) * &
                        & c_width2/(c_width2 + (z1(i,i_1,2) - z2(j,j_1,2))**2))

                    atomic_distance(i_1,j_1) = l2dist

                enddo
            enddo

            do k = 1, nsigmas
                kernels(k, i, j) =  sum(exp(atomic_distance(:ni,:nj) * inv_sigma2(k)))
            enddo

        enddo
    enddo
    !$OMP END PARALLEL DO

    deallocate(atomic_distance)
    deallocate(selfl21)
    deallocate(selfl22)
    deallocate(sin1)
    deallocate(sin2)

end subroutine fget_local_kernels_arad


subroutine fget_local_symmetric_kernels_arad(q1, z1, n1, sigmas, nm1, nsigmas, &
        & width, cut_distance, r_width, c_width, kernels)

    use arad, only: atomic_distl2

    implicit none

    ! ARAD descriptors for the training set, format (i,j_1,5,m_1)
    double precision, dimension(:,:,:,:), intent(in) :: q1

    ! ARAD atom-types for each atom in each molecule, format (i, j_1, 2)
    double precision, dimension(:,:,:), intent(in) :: z1

    ! List of numbers of atoms in each molecule
    integer, dimension(:), intent(in) :: n1

    ! Sigma in the Gaussian kernel
    double precision, dimension(:), intent(in) :: sigmas

    ! Number of molecules
    integer, intent(in) :: nm1

    ! Number of sigmas
    integer, intent(in) :: nsigmas

    ! -1.0 / sigma^2 for use in the kernel
    double precision, dimension(nsigmas) :: inv_sigma2

    ! ARAD parameters
    double precision, intent(in) :: width
    double precision, intent(in) :: cut_distance
    double precision, intent(in) :: r_width
    double precision, intent(in) :: c_width

    ! Resulting alpha vector
    double precision, dimension(nsigmas,nm1,nm1), intent(out) :: kernels

    ! Internal counters
    integer :: i, j, k, ni, nj
    integer :: m_1, i_1, j_1

    ! Pre-computed constants
    double precision :: r_width2
    double precision :: c_width2
    double precision :: inv_cut

    ! Temporary variables necessary for parallelization
    double precision :: l2dist
    double precision, allocatable, dimension(:,:) :: atomic_distance

    ! Pre-computed terms in the full distance matrix
    double precision, allocatable, dimension(:,:) :: selfl21

    ! Pre-computed sine terms
    double precision, allocatable, dimension(:,:,:) :: sin1

    ! Value of PI at full FORTRAN precision.
    double precision, parameter :: pi = 4.0d0 * atan(1.0d0)

    r_width2 = r_width**2
    c_width2 = c_width**2

    inv_cut = pi / (2.0d0 * cut_distance)
    inv_sigma2(:) = -1.0d0 / (sigmas(:))**2

    allocate(sin1(nm1, maxval(n1), maxval(n1)))

    !$OMP PARALLEL DO PRIVATE(ni)
    do i = 1, nm1
        ni = n1(i)
        do m_1 = 1, ni
            do i_1 = 1, ni
                sin1(i, i_1, m_1) = 1.0d0 - sin(q1(i,i_1,1,m_1) * inv_cut)
            enddo
        enddo
    enddo
    !$OMP END PARALLEL DO

    allocate(selfl21(nm1, maxval(n1)))

    !$OMP PARALLEL DO PRIVATE(ni)
    do i = 1, nm1
        ni = n1(i)
        do i_1 = 1, ni
            selfl21(i,i_1) = atomic_distl2(q1(i,i_1,:,:), q1(i,i_1,:,:), n1(i), n1(i), &
                & sin1(i,i_1,:), sin1(i,i_1,:), width, cut_distance, r_width, c_width)
        enddo
    enddo
    !$OMP END PARALLEL DO

    allocate(atomic_distance(maxval(n1), maxval(n1)))

    kernels(:,:,:) = 0.0d0
    atomic_distance(:,:) = 0.0d0

    !$OMP PARALLEL DO PRIVATE(l2dist,atomic_distance,ni,nj) schedule(dynamic)
    do j = 1, nm1
        nj = n1(j)
        do i = 1, j
            ni = n1(i)

            atomic_distance(:,:) = 0.0d0

            do i_1 = 1, ni
                do j_1 = 1, nj

                    l2dist = atomic_distl2(q1(i,i_1,:,:), q1(j,j_1,:,:), n1(i), n1(j), &
                        & sin1(i,i_1,:), sin1(j,j_1,:), width, cut_distance, r_width, c_width)

                    l2dist = selfl21(i,i_1) + selfl21(j,j_1) - 2.0d0 * l2dist &
                        & * (r_width2/(r_width2 + (z1(i,i_1,1) - z1(j,j_1,1))**2) &
                        & * c_width2/(c_width2 + (z1(i,i_1,2) - z1(j,j_1,2))**2))

                    atomic_distance(i_1,j_1) = l2dist

                enddo
            enddo

            do k = 1, nsigmas
                kernels(k, i, j) =  sum(exp(atomic_distance(:ni,:nj) * inv_sigma2(k)))
                kernels(k, j, i) =  kernels(k, i, j)
            enddo

        enddo
    enddo
    !$OMP END PARALLEL DO

    deallocate(atomic_distance)
    deallocate(selfl21)
    deallocate(sin1)

end subroutine fget_local_symmetric_kernels_arad

subroutine fget_atomic_kernels_arad(q1, q2, z1, z2, n1, n2, sigmas, na1, na2, nsigmas, &
        & width, cut_distance, r_width, c_width, kernels)

    use arad, only: atomic_distl2

    implicit none

    ! ARAD descriptors for each atom in the training set, format (i,5,m_1)
    double precision, dimension(:,:,:), intent(in) :: q1
    double precision, dimension(:,:,:), intent(in) :: q2

    ! ARAD atom-types for each atom, format (i, 2)
    double precision, dimension(:,:), intent(in) :: z1
    double precision, dimension(:,:), intent(in) :: z2

    ! Sigma in the Gaussian kernel
    double precision, dimension(:), intent(in) :: sigmas

    ! List of numbers of atoms in each molecule
    integer, dimension(:), intent(in) :: n1
    integer, dimension(:), intent(in) :: n2

    ! Number of atom
    integer, intent(in) :: na1
    integer, intent(in) :: na2

    ! Number of sigmas
    integer, intent(in) :: nsigmas

    ! -1.0 / sigma^2 for use in the kernel
    double precision, dimension(nsigmas) :: inv_sigma2

    ! ARAD parameters
    double precision, intent(in) :: width
    double precision, intent(in) :: cut_distance
    double precision, intent(in) :: r_width
    double precision, intent(in) :: c_width

    ! Resulting alpha vector
    double precision, dimension(nsigmas,na1,na2), intent(out) :: kernels

    ! Internal counters
    integer :: i, j, k, ni
    integer :: m_1

    ! Pre-computed constants
    double precision :: r_width2
    double precision :: c_width2
    double precision :: inv_cut

    ! Temporary variables necessary for parallelization
    double precision :: l2dist

    ! Pre-computed terms in the full distance matrix
    double precision, allocatable, dimension(:) :: selfl21
    double precision, allocatable, dimension(:) :: selfl22

    ! Pre-computed sine terms
    double precision, allocatable, dimension(:,:) :: sin1
    double precision, allocatable, dimension(:,:) :: sin2

    ! Value of PI at full FORTRAN precision.
    double precision, parameter :: pi = 4.0d0 * atan(1.0d0)

    r_width2 = r_width**2
    c_width2 = c_width**2

    inv_cut = pi / (2.0d0 * cut_distance)
    inv_sigma2(:) = -1.0d0 / (sigmas(:))**2

    allocate(sin1(na1, maxval(n1)))
    allocate(sin2(na2, maxval(n2)))

    !$OMP PARALLEL DO PRIVATE(ni)
    do i = 1, na1
        ni = n1(i)
        do m_1 = 1, ni
            sin1(i, m_1) = 1.0d0 - sin(q1(i,1,m_1) * inv_cut)
        enddo
    enddo
    !$OMP END PARALLEL DO

    !$OMP PARALLEL DO PRIVATE(ni)
    do i = 1, na2
        ni = n2(i)
        do m_1 = 1, ni
            sin2(i, m_1) = 1.0d0 - sin(q2(i,1,m_1) * inv_cut)
        enddo
    enddo
    !$OMP END PARALLEL DO

    allocate(selfl21(na1))
    allocate(selfl22(na2))

    !$OMP PARALLEL DO PRIVATE(ni)
    do i = 1, na1
        selfl21(i) = atomic_distl2(q1(i,:,:), q1(i,:,:), n1(i), n1(i), &
            & sin1(i,:), sin1(i,:), width, cut_distance, r_width, c_width)
    enddo
    !$OMP END PARALLEL DO

    !$OMP PARALLEL DO PRIVATE(ni)
    do i = 1, na2
        selfl22(i) = atomic_distl2(q2(i,:,:), q2(i,:,:), n2(i), n2(i), &
            & sin2(i,:), sin2(i,:), width, cut_distance, r_width, c_width)
    enddo
    !$OMP END PARALLEL DO

    kernels(:,:,:) = 0.0d0

    !$OMP PARALLEL DO PRIVATE(l2dist) schedule(dynamic)
    do j = 1, na2
        do i = 1, na1

            l2dist = atomic_distl2(q1(i,:,:), q2(j,:,:), n1(i), n2(j), &
                & sin1(i,:), sin2(j,:), width, cut_distance, r_width, c_width)

            l2dist = selfl21(i) + selfl22(j) - 2.0d0 * l2dist &
                & * (r_width2/(r_width2 + (z1(i,1) - z2(j,1))**2) * &
                & c_width2/(c_width2 + (z1(i,2) - z2(j,2))**2))

            do k = 1, nsigmas
                kernels(k, i, j) =  exp(l2dist * inv_sigma2(k))
            enddo

        enddo
    enddo
    !$OMP END PARALLEL DO

    deallocate(selfl21)
    deallocate(selfl22)
    deallocate(sin1)
    deallocate(sin2)

end subroutine fget_atomic_kernels_arad


subroutine fget_atomic_symmetric_kernels_arad(q1, z1, n1, sigmas, na1, nsigmas, &
        & width, cut_distance, r_width, c_width, kernels)

    use arad, only: atomic_distl2

    implicit none

    ! ARAD descriptors for each atom in the training set, format (i,5,m_1)
    double precision, dimension(:,:,:), intent(in) :: q1

    ! ARAD atom-types for each atom, format (i, 2)
    double precision, dimension(:,:), intent(in) :: z1

    ! Sigma in the Gaussian kernel
    double precision, dimension(:), intent(in) :: sigmas

    ! List of numbers of atoms in each molecule
    integer, dimension(:), intent(in) :: n1

    ! Number of atom
    integer, intent(in) :: na1

    ! Number of sigmas
    integer, intent(in) :: nsigmas

    ! -1.0 / sigma^2 for use in the kernel
    double precision, dimension(nsigmas) :: inv_sigma2

    ! ARAD parameters
    double precision, intent(in) :: width
    double precision, intent(in) :: cut_distance
    double precision, intent(in) :: r_width
    double precision, intent(in) :: c_width

    ! Resulting alpha vector
    double precision, dimension(nsigmas,na1,na1), intent(out) :: kernels

    ! Internal counters
    integer :: i, j, k, ni
    integer :: m_1

    ! Pre-computed constants
    double precision :: r_width2
    double precision :: c_width2
    double precision :: inv_cut

    ! Temporary variables necessary for parallelization
    double precision :: l2dist

    ! Pre-computed terms in the full distance matrix
    double precision, allocatable, dimension(:) :: selfl21

    ! Pre-computed sine terms
    double precision, allocatable, dimension(:,:) :: sin1
    
    ! Value of PI at full FORTRAN precision.
    double precision, parameter :: pi = 4.0d0 * atan(1.0d0)

    r_width2 = r_width**2
    c_width2 = c_width**2

    inv_cut = pi / (2.0d0 * cut_distance)
    inv_sigma2(:) = -1.0d0 / (sigmas(:))**2

    allocate(sin1(na1, maxval(n1)))

    !$OMP PARALLEL DO PRIVATE(ni)
    do i = 1, na1
        ni = n1(i)
        do m_1 = 1, ni
            sin1(i, m_1) = 1.0d0 - sin(q1(i,1,m_1) * inv_cut)
        enddo
    enddo
    !$OMP END PARALLEL DO

    allocate(selfl21(na1))

    !$OMP PARALLEL DO PRIVATE(ni)
    do i = 1, na1
        selfl21(i) = atomic_distl2(q1(i,:,:), q1(i,:,:), n1(i), n1(i), &
            & sin1(i,:), sin1(i,:), width, cut_distance, r_width, c_width)
    enddo
    !$OMP END PARALLEL DO

    kernels(:,:,:) = 0.0d0

    !$OMP PARALLEL DO PRIVATE(l2dist) schedule(dynamic)
    do j = 1, na1
        do i = j, na1

            l2dist = atomic_distl2(q1(i,:,:), q1(j,:,:), n1(i), n1(j), &
                & sin1(i,:), sin1(j,:), width, cut_distance, r_width, c_width)

            l2dist = selfl21(i) + selfl21(j) - 2.0d0 * l2dist &
                & * (r_width2/(r_width2 + (z1(i,1) - z1(j,1))**2) * &
                & c_width2/(c_width2 + (z1(i,2) - z1(j,2))**2))

            do k = 1, nsigmas
                kernels(k, i, j) =  exp(l2dist * inv_sigma2(k))
                kernels(k, j, i) =  exp(l2dist * inv_sigma2(k))
            enddo

        enddo
    enddo
    !$OMP END PARALLEL DO

    deallocate(selfl21)
    deallocate(sin1)

end subroutine fget_atomic_symmetric_kernels_arad


subroutine fget_global_symmetric_kernels_arad(q1, z1, n1, sigmas, nm1, nsigmas, &
        & width, cut_distance, r_width, c_width, kernels)

    use arad, only: atomic_distl2

    implicit none

    ! ARAD descriptors for the training set, format (i,j_1,5,m_1)
    double precision, dimension(:,:,:,:), intent(in) :: q1

    ! ARAD atom-types for each atom in each molecule, format (i, j_1, 2)
    double precision, dimension(:,:,:), intent(in) :: z1

    ! List of numbers of atoms in each molecule
    integer, dimension(:), intent(in) :: n1

    ! Sigma in the Gaussian kernel
    double precision, dimension(:), intent(in) :: sigmas

    ! Number of molecules
    integer, intent(in) :: nm1

    ! Number of sigmas
    integer, intent(in) :: nsigmas

    ! -1.0 / sigma^2 for use in the kernel
    double precision, dimension(nsigmas) :: inv_sigma2

    ! ARAD parameters
    double precision, intent(in) :: width
    double precision, intent(in) :: cut_distance
    double precision, intent(in) :: r_width
    double precision, intent(in) :: c_width

    ! Resulting alpha vector
    double precision, dimension(nsigmas,nm1,nm1), intent(out) :: kernels

    ! Internal counters
    integer :: i, j, k, ni, nj
    integer :: m_1, i_1, j_1

    ! Pre-computed constants
    double precision :: r_width2
    double precision :: c_width2
    double precision :: inv_cut

    ! Temporary variables necessary for parallelization
    double precision :: l2dist
    double precision, allocatable, dimension(:,:) :: atomic_distance

    ! Pre-computed terms in the full distance matrix
    double precision, allocatable, dimension(:) :: selfl21

    ! Pre-computed sine terms
    double precision, allocatable, dimension(:,:,:) :: sin1

    ! Value of PI at full FORTRAN precision.
    double precision, parameter :: pi = 4.0d0 * atan(1.0d0)
    double precision :: mol_dist

    r_width2 = r_width**2
    c_width2 = c_width**2

    inv_cut = pi / (2.0d0 * cut_distance)
    inv_sigma2(:) = -1.0d0 / (sigmas(:))**2

    allocate(sin1(nm1, maxval(n1), maxval(n1)))

    !$OMP PARALLEL DO PRIVATE(ni)
    do i = 1, nm1
        ni = n1(i)
        do m_1 = 1, ni
            do i_1 = 1, ni
                sin1(i, i_1, m_1) = 1.0d0 - sin(q1(i,i_1,1,m_1) * inv_cut)
            enddo
        enddo
    enddo
    !$OMP END PARALLEL DO

    allocate(selfl21(nm1))

    selfl21 = 0.0d0

    !$OMP PARALLEL DO PRIVATE(ni) REDUCTION(+:selfl21)
    do i = 1, nm1
        ni = n1(i)
        do i_1 = 1, ni
            do j_1 = 1, ni

                selfl21(i) = selfl21(i) + atomic_distl2(q1(i,i_1,:,:), q1(i,j_1,:,:), n1(i), n1(i), &
                    & sin1(i,i_1,:), sin1(i,j_1,:), width, cut_distance, r_width, c_width) &
                    & * (r_width2/(r_width2 + (z1(i,i_1,1) - z1(i,j_1,1))**2) * &
                       & c_width2/(c_width2 + (z1(i,i_1,2) - z1(i,j_1,2))**2))

            enddo
        enddo
    enddo
    !$OMP END PARALLEL DO

    allocate(atomic_distance(maxval(n1), maxval(n1)))

    kernels(:,:,:) = 0.0d0
    atomic_distance(:,:) = 0.0d0

    !$OMP PARALLEL DO PRIVATE(l2dist,atomic_distance,ni,nj,mol_dist) schedule(dynamic)
    do j = 1, nm1
        nj = n1(j)
        do i = 1, j! nm1

            ni = n1(i)

            atomic_distance(:,:) = 0.0d0

            do i_1 = 1, ni
                do j_1 = 1, nj

                    l2dist = atomic_distl2(q1(i,i_1,:,:), q1(j,j_1,:,:), n1(i), n1(j), &
                        & sin1(i,i_1,:), sin1(j,j_1,:), width, cut_distance, r_width, c_width)

                    L2dist =  l2dist * (r_width2/(r_width2 + (z1(i,i_1,1) - z1(j,j_1,1))**2) * &
                                      & c_width2/(c_width2 + (z1(i,i_1,2) - z1(j,j_1,2))**2))

                    atomic_distance(i_1,j_1) = l2dist

                enddo
            enddo

            mol_dist = selfl21(i) + selfl21(j) - 2.0d0 * sum(atomic_distance(:ni,:nj))

            do k = 1, nsigmas
                kernels(k, i, j) =  exp(mol_dist * inv_sigma2(k))
                kernels(k, j, i) =  kernels(k, i, j)
            enddo

        enddo
    enddo
    !$OMP END PARALLEL DO

    deallocate(atomic_distance)
    deallocate(selfl21)
    deallocate(sin1)

end subroutine fget_global_symmetric_kernels_arad


subroutine fget_global_kernels_arad(q1, q2, z1, z2, n1, n2, sigmas, nm1, nm2, nsigmas, &
        & width, cut_distance, r_width, c_width, kernels)

    use arad, only: atomic_distl2

    implicit none

    ! ARAD descriptors for the training set, format (i,j_1,5,m_1)
    double precision, dimension(:,:,:,:), intent(in) :: q1
    double precision, dimension(:,:,:,:), intent(in) :: q2

    ! ARAD atom-types for each atom in each molecule, format (i, j_1, 2)
    double precision, dimension(:,:,:), intent(in) :: z1
    double precision, dimension(:,:,:), intent(in) :: z2

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

    ! ARAD parameters
    double precision, intent(in) :: width
    double precision, intent(in) :: cut_distance
    double precision, intent(in) :: r_width
    double precision, intent(in) :: c_width

    ! Resulting alpha vector
    double precision, dimension(nsigmas,nm1,nm2), intent(out) :: kernels

    ! Internal counters
    integer :: i, j, k, ni, nj
    integer :: m_1, i_1, j_1

    ! Pre-computed constants
    double precision :: r_width2
    double precision :: c_width2
    double precision :: inv_cut

    ! Temporary variables necessary for parallelization
    double precision :: l2dist
    double precision, allocatable, dimension(:,:) :: atomic_distance

    ! Pre-computed terms in the full distance matrix
    double precision, allocatable, dimension(:) :: selfl21
    double precision, allocatable, dimension(:) :: selfl22

    ! Pre-computed sine terms
    double precision, allocatable, dimension(:,:,:) :: sin1
    double precision, allocatable, dimension(:,:,:) :: sin2

    ! Value of PI at full FORTRAN precision.
    double precision, parameter :: pi = 4.0d0 * atan(1.0d0)

    double precision :: mol_dist

    r_width2 = r_width**2
    c_width2 = c_width**2

    inv_cut = pi / (2.0d0 * cut_distance)
    inv_sigma2(:) = -1.0d0 / (sigmas(:))**2

    allocate(sin1(nm1, maxval(n1), maxval(n1)))
    allocate(sin2(nm2, maxval(n2), maxval(n2)))

    sin1 = 0.0d0
    sin2 = 0.0d0

    !$OMP PARALLEL DO PRIVATE(ni)
    do i = 1, nm1
        ni = n1(i)
        do m_1 = 1, ni
            do i_1 = 1, ni
                if (q1(i,i_1,1,m_1) < cut_distance) then
                    sin1(i, i_1, m_1) = 1.0d0 - sin(q1(i,i_1,1,m_1) * inv_cut)
                endif
            enddo
        enddo
    enddo
    !$OMP END PARALLEL DO

    !$OMP PARALLEL DO PRIVATE(ni)
    do i = 1, nm2
        ni = n2(i)
        do m_1 = 1, ni
            do i_1 = 1, ni
                if (q2(i,i_1,1,m_1) < cut_distance) then
                    sin2(i, i_1, m_1) = 1.0d0 - sin(q2(i,i_1,1,m_1) * inv_cut)
                endif
            enddo
        enddo
    enddo
    !$OMP END PARALLEL DO

    allocate(selfl21(nm1))
    allocate(selfl22(nm2))

    selfl21 = 0.0d0
    selfl22 = 0.0d0

    !$OMP PARALLEL DO PRIVATE(ni) REDUCTION(+:selfl21)
    do i = 1, nm1
        ni = n1(i)
        do i_1 = 1, ni
            do j_1 = 1, ni

                selfl21(i) = selfl21(i) + atomic_distl2(q1(i,i_1,:,:), q1(i,j_1,:,:), n1(i), n1(i), &
                    & sin1(i,i_1,:), sin1(i,j_1,:), width, cut_distance, r_width, c_width) &
                    & * (r_width2/(r_width2 + (z1(i,i_1,1) - z1(i,j_1,1))**2) * &
                       & c_width2/(c_width2 + (z1(i,i_1,2) - z1(i,j_1,2))**2))

            enddo

        enddo
    enddo
    !$OMP END PARALLEL DO

    !$OMP PARALLEL DO PRIVATE(ni) REDUCTION(+:selfl22)
    do i = 1, nm2
        ni = n2(i)
        do i_1 = 1, ni
            do j_1 = 1, ni

                selfl22(i) = selfl22(i) + atomic_distl2(q2(i,i_1,:,:), q2(i,j_1,:,:), n2(i), n2(i), &
                    & sin2(i,i_1,:), sin2(i,j_1,:), width, cut_distance, r_width, c_width) &
                    &* (r_width2/(r_width2 + (z2(i,i_1,1) - z2(i,j_1,1))**2) * &
                      & c_width2/(c_width2 + (z2(i,i_1,2) - z2(i,j_1,2))**2))


            enddo

        enddo
    enddo
    !$OMP END PARALLEL DO


    allocate(atomic_distance(maxval(n1), maxval(n2)))

    kernels(:,:,:) = 0.0d0
    atomic_distance(:,:) = 0.0d0


    !$OMP PARALLEL DO PRIVATE(l2dist,atomic_distance,ni,nj,mol_dist) schedule(dynamic)

    do j = 1, nm2
        nj = n2(j)
        do i = 1, nm1
            ni = n1(i)

            atomic_distance(:,:) = 0.0d0

            do i_1 = 1, ni
                do j_1 = 1, nj

                    l2dist = atomic_distl2(q1(i,i_1,:,:), q2(j,j_1,:,:), n1(i), n2(j), &
                        & sin1(i,i_1,:), sin2(j,j_1,:), width, cut_distance, r_width, c_width)


                    L2dist =  l2dist * (r_width2/(r_width2 + (z1(i,i_1,1) - z2(j,j_1,1))**2) * &
                       & c_width2/(c_width2 + (z1(i,i_1,2) - z2(j,j_1,2))**2))


                    atomic_distance(i_1,j_1) = l2dist

                enddo
            enddo

            mol_dist = selfl21(i) + selfl22(j) - 2.0d0 * sum(atomic_distance(:ni,:nj))

            do k = 1, nsigmas
                kernels(k, i, j) =  exp(mol_dist * inv_sigma2(k))

            enddo

        enddo
    enddo
    !$OMP END PARALLEL DO

    deallocate(atomic_distance)
    deallocate(selfl21)
    deallocate(selfl22)
    deallocate(sin1)
    deallocate(sin2)

end subroutine fget_global_kernels_arad
