! MIT License
!
! Copyright (c) 2018 Lars Andersen Bratholm
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

module acsf_utils

    implicit none

contains

function decay(r, rc, natoms) result(f)

    implicit none

    double precision, intent(in), dimension(:,:) :: r
    double precision, intent(in) :: rc
    integer, intent(in) :: natoms
    double precision, dimension(natoms, natoms) :: f

    double precision, parameter :: pi = 4.0d0 * atan(1.0d0)

    f = 0.5d0 * (cos(pi*r / rc) + 1.0d0)


end function decay

function calc_angle(a, b, c) result(angle)

    implicit none

    double precision, intent(in), dimension(3) :: a
    double precision, intent(in), dimension(3) :: b
    double precision, intent(in), dimension(3) :: c

    double precision, dimension(3) :: v1
    double precision, dimension(3) :: v2

    double precision :: cos_angle
    double precision :: angle

    v1 = a - b
    v2 = c - b

    v1 = v1 / norm2(v1)
    v2 = v2 / norm2(v2)

    cos_angle = dot_product(v1,v2)

    ! Clipping
    if (cos_angle > 1.0d0) cos_angle = 1.0d0
    if (cos_angle < -1.0d0) cos_angle = -1.0d0

    angle = acos(cos_angle)

end function calc_angle

end module acsf_utils


subroutine fgenerate_acsf(coordinates, nuclear_charges, elements, &
                          & Rs2, Rs3, Ts, eta2, eta3, zeta, rcut, acut, natoms, descr_size, descr)

    use acsf_utils, only: decay, calc_angle

    implicit none

    double precision, intent(in), dimension(:, :) :: coordinates
    integer, intent(in), dimension(:) :: nuclear_charges
    integer, intent(in), dimension(:) :: elements
    double precision, intent(in), dimension(:) :: Rs2
    double precision, intent(in), dimension(:) :: Rs3
    double precision, intent(in), dimension(:) :: Ts
    double precision, intent(in) :: eta2
    double precision, intent(in) :: eta3
    double precision, intent(in) :: zeta
    double precision, intent(in) :: rcut
    double precision, intent(in) :: acut
    integer, intent(in) :: natoms
    integer, intent(in) :: descr_size
    double precision, intent(out), dimension(natoms, descr_size) :: descr

    integer :: i, j, k, l, n, m, p, q, r, z, nelements, nbasis2, nbasis3, nabasis
    integer, allocatable, dimension(:) :: element_types
    double precision :: norm, rij, rik, angle
    double precision, allocatable, dimension(:) :: radial, angular, a, b, c, atom_descr
    double precision, allocatable, dimension(:, :) :: distance_matrix, rdecay

    double precision, parameter :: pi = 4.0d0 * atan(1.0d0)

    if (natoms /= size(nuclear_charges, dim=1)) then
        write(*,*) "ERROR: Atom Centered Symmetry Functions creation"
        write(*,*) natoms, "coordinates, but", &
            & size(nuclear_charges, dim=1), "atom_types!"
        stop
    endif


    nelements = size(elements)
    ! Allocate temporary
    allocate(element_types(natoms))

    ! Store element index of every atom
    !$OMP PARALLEL DO
    do i = 1, natoms
        do j = 1, nelements
            if (nuclear_charges(i) .eq. elements(j)) then
                element_types(i) = j
                continue
            endif
        enddo
    enddo
    !$OMP END PARALLEL DO


    ! Get distance matrix
    ! Allocate temporary
    allocate(distance_matrix(natoms, natoms))
    distance_matrix = 0.0d0


    !$OMP PARALLEL DO PRIVATE(norm)
    do i = 1, natoms
        do j = i+1, natoms
            norm = sqrt(sum((coordinates(j,:) - coordinates(i,:))**2))
            distance_matrix(i, j) = norm
            distance_matrix(j, i) = norm
        enddo
    enddo
    !$OMP END PARALLEL DO

    nbasis2 = size(Rs2)
    descr = 0.0d0
    rdecay = decay(distance_matrix, rcut, natoms)

    ! Allocate temporary
    allocate(radial(nbasis2))

    !$OMP PARALLEL DO PRIVATE(n,m,norm,radial) REDUCTION(+:descr)
    do i = 1, natoms
        m = element_types(i)
        do j = i + 1, natoms
            n = element_types(j)
            norm = distance_matrix(i,j)
            if (norm <= rcut) then
                radial = exp(-eta2*(norm - Rs2)**2) * rdecay(i,j)
                descr(i, (n-1)*nbasis2 + 1:n*nbasis2) = descr(i, (n-1)*nbasis2 + 1:n*nbasis2) + radial
                descr(j, (m-1)*nbasis2 + 1:m*nbasis2) = descr(j, (m-1)*nbasis2 + 1:m*nbasis2) + radial
            endif
        enddo
    enddo
    !$OMP END PARALLEL DO

    deallocate(radial)

    nbasis3 = size(Rs3)
    nabasis = size(Ts)

    rdecay = decay(distance_matrix, acut, natoms)
    ! Allocate temporary
    allocate(radial(nbasis3))
    allocate(angular(nabasis))
    allocate(a(3))
    allocate(b(3))
    allocate(c(3))
    allocate(atom_descr(descr_size))

    ! This could probably be done more efficiently if it's a bottleneck
    ! Also the order is a bit wobbly compared to the tensorflow implementation
    !$OMP PARALLEL DO PRIVATE(n,m,p,q,z,rij,rik,angle,a,b,c,radial,angular,atom_descr)
    do i = 1, natoms
        atom_descr = 0.0d0
        do j = 1, natoms - 1
            if (i .eq. j) cycle
            n = element_types(j)
            rij = distance_matrix(i,j)
            if (rij > acut) cycle
            do k = j + 1, natoms
                if (i .eq. k) cycle
                m = element_types(k)
                rik = distance_matrix(i,k)
                if (rik > acut) cycle

                a = coordinates(j,:)
                b = coordinates(i,:)
                c = coordinates(k,:)
                angle = calc_angle(a,b,c)
                radial = exp(-eta3*(0.5d0 * (rij+rik) - Rs3)**2) * rdecay(i,j) * rdecay(i,k)
                angular = 2.0d0 * ((1.0d0 + cos(angle - Ts)) * 0.5d0) ** zeta
                p = min(n,m) - 1
                q = max(n,m) - 1
                do l = 1, nbasis3
                    !z = nelements * nbasis2 + nbasis3 * nabasis * (q - (p * (p + 1 - 2 * nelements)) / 2) &
                    !    & + (l - 1) * nbasis3 + 1
                    z = nelements * nbasis2 + nbasis3 * nabasis * (nelements * p + q) + (l-1) * nabasis + 1
                    atom_descr(z:z + nabasis) = atom_descr(z:z + nabasis) + angular * radial(l)
                    !do r = 1, nabasis
                    !    !write(*,'(i3, i3, i3, i3, i3, i3, i3, f8.5, f6.3, f6.3, f6.3, i3)') i, j, k, p, q, r, l, &
                    !    !    angular(r) * radial(l), rij, rik, angle, z+r
                    !    write(*,*) i, j, k, l, r, z+r-1, angular(r)*radial(l)
                    !enddo
                enddo
            enddo
        enddo
        descr(i,:) = descr(i,:) + atom_descr
    enddo
    !$OMP END PARALLEL DO

    deallocate(element_types)
    deallocate(distance_matrix)
    deallocate(radial)
    deallocate(angular)
    deallocate(a)
    deallocate(b)
    deallocate(c)
    deallocate(atom_descr)

end subroutine fgenerate_acsf
