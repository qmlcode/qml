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

function decay(r, invrc, natoms) result(f)

    implicit none

    double precision, intent(in), dimension(:,:) :: r
    double precision, intent(in) :: invrc
    integer, intent(in) :: natoms
    double precision, dimension(natoms, natoms) :: f

    double precision, parameter :: pi = 4.0d0 * atan(1.0d0)

    ! Decaying function reaching 0 at rc
    f = 0.5d0 * (cos(pi * r * invrc) + 1.0d0)


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
                          & Rs2, Rs3, Ts, eta2, eta3, zeta, rcut, acut, natoms, rep_size, rep)

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
    integer, intent(in) :: rep_size
    double precision, intent(out), dimension(natoms, rep_size) :: rep

    integer :: i, j, k, l, n, m, p, q, s, z, nelements, nbasis2, nbasis3, nabasis
    integer, allocatable, dimension(:) :: element_types
    double precision :: rij, rik, angle, invcut
    double precision, allocatable, dimension(:) :: radial, angular, a, b, c
    double precision, allocatable, dimension(:, :) :: distance_matrix, rdecay, rep3

    double precision, parameter :: pi = 4.0d0 * atan(1.0d0)

    if (natoms /= size(nuclear_charges, dim=1)) then
        write(*,*) "ERROR: Atom Centered Symmetry Functions creation"
        write(*,*) natoms, "coordinates, but", &
            & size(nuclear_charges, dim=1), "atom_types!"
        stop
    endif


    ! number of element types
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


    !$OMP PARALLEL DO PRIVATE(rij)
    do i = 1, natoms
        do j = i+1, natoms
            rij = norm2(coordinates(j,:) - coordinates(i,:))
            distance_matrix(i, j) = rij
            distance_matrix(j, i) = rij
        enddo
    enddo
    !$OMP END PARALLEL DO

    ! number of basis functions in the two body term
    nbasis2 = size(Rs2)

    ! Inverse of the two body cutoff
    invcut = 1.0d0 / rcut
    ! pre-calculate the radial decay in the two body terms
    rdecay = decay(distance_matrix, invcut, natoms)

    ! Allocate temporary
    allocate(radial(nbasis2))

    !$OMP PARALLEL DO PRIVATE(n,m,rij,radial) REDUCTION(+:rep)
    do i = 1, natoms
        ! index of the element of atom i
        m = element_types(i)
        do j = i + 1, natoms
            ! index of the element of atom j
            n = element_types(j)
            ! distance between atoms i and j
            rij = distance_matrix(i,j)
            if (rij <= rcut) then
                ! two body term of the representation
                radial = exp(-eta2*(rij - Rs2)**2) * rdecay(i,j)
                rep(i, (n-1)*nbasis2 + 1:n*nbasis2) = rep(i, (n-1)*nbasis2 + 1:n*nbasis2) + radial
                rep(j, (m-1)*nbasis2 + 1:m*nbasis2) = rep(j, (m-1)*nbasis2 + 1:m*nbasis2) + radial
            endif
        enddo
    enddo
    !$OMP END PARALLEL DO

    deallocate(radial)

    ! number of radial basis functions in the three body term
    nbasis3 = size(Rs3)
    ! number of radial basis functions in the three body term
    nabasis = size(Ts)

    ! Inverse of the three body cutoff
    invcut = 1.0d0 / acut
    ! pre-calculate the radial decay in the three body terms
    rdecay = decay(distance_matrix, invcut, natoms)

    ! Allocate temporary
    allocate(rep3(natoms,rep_size))
    allocate(a(3))
    allocate(b(3))
    allocate(c(3))
    allocate(radial(nbasis3))
    allocate(angular(nabasis))

    rep3 = 0.0d0

    ! This could probably be done more efficiently if it's a bottleneck
    ! Also the order is a bit wobbly compared to the tensorflow implementation
    !$OMP PARALLEL DO PRIVATE(rij, n, rik, m, a, b, c, angle, radial, angular, &
    !$OMP p, q, s, z) REDUCTION(+:rep3) COLLAPSE(2) SCHEDULE(dynamic)
    do i = 1, natoms
        do j = 1, natoms - 1
            if (i .eq. j) cycle
            ! distance between atoms i and j
            rij = distance_matrix(i,j)
            if (rij > acut) cycle
            ! index of the element of atom j
            n = element_types(j)
            do k = j + 1, natoms
                if (i .eq. k) cycle
                ! distance between atoms i and k
                rik = distance_matrix(i,k)
                if (rik > acut) cycle
                ! index of the element of atom k
                m = element_types(k)
                ! coordinates of atoms j, i, k
                a = coordinates(j,:)
                b = coordinates(i,:)
                c = coordinates(k,:)
                ! angle between atoms i, j and k centered on i
                angle = calc_angle(a,b,c)
                ! The radial part of the three body terms including decay
                radial = exp(-eta3*(0.5d0 * (rij+rik) - Rs3)**2) * rdecay(i,j) * rdecay(i,k)
                ! The angular part of the three body terms
                angular = 2.0d0 * ((1.0d0 + cos(angle - Ts)) * 0.5d0) ** zeta
                ! The lowest of the element indices for atoms j and k
                p = min(n,m) - 1
                ! The highest of the element indices for atoms j and k
                q = max(n,m) - 1
                ! calculate the indices that the three body terms should be added to
                s = nelements * nbasis2 + nbasis3 * nabasis * (nelements * p + q) + 1
                do l = 1, nbasis3
                    ! calculate the indices that the three body terms should be added to
                    z = s + (l-1) * nabasis
                    ! Add the contributions from atoms i,j and k
                    rep3(i, z:z + nabasis - 1) = rep3(i, z:z + nabasis - 1) + angular * radial(l)
                enddo
            enddo
        enddo
    enddo
    !$OMP END PARALLEL DO

    rep = rep + rep3

    deallocate(element_types)
    deallocate(rdecay)
    deallocate(distance_matrix)
    deallocate(rep3)
    deallocate(a)
    deallocate(b)
    deallocate(c)
    deallocate(radial)
    deallocate(angular)

end subroutine fgenerate_acsf

subroutine fgenerate_acsf_and_gradients(coordinates, nuclear_charges, elements, &
                          & Rs2, Rs3, Ts, eta2, eta3, zeta, rcut, acut, natoms, &
                          & rep_size, rep, grad)

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
    integer, intent(in) :: rep_size
    double precision, intent(out), dimension(natoms, rep_size) :: rep
    double precision, intent(out), dimension(natoms, rep_size, natoms, 3) :: grad

    integer :: i, j, k, l, n, m, p, q, s, t, z, nelements, nbasis2, nbasis3, nabasis, twobody_size
    integer, allocatable, dimension(:) :: element_types
    double precision :: rij, rik, angle, dot, rij2, rik2, invrij, invrik, invrij2, invrik2, invcut
    double precision, allocatable, dimension(:) :: radial_base, radial, angular, a, b, c, atom_rep
    double precision, allocatable, dimension(:) :: angular_base, d_radial, d_radial_d_i
    double precision, allocatable, dimension(:) :: d_radial_d_j, d_radial_d_k, d_ijdecay, d_ikdecay
    double precision, allocatable, dimension(:) :: d_angular, part, radial_part
    double precision, allocatable, dimension(:) :: d_angular_d_i, d_angular_d_j, d_angular_d_k
    double precision, allocatable, dimension(:, :) :: distance_matrix, rdecay, sq_distance_matrix
    double precision, allocatable, dimension(:, :) :: inv_distance_matrix, inv_sq_distance_matrix
    double precision, allocatable, dimension(:, :, :) :: atom_grad

    double precision, parameter :: pi = 4.0d0 * atan(1.0d0)

    if (natoms /= size(nuclear_charges, dim=1)) then
        write(*,*) "ERROR: Atom Centered Symmetry Functions creation"
        write(*,*) natoms, "coordinates, but", &
            & size(nuclear_charges, dim=1), "atom_types!"
        stop
    endif


    ! Number of unique elements
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
    allocate(sq_distance_matrix(natoms, natoms))
    allocate(inv_distance_matrix(natoms, natoms))
    allocate(inv_sq_distance_matrix(natoms, natoms))
    distance_matrix = 0.0d0
    sq_distance_matrix = 0.0d0
    inv_distance_matrix = 0.0d0
    inv_sq_distance_matrix = 0.0d0


    !$OMP PARALLEL DO PRIVATE(rij,rij2,invrij,invrij2) SCHEDULE(dynamic)
    do i = 1, natoms
        do j = i+1, natoms
            rij = norm2(coordinates(j,:) - coordinates(i,:))
            distance_matrix(i, j) = rij
            distance_matrix(j, i) = rij
            rij2 = rij * rij
            sq_distance_matrix(i, j) = rij2
            sq_distance_matrix(j, i) = rij2
            invrij = 1.0d0 / rij
            inv_distance_matrix(i, j) = invrij
            inv_distance_matrix(j, i) = invrij
            invrij2 = invrij * invrij
            inv_sq_distance_matrix(i, j) = invrij2
            inv_sq_distance_matrix(j, i) = invrij2
        enddo
    enddo
    !$OMP END PARALLEL DO


    ! Number of two body basis functions
    nbasis2 = size(Rs2)

    ! Inverse of the two body cutoff distance
    invcut = 1.0d0 / rcut
    ! Pre-calculate the two body decay
    rdecay = decay(distance_matrix, invcut, natoms)

    ! Allocate temporary
    allocate(radial_base(nbasis2))
    allocate(radial(nbasis2))
    allocate(radial_part(nbasis2))
    allocate(part(nbasis2))

    !$OMP PARALLEL DO PRIVATE(m,n,rij,invrij,radial_base,radial,radial_part,part) REDUCTION(+:rep,grad) &
    !$OMP SCHEDULE(dynamic)
    do i = 1, natoms
        ! The element index of atom i
        m = element_types(i)
        do j = i + 1, natoms
            ! The element index of atom j
            n = element_types(j)
            ! Distance between atoms i and j
            rij = distance_matrix(i,j)
            if (rij <= rcut) then
                invrij = inv_distance_matrix(i,j)
                ! part of the two body terms
                radial_base = exp(-eta2*(rij - Rs2)**2)
                ! The full two body term between atom i and j
                radial = radial_base * rdecay(i,j)
                ! Add the contributions from atoms i and j
                rep(i, (n-1)*nbasis2 + 1:n*nbasis2) = rep(i, (n-1)*nbasis2 + 1:n*nbasis2) + radial
                rep(j, (m-1)*nbasis2 + 1:m*nbasis2) = rep(j, (m-1)*nbasis2 + 1:m*nbasis2) + radial
                ! Part of the gradients that can be reused for x,y and z coordinates
                radial_part = - radial_base * invrij * (2.0d0 * eta2 * (rij - Rs2) * rdecay(i,j) + &
                    & 0.5d0 * pi * sin(pi*rij * invcut) * invcut)
                do k = 1, 3
                    ! The gradients wrt coordinates
                    part = radial_part * (coordinates(i,k) - coordinates(j,k))
                    grad(i, (n-1)*nbasis2 + 1:n*nbasis2, i, k) = grad(i, (n-1)*nbasis2 + 1:n*nbasis2, i, k) + part
                    grad(i, (n-1)*nbasis2 + 1:n*nbasis2, j, k) = grad(i, (n-1)*nbasis2 + 1:n*nbasis2, j, k) - part
                    grad(j, (m-1)*nbasis2 + 1:m*nbasis2, j, k) = grad(j, (m-1)*nbasis2 + 1:m*nbasis2, j, k) - part
                    grad(j, (m-1)*nbasis2 + 1:m*nbasis2, i, k) = grad(j, (m-1)*nbasis2 + 1:m*nbasis2, i, k) + part
                enddo
            endif
        enddo
    enddo
    !$OMP END PARALLEL DO

    deallocate(radial_base)
    deallocate(radial)
    deallocate(radial_part)
    deallocate(part)


    ! Number of radial basis functions in the three body term
    nbasis3 = size(Rs3)
    ! Number of angular basis functions in the three body term
    nabasis = size(Ts)
    ! Size of two body terms
    twobody_size = nelements * nbasis2

    ! Inverse of the three body cutoff distance
    invcut = 1.0d0 / acut
    ! Pre-calculate the three body decay
    rdecay = decay(distance_matrix, invcut, natoms)

    ! Allocate temporary
    allocate(atom_rep(rep_size - twobody_size))
    allocate(atom_grad(rep_size - twobody_size, natoms, 3))
    allocate(a(3))
    allocate(b(3))
    allocate(c(3))
    allocate(radial(nbasis3))
    allocate(angular_base(nabasis))
    allocate(angular(nabasis))
    allocate(d_angular(nabasis))
    allocate(d_angular_d_i(3))
    allocate(d_angular_d_j(3))
    allocate(d_angular_d_k(3))
    allocate(d_radial(nbasis3))
    allocate(d_radial_d_i(3))
    allocate(d_radial_d_j(3))
    allocate(d_radial_d_k(3))
    allocate(d_ijdecay(3))
    allocate(d_ikdecay(3))

    ! This could probably be done more efficiently if it's a bottleneck
    ! The order is a bit wobbly compared to the tensorflow implementation
    !$OMP PARALLEL DO PRIVATE(atom_rep,atom_grad,a,b,c,radial,angular_base, &
    !$OMP angular,d_angular,rij,n,rij2,invrij,invrij2,d_angular_d_i, &
    !$OMP d_angular_d_j,d_angular_d_k,rik,m,rik2,invrik,invrik2,angle, &
    !$OMP p,q,dot,d_radial,d_radial_d_i,d_radial_d_j,d_radial_d_k,s,z, &
    !$OMP d_ijdecay,d_ikdecay) SCHEDULE(dynamic)
    do i = 1, natoms
        atom_rep = 0.0d0
        atom_grad = 0.0d0
        do j = 1, natoms - 1
            if (i .eq. j) cycle
            ! distance between atom i and j
            rij = distance_matrix(i,j)
            if (rij > acut) cycle
            ! index of the element of atom j
            n = element_types(j)
            ! squared distance between atom i and j
            rij2 = sq_distance_matrix(i,j)
            ! inverse distance between atom i and j
            invrij = inv_distance_matrix(i,j)
            ! inverse squared distance between atom i and j
            invrij2 = inv_sq_distance_matrix(i,j)
            do k = j + 1, natoms
                if (i .eq. k) cycle
                ! distance between atom i and k
                rik = distance_matrix(i,k)
                if (rik > acut) cycle
                ! index of the element of atom k
                m = element_types(k)
                ! squared distance between atom i and k
                rik2 = sq_distance_matrix(i,k)
                ! inverse distance between atom i and k
                invrik = inv_distance_matrix(i,k)
                ! inverse squared distance between atom i and k
                invrik2 = inv_sq_distance_matrix(i,k)
                ! coordinates of atoms j, i, k
                a = coordinates(j,:)
                b = coordinates(i,:)
                c = coordinates(k,:)
                ! angle between atom i, j and k, centered on i
                angle = calc_angle(a,b,c)
                ! part of the radial part of the 3body terms
                radial = exp(-eta3*(0.5d0 * (rij+rik) - Rs3)**2)
                ! used in the angular part of the 3body terms and in gradients
                angular_base = ((1.0d0 + cos(angle - Ts)) * 0.5d0)
                ! angular part of the 3body terms
                angular = 2.0d0 * angular_base ** zeta
                ! the lowest index of the elements of j,k
                p = min(n,m) - 1
                ! the highest index of the elements of j,k
                q = max(n,m) - 1
                ! Dot product between the vectors connecting atom i,j and i,k
                dot = dot_product(a-b,c-b)
                ! Part of the derivative of the angular basis functions wrt coordinates (dim(nabasis))
                ! including decay
                d_angular = (zeta * angular * sin(angle-Ts) * rdecay(i,j) * rdecay(i,k)) / &
                    & (2.0d0 * sqrt(rij2 * rik2 - dot**2) * angular_base)
                ! Part of the derivative of the angular basis functions wrt atom j (dim(3))
                d_angular_d_j = c - b + dot * ((b - a) * invrij2)
                ! Part of the derivative of the angular basis functions wrt atom k (dim(3))
                d_angular_d_k = a - b + dot * ((b - c) * invrik2)
                ! Part of the derivative of the angular basis functions wrt atom i (dim(3))
                d_angular_d_i = - (d_angular_d_j + d_angular_d_k)
                ! Part of the derivative of the radial basis functions wrt coordinates (dim(nbasis3))
                ! including decay
                d_radial = radial * eta3 * (0.5d0 * (rij+rik) - Rs3) * rdecay(i,j) * rdecay(i,k)
                ! Part of the derivative of the radial basis functions wrt atom j (dim(3))
                d_radial_d_j = (b - a) * invrij
                ! Part of the derivative of the radial basis functions wrt atom k (dim(3))
                d_radial_d_k = (b - c) * invrik
                ! Part of the derivative of the radial basis functions wrt atom i (dim(3))
                d_radial_d_i = - (d_radial_d_j + d_radial_d_k)
                ! Part of the derivative of the i,j decay functions wrt coordinates (dim(3))
                d_ijdecay = - pi * (b - a) * sin(pi * rij * invcut) * 0.5d0 * invrij * invcut
                ! Part of the derivative of the i,k decay functions wrt coordinates (dim(3))
                d_ikdecay = - pi * (b - c) * sin(pi * rik * invcut) * 0.5d0 * invrik * invcut

                ! Get index of where the contributions of atoms i,j,k should be added
                s = nbasis3 * nabasis * (nelements * p + q) + 1
                do l = 1, nbasis3
                    ! Get index of where the contributions of atoms i,j,k should be added
                    z = s + (l-1) * nabasis
                    ! Add the contributions for atoms i,j,k
                    atom_rep(z:z + nabasis - 1) = atom_rep(z:z + nabasis - 1) + angular * radial(l) * rdecay(i,j) * rdecay(i,k)
                    do t = 1, 3
                        ! Add up all gradient contributions wrt atom i
                        atom_grad(z:z + nabasis - 1, i, t) = atom_grad(z:z + nabasis - 1, i, t) + &
                            & d_angular * d_angular_d_i(t) * radial(l) + &
                            & angular * d_radial(l) * d_radial_d_i(t) + &
                            & angular * radial(l) * (d_ijdecay(t) * rdecay(i,k) + rdecay(i,j) * d_ikdecay(t))
                        ! Add up all gradient contributions wrt atom j
                        atom_grad(z:z + nabasis - 1, j, t) = atom_grad(z:z + nabasis - 1, j, t) + &
                            & d_angular * d_angular_d_j(t) * radial(l) + &
                            & angular * d_radial(l) * d_radial_d_j(t) - &
                            & angular * radial(l) * d_ijdecay(t) * rdecay(i,k)
                        ! Add up all gradient contributions wrt atom k
                        atom_grad(z:z + nabasis - 1, k, t) = atom_grad(z:z + nabasis - 1, k, t) + &
                            & d_angular * d_angular_d_k(t) * radial(l) + &
                            & angular * d_radial(l) * d_radial_d_k(t) - &
                            & angular * radial(l) * rdecay(i,j) * d_ikdecay(t) 
                    enddo
                enddo
            enddo
        enddo
        rep(i, twobody_size + 1:) = rep(i, twobody_size + 1:) + atom_rep
        grad(i, twobody_size + 1:,:,:) = grad(i, twobody_size + 1:,:,:) + atom_grad
    enddo
    !$OMP END PARALLEL DO


    deallocate(rdecay)
    deallocate(element_types)
    deallocate(distance_matrix)
    deallocate(inv_distance_matrix)
    deallocate(sq_distance_matrix)
    deallocate(inv_sq_distance_matrix)
    deallocate(atom_rep)
    deallocate(atom_grad)
    deallocate(a)
    deallocate(b)
    deallocate(c)
    deallocate(radial)
    deallocate(angular_base)
    deallocate(angular)
    deallocate(d_angular)
    deallocate(d_angular_d_i)
    deallocate(d_angular_d_j)
    deallocate(d_angular_d_k)
    deallocate(d_radial)
    deallocate(d_radial_d_i)
    deallocate(d_radial_d_j)
    deallocate(d_radial_d_k)
    deallocate(d_ijdecay)
    deallocate(d_ikdecay)


end subroutine fgenerate_acsf_and_gradients
