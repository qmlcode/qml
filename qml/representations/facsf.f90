





















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

function calc_cos_angle(a, b, c) result(cos_angle)

    implicit none

    double precision, intent(in), dimension(3) :: a
    double precision, intent(in), dimension(3) :: b
    double precision, intent(in), dimension(3) :: c

    double precision, dimension(3) :: v1
    double precision, dimension(3) :: v2

    double precision :: cos_angle

    v1 = a - b
    v2 = c - b

    v1 = v1 / norm2(v1)
    v2 = v2 / norm2(v2)

    cos_angle = dot_product(v1,v2)

end function calc_cos_angle

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
    double precision, allocatable, dimension(:, :) :: distance_matrix, rdecay,  rep_subset

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
                cycle
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
    allocate(rep_subset(natoms, nelements * nbasis2))

    rep_subset = 0.0d0

    !$OMP PARALLEL DO PRIVATE(n,m,rij,radial) COLLAPSE(2) REDUCTION(+:rep_subset) SCHEDULE(dynamic)
    do i = 1, natoms
        do j = 1, natoms
            if (j .le. i) cycle
            rij = distance_matrix(i,j)
            if (rij <= rcut) then
                ! index of the element of atom i
                m = element_types(i)
                ! index of the element of atom j
                n = element_types(j)
                ! distance between atoms i and j
                ! two body term of the representation
                radial = exp(-eta2*(rij - Rs2)**2) * rdecay(i,j)
                rep_subset(i, (n-1)*nbasis2 + 1:n*nbasis2) = rep_subset(i, (n-1)*nbasis2 + 1:n*nbasis2) + radial
                rep_subset(j, (m-1)*nbasis2 + 1:m*nbasis2) = rep_subset(j, (m-1)*nbasis2 + 1:m*nbasis2) + radial
            endif
        enddo
    enddo
    !$OMP END PARALLEL DO

    rep(:, 1:nelements * nbasis2) = rep_subset(:,:)

    deallocate(radial)
    deallocate(rep_subset)

    ! number of radial basis functions in the three body term
    nbasis3 = size(Rs3)
    ! number of radial basis functions in the three body term
    nabasis = size(Ts)

    ! Inverse of the three body cutoff
    invcut = 1.0d0 / acut
    ! pre-calculate the radial decay in the three body terms
    rdecay = decay(distance_matrix, invcut, natoms)

    ! Allocate temporary
    allocate(rep_subset(natoms,rep_size - nelements * nbasis2))
    allocate(a(3))
    allocate(b(3))
    allocate(c(3))
    allocate(radial(nbasis3))
    allocate(angular(nabasis))

    rep_subset = 0.0d0

    ! This could probably be done more efficiently if it's a bottleneck
    ! Also the order is a bit wobbly compared to the tensorflow implementation
    !$OMP PARALLEL DO PRIVATE(rij, n, rik, m, a, b, c, angle, radial, angular, &
    !$OMP p, q, s, z) REDUCTION(+:rep_subset) COLLAPSE(2) SCHEDULE(dynamic)
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
                s = nbasis3 * nabasis * (-(p * (p + 1))/2 + q + nelements * p) + 1
                do l = 1, nbasis3
                    ! calculate the indices that the three body terms should be added to
                    z = s + (l-1) * nabasis
                    ! Add the contributions from atoms i,j and k
                    rep_subset(i, z:z + nabasis - 1) = rep_subset(i, z:z + nabasis - 1) + angular * radial(l)
                enddo
            enddo
        enddo
    enddo
    !$OMP END PARALLEL DO

    rep(:, nelements * nbasis2 + 1:) = rep_subset(:,:)

    deallocate(element_types)
    deallocate(rdecay)
    deallocate(distance_matrix)
    deallocate(rep_subset)
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
    double precision, allocatable, dimension(:, :) :: rep_subset
    double precision, allocatable, dimension(:, :, :) :: atom_grad
    double precision, allocatable, dimension(:, :, :, :) :: grad_subset

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
                cycle
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
        do j = 1, natoms
            if (j .le. i) cycle
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
    allocate(rep_subset(natoms, nelements * nbasis2))
    allocate(grad_subset(natoms, nelements * nbasis2, natoms, 3))

    grad_subset = 0.0d0
    rep_subset = 0.0d0

    !$OMP PARALLEL DO PRIVATE(m,n,rij,invrij,radial_base,radial,radial_part,part) REDUCTION(+:rep_subset,grad_subset) &
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
                rep_subset(i, (n-1)*nbasis2 + 1:n*nbasis2) = rep_subset(i, (n-1)*nbasis2 + 1:n*nbasis2) + radial
                rep_subset(j, (m-1)*nbasis2 + 1:m*nbasis2) = rep_subset(j, (m-1)*nbasis2 + 1:m*nbasis2) + radial
                ! Part of the gradients that can be reused for x,y and z coordinates
                radial_part = - radial_base * invrij * (2.0d0 * eta2 * (rij - Rs2) * rdecay(i,j) + &
                    & 0.5d0 * pi * sin(pi*rij * invcut) * invcut)
                do k = 1, 3
                    ! The gradients wrt coordinates
                    part = radial_part * (coordinates(i,k) - coordinates(j,k))
                    grad_subset(i, (n-1)*nbasis2 + 1:n*nbasis2, i, k) = &
                        grad_subset(i, (n-1)*nbasis2 + 1:n*nbasis2, i, k) + part
                    grad_subset(i, (n-1)*nbasis2 + 1:n*nbasis2, j, k) = &
                        grad_subset(i, (n-1)*nbasis2 + 1:n*nbasis2, j, k) - part
                    grad_subset(j, (m-1)*nbasis2 + 1:m*nbasis2, j, k) = &
                        grad_subset(j, (m-1)*nbasis2 + 1:m*nbasis2, j, k) - part
                    grad_subset(j, (m-1)*nbasis2 + 1:m*nbasis2, i, k) = &
                        grad_subset(j, (m-1)*nbasis2 + 1:m*nbasis2, i, k) + part
                enddo
            endif
        enddo
    enddo
    !$OMP END PARALLEL DO

    rep(:,:nelements*nbasis2) = rep_subset(:,:)
    grad(:,:nelements*nbasis2,:,:) = grad_subset(:,:,:,:)

    deallocate(radial_base)
    deallocate(radial)
    deallocate(radial_part)
    deallocate(part)
    deallocate(rep_subset)
    deallocate(grad_subset)


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
                    ! & (2.0d0 * sqrt(rij2 * rik2 - dot**2) * angular_base)
                    & (2.0d0 * max(1d-10, sqrt(abs(rij2 * rik2 - dot**2)) * angular_base))
                ! write(*,*) angular_base
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
                s = nbasis3 * nabasis * (-(p * (p + 1))/2 + q + nelements * p) + 1
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
        rep(i, twobody_size + 1:) = atom_rep
        grad(i, twobody_size + 1:,:,:) = atom_grad
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


subroutine fgenerate_fchl_acsf(coordinates, nuclear_charges, elements, &
                          & Rs2, Rs3, Ts, eta2, eta3, zeta, rcut, acut, natoms, natoms_tot, rep_size, &
                          & two_body_decay, three_body_decay, three_body_weight, rep)

    use acsf_utils, only: decay, calc_angle, calc_cos_angle

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
    integer, intent(in) :: natoms, natoms_tot ! natoms_tot includes "virtual" atoms created to satisfy periodic boundary conditions.
    integer, intent(in) :: rep_size
    double precision, intent(in) :: two_body_decay
    double precision, intent(in) :: three_body_decay
    double precision, intent(in) :: three_body_weight

    double precision, intent(out), dimension(natoms, rep_size) :: rep

    integer:: two_body_rep_size
    integer :: i, j, k, l, n, m, o, p, q, s, z, nelements, nbasis2, nbasis3, nabasis, j_id, j_id_max
    integer, allocatable, dimension(:) :: element_types
    double precision :: rij, rik, angle, cos_1, cos_2, cos_3, invcut
    ! double precision :: angle_1, angle_2, angle_3
    double precision, allocatable, dimension(:) :: radial, angular, a, b, c
    double precision, allocatable, dimension(:, :) :: distance_matrix, rdecay
    double precision, allocatable, dimension(:, :) :: saved_radial
    integer, allocatable, dimension(:):: saved_j

    double precision :: mu, sigma, ksi3

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
    allocate(element_types(natoms_tot))

    ! Store element index of every atom
    !$OMP PARALLEL DO SCHEDULE(dynamic)
    do i = 1, natoms_tot
        do j = 1, nelements
            if (nuclear_charges(modulo(i-1, natoms)+1) .eq. elements(j)) then
                element_types(i) = j
                exit
            endif
        enddo
    enddo
    !$OMP END PARALLEL DO


    ! Get distance matrix
    ! Allocate temporary
    allocate(distance_matrix(natoms_tot, natoms_tot))
    distance_matrix = 0.0d0


    !$OMP PARALLEL DO PRIVATE(rij) SCHEDULE(dynamic)
    do i = 1, natoms_tot
        do j = i+1, natoms_tot
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
    rdecay = decay(distance_matrix, invcut, natoms_tot)

    ! Allocate temporary
    allocate(radial(nbasis2))
    allocate(saved_radial(nbasis2, natoms_tot), saved_j(natoms_tot))
    rep=0.0d0
    !$OMP PARALLEL DO PRIVATE(n,m,rij,radial,mu,sigma,saved_radial, saved_j, j_id_max) SCHEDULE(dynamic)
    do i = 1, natoms
        ! index of the element of atom i
        m = element_types(i)
        j_id_max=0
        do j = i + 1, natoms_tot
            ! distance between atoms i and j
            rij = distance_matrix(i,j)
            if (rij > rcut) cycle
            j_id_max=j_id_max+1
            saved_j(j_id_max)=j
            ! index of the element of atom j
            n = element_types(j)
            ! two body term of the representation
            mu    = log(rij / sqrt(1.0d0 + eta2  / rij**2))
            sigma = sqrt(log(1.0d0 + eta2  / rij**2))
            radial(:) = 0.0d0

            do k = 1, nbasis2
                radial(k) = 1.0d0/(sigma* sqrt(2.0d0*pi) * Rs2(k)) * rdecay(i,j) &
                              & * exp( - (log(Rs2(k)) - mu)**2 / (2.0d0 * sigma**2) ) / rij**two_body_decay
            enddo
            saved_radial(:, j_id_max)=radial(:)
        enddo
        !$OMP CRITICAL
        do j_id = 1, j_id_max
            j=saved_j(j_id)
            n=element_types(j)
            rep(i, (n-1)*nbasis2 + 1:n*nbasis2) = rep(i, (n-1)*nbasis2 + 1:n*nbasis2) + saved_radial(:, j_id)
            if (j<=natoms) rep(j, (m-1)*nbasis2 + 1:m*nbasis2) = rep(j, (m-1)*nbasis2 + 1:m*nbasis2) + saved_radial(:, j_id)
        enddo
        !$OMP END CRITICAL
    enddo
    !$OMP END PARALLEL DO

    deallocate(radial)
    deallocate(saved_radial, saved_j)

    ! number of radial basis functions in the three body term
    nbasis3 = size(Rs3)
    ! number of radial basis functions in the three body term
    nabasis = size(Ts)

    ! Inverse of the three body cutoff
    invcut = 1.0d0 / acut
    ! pre-calculate the radial decay in the three body terms
    rdecay = decay(distance_matrix, invcut, natoms_tot)

    ! Allocate temporary
    allocate(a(3))
    allocate(b(3))
    allocate(c(3))
    allocate(radial(nbasis3))
    allocate(angular(nabasis))
    two_body_rep_size=nbasis2*nelements

    ! This could probably be done more efficiently if it's a bottleneck
    ! Also the order is a bit wobbly compared to the tensorflow implementation
    !$OMP PARALLEL DO PRIVATE(rij, n, rik, m, a, b, c, angle, radial, angular, &
    !$OMP cos_1, cos_2, cos_3, mu, sigma, o, ksi3, &
    !$OMP p, q, s, z, l) SCHEDULE(dynamic)
    do i = 1, natoms
        do j = 1, natoms_tot - 1
            if (i .eq. j) cycle
            ! distance between atoms i and j
            rij = distance_matrix(i,j)
            if (rij > acut)  cycle
            ! index of the element of atom j
            n = element_types(j)
            do k = j + 1, natoms_tot
                if (i .eq. k) cycle
                if (j .eq. k) cycle
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
                angle   = calc_angle(a,b,c)
                cos_1 = calc_cos_angle(a,b,c)
                cos_2 = calc_cos_angle(a,c,b)
                cos_3 = calc_cos_angle(b,a,c)

                ! The radial part of the three body terms including decay
                radial = exp(-eta3*(0.5d0 * (rij+rik) - Rs3)**2) * rdecay(i,j) * rdecay(i,k)

                ksi3 = (1.0d0 + 3.0d0 * cos_1 * cos_2 * cos_3) &
                     & / (distance_matrix(i,k) * distance_matrix(i,j) * distance_matrix(j,k) &
                 & )**three_body_decay * three_body_weight

                angular = 0.0d0
                do l = 1, nabasis/2

                    o = l*2-1
                    angular(2*l-1) = angular(2*l-1) + 2*cos(o * angle) &
                        & * exp(-(zeta * o)**2 /2)

                    angular(2*l) = angular(2*l) + 2*sin(o * angle) &
                        & * exp(-(zeta * o)**2 /2)

                enddo

                ! The lowest of the element indices for atoms j and k
                p = min(n,m) - 1
                ! The highest of the element indices for atoms j and k
                q = max(n,m) - 1
                ! calculate the indices that the three body terms should be added to
                s = two_body_rep_size + nbasis3 * nabasis * (-(p * (p + 1))/2 + q + nelements * p) + 1

                do l = 1, nbasis3
                    ! calculate the indices that the three body terms should be added to
                    z = s + (l-1) * nabasis
                    ! Add the contributions from atoms i,j and k
                    rep(i, z:z + nabasis - 1) = rep(i, z:z + nabasis - 1) + angular * radial(l) * ksi3
                enddo
            enddo
        enddo
    enddo
    !$OMP END PARALLEL DO

    deallocate(element_types)
    deallocate(rdecay)
    deallocate(distance_matrix)
    deallocate(a)
    deallocate(b)
    deallocate(c)
    deallocate(radial)
    deallocate(angular)

end subroutine fgenerate_fchl_acsf

subroutine fgenerate_fchl_acsf_and_gradients(coordinates, nuclear_charges, elements, &
    & Rs2, Rs3, Ts, eta2, eta3, zeta, rcut, acut, natoms, natoms_tot, rep_size, &
    & two_body_decay, three_body_decay, three_body_weight, rep, grad)

    use acsf_utils, only: decay, calc_angle, calc_cos_angle

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

    double precision, intent(in) :: two_body_decay
    double precision, intent(in) :: three_body_decay
    double precision, intent(in) :: three_body_weight

    double precision :: mu, sigma, dx, exp_s2, scaling, dscal, ddecay
    double precision :: cos_i, cos_j, cos_k
    double precision, allocatable, dimension(:) :: exp_ln
    double precision, allocatable, dimension(:) :: log_Rs2

    integer, intent(in) :: natoms, natoms_tot ! natoms_tot includes "virtual" atoms created to satisfy periodic boundary conditions.
    integer, intent(in) :: rep_size
    double precision, intent(out), dimension(natoms, rep_size) :: rep
    double precision, intent(out), dimension(natoms, rep_size, natoms, 3) :: grad

!   Introduced to make OpenMP parallelization easier.
    double precision, dimension(:, :, :), allocatable:: add_rep
    double precision, dimension(:, :, :, :), allocatable :: add_grad

    integer :: i, j, k, l, m, n,  p, q, s, t, z, nelements, nbasis2, nbasis3, nabasis, twobody_size
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

    double precision :: atm, atm_i, atm_j, atm_k
    double precision :: invrjk, invr_atm, vi, vj, vk
    double precision, allocatable, dimension(:) :: d_atm_i, d_atm_j, d_atm_k
    double precision, allocatable, dimension(:) :: d_atm_ii, d_atm_ji, d_atm_ki
    double precision, allocatable, dimension(:) :: d_atm_ij, d_atm_jj, d_atm_kj
    double precision, allocatable, dimension(:) :: d_atm_ik, d_atm_jk, d_atm_kk
    double precision, allocatable, dimension(:) :: d_atm_i2, d_atm_j2, d_atm_k2
    double precision, allocatable, dimension(:) :: d_atm_i3, d_atm_j3, d_atm_k3
    double precision, allocatable, dimension(:) :: d_atm_extra_i, d_atm_extra_j, d_atm_extra_k

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
    allocate(element_types(natoms_tot))

    ! Store element index of every atom
    !$OMP PARALLEL DO SCHEDULE(dynamic)
    do i = 1, natoms_tot
        do j = 1, nelements
            if (nuclear_charges(modulo(i-1, natoms)+1) .eq. elements(j)) then
                element_types(i) = j
                exit
            endif
        enddo
    enddo
    !$OMP END PARALLEL DO



    ! Get distance matrix
    ! Allocate temporary
    allocate(distance_matrix(natoms_tot, natoms_tot))
    allocate(sq_distance_matrix(natoms_tot, natoms_tot))
    allocate(inv_distance_matrix(natoms_tot, natoms_tot))
    allocate(inv_sq_distance_matrix(natoms_tot, natoms_tot))
    distance_matrix = 0.0d0
    sq_distance_matrix = 0.0d0
    inv_distance_matrix = 0.0d0
    inv_sq_distance_matrix = 0.0d0


    !$OMP PARALLEL DO PRIVATE(rij,rij2,invrij,invrij2) SCHEDULE(dynamic)
    do i = 1, natoms_tot
        do j = i+1, natoms_tot
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
    rdecay = decay(distance_matrix, invcut, natoms_tot)

    ! Allocate temporary
    allocate(radial_base(nbasis2))
    allocate(radial(nbasis2))
    allocate(radial_part(nbasis2))
    allocate(part(nbasis2))
    allocate(exp_ln(nbasis2))
    allocate(log_Rs2(nbasis2))

    grad =0.0d0
    rep = 0.0d0

    allocate(add_rep(nbasis2, natoms, natoms), add_grad(nbasis2, 3, natoms, natoms))
    add_rep=0.0d0
    add_grad=0.0d0
    log_Rs2(:) = log(Rs2(:))

    !$OMP PARALLEL DO PRIVATE(m, n, rij, invrij, mu, sigma, exp_s2, exp_ln,&
    !$OMP scaling, radial_base, radial, dx, part, dscal, ddecay) SCHEDULE(dynamic)
    do i = 1, natoms
        ! The element index of atom i
        m = element_types(i) ! moved inside loop to enable COLLAPSE
        do j = i + 1, natoms_tot
            rij = distance_matrix(i,j)
            if (rij > rcut) continue
            ! The element index of atom j
            n = element_types(j)
            ! Distance between atoms i and j
            invrij = inv_distance_matrix(i,j)
            mu    = log(rij / sqrt(1.0d0 + eta2  * inv_sq_distance_matrix(i, j)))
            sigma = sqrt(log(1.0d0 + eta2  * inv_sq_distance_matrix(i, j)))
            exp_s2 = exp(sigma**2)
            exp_ln = exp(-(log_Rs2(:) - mu)**2 / sigma**2 * 0.5d0) * sqrt(2.0d0)

            scaling = 1.0d0 / rij**two_body_decay


            radial_base(:) = 1.0d0/(sigma* sqrt(2.0d0*pi) * Rs2(:)) * exp(-(log_Rs2(:) - mu)**2 / (2.0d0 * sigma**2))

            radial(:) = radial_base(:) * scaling * rdecay(i,j)

            rep(i, (n-1)*nbasis2 + 1:n*nbasis2) = rep(i, (n-1)*nbasis2 + 1:n*nbasis2) + radial
            if (j<=natoms) add_rep(:, i, j)=radial
            do k = 1, 3
                dx = -(coordinates(i,k) - coordinates(j,k))
                part(:) = ((log_Rs2(:) - mu) * (-dx *(rij**2 * exp_s2 + eta2) / (rij * sqrt(exp_s2))**3) &
                            &* sqrt(exp_s2) / (sigma**2 * rij) + (log_Rs2(:) - mu) ** 2 * eta2 * dx / &
                            &(sigma**4 * rij**4 * exp_s2)) * exp_ln / (Rs2(:) * sigma  * sqrt(pi) * 2) &
                            &- exp_ln  * eta2 * dx / (Rs2(:) * sigma**3 *sqrt(pi) * rij**4 * exp_s2 * 2.0d0)

                dscal = two_body_decay * dx / rij**(two_body_decay+2.0d0)
                ddecay = dx * 0.5d0 * pi * sin(pi*rij * invcut) * invcut * invrij

                part(:) = part(:) * scaling * rdecay(i,j) + radial_base(:) * dscal * rdecay(i,j) &
                                & + radial_base(:) * scaling * ddecay

                ! The gradients wrt coordinates
                grad(i, (n-1)*nbasis2 + 1:n*nbasis2, i, k) = grad(i, (n-1)*nbasis2 + 1:n*nbasis2, i, k) + part
                if (j<=natoms) then
                    grad(i, (n-1)*nbasis2 + 1:n*nbasis2, j, k) = grad(i, (n-1)*nbasis2 + 1:n*nbasis2, j, k) - part
                    grad(j, (m-1)*nbasis2 + 1:m*nbasis2, i, k) = grad(j, (m-1)*nbasis2 + 1:m*nbasis2, i, k) + part
                    add_grad(:, k, i, j)=part
                endif
            enddo
        enddo
    enddo
    !$OMP END PARALLEL DO

    !$OMP PARALLEL DO PRIVATE(m) SCHEDULE(dynamic)
    do j = 2, natoms
        do i = 1, j-1
            if (distance_matrix(i,j)>rcut) cycle
            m = element_types(i)
            rep(j, (m-1)*nbasis2 + 1:m*nbasis2) = rep(j, (m-1)*nbasis2 + 1:m*nbasis2) + add_rep(:, i, j)
            do k=1, 3
                grad(j, (m-1)*nbasis2 + 1:m*nbasis2, j, k) = grad(j, (m-1)*nbasis2 + 1:m*nbasis2, j, k) - add_grad(:, k, i, j)
            enddo
        enddo
    enddo
    !$OMP END PARALLEL DO

    deallocate(add_rep, add_grad)

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
    rdecay = decay(distance_matrix, invcut, natoms_tot)

    ! Allocate temporary
    allocate(atom_rep(rep_size - twobody_size))
    allocate(atom_grad(rep_size - twobody_size, natoms, 3))
    allocate(a(3))
    allocate(b(3))
    allocate(c(3))
    allocate(radial(nbasis3))
    allocate(radial_base(nbasis3))
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

    allocate(d_atm_i(3))
    allocate(d_atm_j(3))
    allocate(d_atm_k(3))
    allocate(d_atm_ii(3))
    allocate(d_atm_ij(3))
    allocate(d_atm_ik(3))
    allocate(d_atm_ji(3))
    allocate(d_atm_jj(3))
    allocate(d_atm_jk(3))
    allocate(d_atm_ki(3))
    allocate(d_atm_kj(3))
    allocate(d_atm_kk(3))
    allocate(d_atm_i2(3))
    allocate(d_atm_j2(3))
    allocate(d_atm_k2(3))
    allocate(d_atm_i3(3))
    allocate(d_atm_j3(3))
    allocate(d_atm_k3(3))
    allocate(d_atm_extra_i(3))
    allocate(d_atm_extra_j(3))
    allocate(d_atm_extra_k(3))

    !$OMP PARALLEL DO PRIVATE(atom_rep, atom_grad, rij, n, rij2, invrij, invrij2,&
    !$OMP rik, m, rik2, invrik, invrjk, invrik2, a, b, c, angle, cos_i, cos_k,&
    !$OMP cos_j, radial_base, radial, p, q, dot, angular, d_angular, d_angular_d_j,&
    !$OMP d_angular_d_k, d_angular_d_i, d_radial, d_radial_d_j, d_radial_d_k,&
    !$OMP d_radial_d_i, d_ijdecay, d_ikdecay, invr_atm, atm, atm_i, atm_j, atm_k, vi,&
    !$OMP vj, vk, d_atm_ii, d_atm_ij, d_atm_ik, d_atm_ji, d_atm_jj, d_atm_jk, d_atm_ki,&
    !$OMP d_atm_kj, d_atm_kk, d_atm_extra_i, d_atm_extra_j, d_atm_extra_k, s, z) SCHEDULE(dynamic)
    do i = 1, natoms
        atom_rep = 0.0d0
        atom_grad = 0.0d0
        do j = 1, natoms_tot - 1
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
            do k = j + 1, natoms_tot
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
                ! inverse distance between atom j and k
                invrjk = inv_distance_matrix(j,k)
                ! inverse squared distance between atom i and k
                invrik2 = inv_sq_distance_matrix(i,k)
                ! coordinates of atoms j, i, k
                a = coordinates(j,:)
                b = coordinates(i,:)
                c = coordinates(k,:)
                ! angle between atom i, j and k, centered on i
                angle = calc_angle(a,b,c)
                cos_i = calc_cos_angle(a,b,c)
                cos_k = calc_cos_angle(a,c,b)
                cos_j = calc_cos_angle(b,a,c)

                ! part of the radial part of the 3body terms
                radial_base(:) = exp(-eta3*(0.5d0 * (rij+rik) - Rs3(:))**2)
                radial(:) = radial_base(:) ! * scaling

                p = min(n,m) - 1
                ! the highest index of the elements of j,k
                q = max(n,m) - 1
                ! Dot product between the vectors connecting atom i,j and i,k
                dot = dot_product(a-b,c-b)

                angular(1)   =  exp(-(zeta**2)*0.5d0) * 2 * cos(angle)
                angular(2)   =  exp(-(zeta**2)*0.5d0) * 2 * sin(angle)

                d_angular(1) =  exp(-(zeta**2)*0.5d0) * 2 * sin(angle) / sqrt(max(1d-10, rij2 * rik2 - dot**2))
                d_angular(2) = -exp(-(zeta**2)*0.5d0) * 2 * cos(angle) / sqrt(max(1d-10, rij2 * rik2 - dot**2))

                ! Part of the derivative of the angular basis functions wrt atom j (dim(3))
                d_angular_d_j = c - b + dot * ((b - a) * invrij2)
                ! Part of the derivative of the angular basis functions wrt atom k (dim(3))
                d_angular_d_k = a - b + dot * ((b - c) * invrik2)
                ! Part of the derivative of the angular basis functions wrt atom i (dim(3))
                d_angular_d_i = - (d_angular_d_j + d_angular_d_k)

                ! Part of the derivative of the radial basis functions wrt coordinates (dim(nbasis3))
                ! including decay
                d_radial = radial * eta3 * (0.5d0 * (rij+rik) - Rs3) ! * rdecay(i,j) * rdecay(i,k)
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

                invr_atm = (invrij * invrjk *invrik)**three_body_decay

                ! Axilrod-Teller-Muto term
                atm = (1.0d0 + 3.0d0 * cos_i * cos_j * cos_k) * invr_atm * three_body_weight

                atm_i = (3.0d0 * cos_j * cos_k) * invr_atm * invrij * invrik
                atm_j = (3.0d0 * cos_k * cos_i) * invr_atm * invrij * invrjk
                atm_k = (3.0d0 * cos_i * cos_j) * invr_atm * invrjk * invrik

                vi = dot_product(a-b,c-b)
                vj = dot_product(c-a,b-a)
                vk = dot_product(b-c,a-c)

                d_atm_ii(:) = 2 * b - a - c - vi * ((b-a)*invrij**2 + (b-c)*invrik**2)
                d_atm_ij(:) = c - a - vj * (b-a)*invrij**2
                d_atm_ik(:) = a - c - vk * (b-c)*invrik**2

                d_atm_ji(:) = c - b - vi * (a-b)*invrij**2
                d_atm_jj(:) = 2 * a - b - c - vj * ((a-b)*invrij**2 + (a-c)*invrjk**2)
                d_atm_jk(:) = b - c - vk * (a-c)*invrjk**2

                d_atm_ki(:) = a - b - vi * (c-b)*invrik**2
                d_atm_kj(:) = b - a - vj * (c-a)*invrjk**2
                d_atm_kk(:) = 2 * c - a - b - vk * ((c-a)*invrjk**2 + (c-b)*invrik**2)

                d_atm_extra_i(:) = ((a-b)*invrij**2 + (c-b)*invrik**2) * atm * three_body_decay / three_body_weight
                d_atm_extra_j(:) = ((b-a)*invrij**2 + (c-a)*invrjk**2) * atm * three_body_decay / three_body_weight
                d_atm_extra_k(:) = ((a-c)*invrjk**2 + (b-c)*invrik**2) * atm * three_body_decay / three_body_weight

                ! Get index of where the contributions of atoms i,j,k should be added
                s = nbasis3 * nabasis * (-(p * (p + 1))/2 + q + nelements * p) + 1

                do l = 1, nbasis3

                    ! Get index of where the contributions of atoms i,j,k should be added
                    z = s + (l-1) * nabasis

                    ! Add the contributions for atoms i,j,k
                    atom_rep(z:z + nabasis - 1) = atom_rep(z:z + nabasis - 1) &
                        & + angular * radial(l) * atm * rdecay(i,j) * rdecay(i,k)

                    do t = 1, 3

                        ! Add up all gradient contributions wrt atom i
                        atom_grad(z:z + nabasis - 1, i, t) = atom_grad(z:z + nabasis - 1, i, t) + &
                            & d_angular * d_angular_d_i(t) * radial(l) * atm * rdecay(i,j) * rdecay(i,k) + &
                            & angular * d_radial(l) * d_radial_d_i(t) * atm * rdecay(i,j) * rdecay(i,k) + &
                            & angular * radial(l) * (atm_i * d_atm_ii(t) + atm_j * d_atm_ij(t) &
                            & + atm_k * d_atm_ik(t) + d_atm_extra_i(t)) * three_body_weight * rdecay(i,j) * rdecay(i,k) + &
                            & angular * radial(l) * (d_ijdecay(t) * rdecay(i,k) + rdecay(i,j) * d_ikdecay(t)) * atm
                        ! Add up all gradient contributions wrt atom j
                        if (j<=natoms) &
                        atom_grad(z:z + nabasis - 1, j, t) = atom_grad(z:z + nabasis - 1, j, t) + &
                            & d_angular * d_angular_d_j(t) * radial(l) * atm * rdecay(i,j) * rdecay(i,k) + &
                            & angular * d_radial(l) * d_radial_d_j(t) * atm * rdecay(i,j) * rdecay(i,k) + &
                            & angular * radial(l) * (atm_i * d_atm_ji(t) + atm_j * d_atm_jj(t) &
                            & + atm_k * d_atm_jk(t) + d_atm_extra_j(t)) * three_body_weight * rdecay(i,j) * rdecay(i,k) - &
                            & angular * radial(l) * d_ijdecay(t) * rdecay(i,k) * atm

                        ! Add up all gradient contributions wrt atom k
                        if (k<=natoms) &
                        atom_grad(z:z + nabasis - 1, k, t) = atom_grad(z:z + nabasis - 1, k, t) + &
                            & d_angular * d_angular_d_k(t) * radial(l) * atm * rdecay(i,j) * rdecay(i,k) + &
                            & angular * d_radial(l) * d_radial_d_k(t) * atm * rdecay(i,j) * rdecay(i,k) + &
                            & angular * radial(l) * (atm_i * d_atm_ki(t) + atm_j * d_atm_kj(t) &
                            & + atm_k * d_atm_kk(t) + d_atm_extra_k(t)) * three_body_weight * rdecay(i,j) * rdecay(i,k) - &
                            & angular * radial(l) * rdecay(i,j) * d_ikdecay(t) * atm

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


end subroutine fgenerate_fchl_acsf_and_gradients
