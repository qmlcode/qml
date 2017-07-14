module fchl_utils

    implicit none

contains


pure function calc_angle(a, b, c) result(angle)

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


pure function calc_cos_angle(a, b, c) result(cos_angle)

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


pure function calc_G(a, b, c, d) result(G)

    implicit none

    double precision, intent(in), dimension(3) :: a
    double precision, intent(in), dimension(3) :: b
    double precision, intent(in), dimension(3) :: c
    double precision, intent(in), dimension(3) :: d

    double precision :: G

    G = 3.0d0 / (norm2(a) * norm2(b) * norm2(c) * norm2(d))**3 * ( -4.0d0 + 3.0d0 &
        & * (dot_product(a,b)**2 + dot_product(a,c)**2 + dot_product(a,d)**2 &
        &  + dot_product(b,c)**2 + dot_product(b,d)**2 + dot_product(c,d)**2) &
        & - 9.0d0 * (dot_product(b,c) * dot_product(c,d) * dot_product(d,b)  & 
        &          + dot_product(c,d) * dot_product(d,a) * dot_product(a,c)  &
        &          + dot_product(a,b) * dot_product(b,d) * dot_product(d,a)  &
        &          + dot_product(b,c) * dot_product(c,a) * dot_product(a,b)) &
        & + 27.0d0 * (dot_product(a,b) * dot_product(b,c) * dot_product(c,d) * dot_product(d,a)))


end function calc_G

pure function calc_ksi4(Ai, Bi, Ci, Di) result(ksi4)

    implicit none

    double precision, intent(in), dimension(3) :: Ai
    double precision, intent(in), dimension(3) :: Bi
    double precision, intent(in), dimension(3) :: Ci
    double precision, intent(in), dimension(3) :: Di

    double precision, dimension(3) :: a
    double precision, dimension(3) :: b
    double precision, dimension(3) :: c
    double precision, dimension(3) :: d
    double precision, dimension(3) :: e
    double precision, dimension(3) :: f

    double precision :: ksi4

    a = Ci - Bi
    b = Ai - Ci
    c = Bi - Ai
    d = Ai - Di
    e = Bi - Di
    f = Ci - Di

    ksi4 = calc_G(c, a, f, d) + calc_G(c, e, f, b) + calc_G(b, a, e, d)

end function

function print_ksi4(X, nneighbors) result(ksi)

    implicit none

    double precision, dimension(:,:), intent(in) :: X
    integer, intent(in) :: nneighbors

    double precision :: ksi

    double precision, parameter :: pi = 4.0d0 * atan(1.0d0)

    double precision :: cos_phi
    double precision::dihedral
    integer :: a, b, c

    do a = 2, nneighbors
        do b = 2, nneighbors
            if (b.eq.a) cycle
            do c = 2, nneighbors
                if ((c.eq.a).or.(c.eq.b)) cycle

                cos_phi= (cos(X(a+3,b)) - cos(X(a+3,c)) * cos(X(b+3,c))) &
                    & / (sin(X(a+3,c)) * sin(X(b+3,c)))

                dihedral = acos(cos_phi)

                write (*,*) a, b, c, acos(cos_phi) / pi * 180.0d0

            enddo
        enddo
    enddo

    ksi = 0.0d0

end function print_ksi4

pure function calc_ksi3(X, j, k) result(ksi3)

    implicit none

    double precision, dimension(:,:), intent(in) :: X

    integer, intent(in) :: j
    integer, intent(in) :: k

    double precision :: cos_i, cos_j, cos_k
    double precision :: di, dj, dk

    double precision :: ksi3

    cos_i = calc_cos_angle(x(3:5, k), x(3:5, 1), x(3:5, j))
    cos_j = calc_cos_angle(x(3:5, j), x(3:5, k), x(3:5, 1))
    cos_k = calc_cos_angle(x(3:5, 1), x(3:5, j), x(3:5, k))

    dk = x(1, j)
    dj = x(1, k)
    di = norm2(x(3:5, j) - x(3:5, k))

    ksi3 = (1.0d0 + 3.0d0 * cos_i*cos_j*cos_k) / (di * dj * dk)**3

end function calc_ksi3

pure function cross_product(a, b) result(cross)

    implicit none

    double precision, intent(in), dimension(3) :: a
    double precision, intent(in), dimension(3) :: b

    double precision, dimension(3) :: cross

    cross(1) = a(2) * b(3) - a(3) * b(2)
    cross(2) = a(3) * b(1) - a(1) * b(3)
    cross(3) = a(1) * b(2) - a(2) * b(1)

end function cross_product

pure function calc_dihedral(a, b, c, d) result(dihedral)

    implicit none

    double precision, intent(in), dimension(3) :: a
    double precision, intent(in), dimension(3) :: b
    double precision, intent(in), dimension(3) :: c
    double precision, intent(in), dimension(3) :: d

    double precision, dimension(3) :: b1 
    double precision, dimension(3) :: b2
    double precision, dimension(3) :: b3
    
    double precision, dimension(3) :: x12
    double precision, dimension(3) :: x23
    
    double precision :: dihedral

    b1 = b - a 
    b2 = c - b
    b3 = d - c

    x12 = cross_product(b1, b2)
    x23 = cross_product(b2,b3)

    dihedral = atan2(dot_product(cross_product(x12, x23), b2/norm2(b2)), dot_product(x12, x23))

    ! hack to remove the sign for now 
    dihedral = acos(cos(dihedral))

end function calc_dihedral


pure function atomic_distl2(X1, X2, N1, N2, ksi1, ksi2, sin1, sin2, cos1, cos2, &
    & t_width, d_width, cut_distance, order, pd, ang_norm2, &
    & distance_scale, angular_scale) result(aadist)

    implicit none

    double precision, dimension(:,:), intent(in) :: X1
    double precision, dimension(:,:), intent(in) :: X2

    integer, intent(in) :: N1
    integer, intent(in) :: N2

    double precision, dimension(:), intent(in) :: ksi1
    double precision, dimension(:), intent(in) :: ksi2

    double precision, dimension(:,:,:), intent(in) :: sin1
    double precision, dimension(:,:,:), intent(in) :: sin2
    double precision, dimension(:,:,:), intent(in) :: cos1
    double precision, dimension(:,:,:), intent(in) :: cos2

    double precision, intent(in) :: t_width
    double precision, intent(in) :: d_width 
    double precision, intent(in) :: cut_distance
    integer, intent(in) :: order
    double precision, dimension(:,:), intent(in) :: pd
    double precision, intent(in) :: angular_scale
    double precision, intent(in) :: distance_scale

    double precision :: aadist

    double precision :: d

    integer :: m_1, m_2

    integer :: i, m, p1, p2

    double precision :: angular 
    double precision :: maxgausdist2

    integer :: pmax1
    integer :: pmax2

    double precision :: inv_width
    double precision :: r2

    double precision, dimension(order) :: s
    
    double precision, parameter :: pi = 4.0d0 * atan(1.0d0)

    double precision :: g1 
    double precision :: a0 

    logical, allocatable, dimension(:) :: mask1
    logical, allocatable, dimension(:) :: mask2

    double precision :: temp, sin1_temp, cos1_temp
    double precision, intent(in):: ang_norm2

    pmax1 = int(maxval(x1(2,:n1)))
    pmax2 = int(maxval(x2(2,:n2)))

    allocate(mask1(pmax1))
    allocate(mask2(pmax2))
    mask1 = .true.
    mask2 = .true.

    do i = 1, n1
        mask1(int(x1(2,i))) = .false.
    enddo

    do i = 1, n2
        mask2(int(x2(2,i))) = .false.
    enddo

    a0 = 0.0d0
    g1 = sqrt(2.0d0 * pi)/ang_norm2

    do m = 1, order
        s(m) = g1 * exp(-(t_width * m)**2 / 2.0d0)
    enddo

    inv_width = -1.0d0 / (4.0d0 * d_width**2)

    maxgausdist2 = (8.0d0 * d_width)**2

    aadist = 1.0d0

    do m_1 = 2, N1

        if (X1(1, m_1) > cut_distance) exit

        do m_2 = 2, N2

            if (X2(1, m_2) > cut_distance) exit

            r2 = (X2(1,m_2) - X1(1,m_1))**2

            if (r2 < maxgausdist2) then

                d = exp(r2 * inv_width ) * pd(int(x1(2,m_1)), int(x2(2,m_2)))

                angular = a0 * a0

                do m = 1, order

                    temp = 0.0d0

                    do p1 = 1, pmax1
                        if (mask1(p1)) cycle
                        cos1_temp = cos1(p1,m,m_1)
                        sin1_temp = sin1(p1,m,m_1)

                        do p2 = 1, pmax2
                            if (mask2(p2)) cycle

                            temp = temp + (cos1_temp * cos2(p2,m,m_2) &
                                & + sin1_temp * sin2(p2,m,m_2)) * pd(p2,p1)

                        enddo 
                    enddo 

                    angular = angular + temp * s(m)

                enddo

                aadist = aadist + d * (ksi1(m_1) * ksi2(m_2) * distance_scale &
                    & + angular * angular_scale)

            end if
        end do
    end do

    aadist = aadist * pd(int(x1(2,1)), int(x2(2,1)))

    deallocate(mask1)
    deallocate(mask2)
end function atomic_distl2


end module fchl_utils


subroutine fget_kernels_fchl(x1, x2, n1, n2, nneigh1, nneigh2, &
       & sigmas, nm1, nm2, nsigmas, &
       & t_width, d_width, cut_distance, order, pd, &
       & distance_scale, angular_scale, kernels)

    use fchl_utils, only: atomic_distl2, calc_angle, calc_ksi3, calc_ksi4, calc_dihedral

    implicit none

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

    double precision, intent(in) :: t_width
    double precision, intent(in) :: d_width 
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
    integer :: a, b, m, n

    ! Temporary variables necessary for parallelization
    double precision :: l2dist
    double precision, allocatable, dimension(:,:) :: atomic_distance

    ! Pre-computed terms in the full distance matrix
    double precision, allocatable, dimension(:,:) :: selfl21
    double precision, allocatable, dimension(:,:) :: selfl22

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
    integer :: pj, pk
    integer :: pmax1
    integer :: pmax2
    integer :: nneighi
    double precision :: theta

    double precision :: ksi3
    double precision :: cos_m, sin_m
    double precision :: ang_norm2

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

    inv_sigma2(:) = -1.0d0 / (sigmas(:))**2

    allocate(ksi1(nm1, maxval(n1), maxval(nneigh1)))
    allocate(ksi2(nm2, maxval(n2), maxval(nneigh2)))

    ksi1 = 0.0d0
    ksi2 = 0.0d0

    !$OMP PARALLEL DO PRIVATE(ni, nneighi)
    do a = 1, nm1
        ni = n1(a)
        do i = 1, ni
            nneighi = nneigh1(a, i)
            do j = 2, nneighi
                ksi1(a, i, j) = 1.0d0 / x1(a, i, 1, j)**6
            enddo
        enddo
    enddo
    !$OMP END PARALLEL do

    !$OMP PARALLEL DO PRIVATE(ni, nneighi)
    do a = 1, nm2
        ni = n2(a)
        do i = 1, ni
            nneighi = nneigh2(a, i)
            do j = 2, nneighi
                ksi2(a, i, j) = 1.0d0 / x2(a, i, 1, j)**6
            enddo
        enddo
    enddo
    !$OMP END PARALLEL do


    allocate(cosp1(nm1, maxval(n1), pmax1, order, maxval(nneigh1)))
    allocate(sinp1(nm1, maxval(n1), pmax1, order, maxval(nneigh1)))

    cosp1 = 0.0d0
    sinp1 = 0.0d0

    !$OMP PARALLEL DO PRIVATE(ni, nneighi, ksi3, pj, pk, theta, sin_m, cos_m) REDUCTION(+:cosp1,sinp1)
    do a = 1, nm1
        ni = n1(a)

        do i = 1, ni
            nneighi = nneigh1(a, i)

            do j = 2, nneighi
                do k = j+1, nneighi

                    ksi3 = calc_ksi3(X1(a,i,:,:), j, k)
                    theta = calc_angle(x1(a, i, 3:5, j), &
                        &  x1(a, i, 3:5, 1), x1(a, i, 3:5, k))

                    pj =  int(x1(a,i,2,k))
                    pk =  int(x1(a,i,2,j))

                    do m = 1, order

                        cos_m = (cos(m * theta) - cos((theta + pi) * m))*ksi3
                        sin_m = (sin(m * theta) - sin((theta + pi) * m))*ksi3

                        cosp1(a, i, pj, m, j) = cosp1(a, i, pj, m, j) + cos_m
                        sinp1(a, i, pj, m, j) = sinp1(a, i, pj, m, j) + sin_m

                        cosp1(a, i, pk, m, k) = cosp1(a, i, pk, m, k) + cos_m
                        sinp1(a, i, pk, m, k) = sinp1(a, i, pk, m, k) + sin_m

                    enddo
                enddo
            enddo
        enddo
    enddo
    !$OMP END PARALLEL do

    allocate(cosp2(nm2, maxval(n2), pmax2, order, maxval(nneigh2)))
    allocate(sinp2(nm2, maxval(n2), pmax2, order, maxval(nneigh2)))

    cosp2 = 0.0d0
    sinp2 = 0.0d0

    !$OMP PARALLEL DO PRIVATE(ni, nneighi, ksi3, pj, pk, theta, cos_m, sin_m) REDUCTION(+:cosp2,sinp2)
    do a = 1, nm2
        ni = n2(a)

        do i = 1, ni
            nneighi = nneigh2(a, i)

            do j = 2, nneighi
                do k = j+1, nneighi

                    ksi3 = calc_ksi3(X2(a,i,:,:), j, k)
                    theta = calc_angle(x2(a, i, 3:5, j), &
                        &  x2(a, i, 3:5, 1), x2(a, i, 3:5, k))

                    pj =  int(x2(a,i,2,k))
                    pk =  int(x2(a,i,2,j))

                    do m = 1, order

                        cos_m = (cos(m * theta) - cos((theta + pi) * m))*ksi3
                        sin_m = (sin(m * theta) - sin((theta + pi) * m))*ksi3

                        cosp2(a, i, pj, m, j) = cosp2(a, i, pj, m, j) + cos_m
                        sinp2(a, i, pj, m, j) = sinp2(a, i, pj, m, j) + sin_m

                        cosp2(a, i, pk, m, k) = cosp2(a, i, pk, m, k) + cos_m
                        sinp2(a, i, pk, m, k) = sinp2(a, i, pk, m, k) + sin_m

                    enddo
                enddo
            enddo
        enddo
    enddo
    !$OMP END PARALLEL do

    allocate(selfl21(nm1, maxval(n1)))
    allocate(selfl22(nm2, maxval(n2)))

    !$OMP PARALLEL DO PRIVATE(ni)
    do a = 1, nm1
        ni = n1(a)
        do i = 1, ni
            selfl21(a,i) = atomic_distl2(x1(a,i,:,:), x1(a,i,:,:), &
                & nneigh1(a,i), nneigh1(a,i), ksi1(a,i,:), ksi1(a,i,:), &
                & sinp1(a,i,:,:,:), sinp1(a,i,:,:,:), &
                & cosp1(a,i,:,:,:), cosp1(a,i,:,:,:), &
                & t_width, d_width, cut_distance, order, & 
                & pd, ang_norm2,distance_scale, angular_scale)
        enddo
    enddo
    !$OMP END PARALLEL DO

    !$OMP PARALLEL DO PRIVATE(ni)
    do a = 1, nm2
        ni = n2(a)
        do i = 1, ni
            selfl22(a,i) = atomic_distl2(x2(a,i,:,:), x2(a,i,:,:), &
                & nneigh2(a,i), nneigh2(a,i), ksi2(a,i,:), ksi2(a,i,:), &
                & sinp2(a,i,:,:,:), sinp2(a,i,:,:,:), &
                & cosp2(a,i,:,:,:), cosp2(a,i,:,:,:), &
                & t_width, d_width, cut_distance, order, &
                & pd, ang_norm2, distance_scale, angular_scale)
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

                    l2dist = atomic_distl2(x1(a,i,:,:), x2(b,j,:,:), & 
                        & nneigh1(a,i), nneigh2(b,j), ksi1(a,i,:), ksi2(b,j,:), &
                        & sinp1(a,i,:,:,:), sinp2(b,j,:,:,:), &
                        & cosp1(a,i,:,:,:), cosp2(b,j,:,:,:), &
                        & t_width, d_width, cut_distance, order, &
                        & pd, ang_norm2, distance_scale, angular_scale)

                    l2dist = selfl21(a,i) + selfl22(b,j) - 2.0d0 * l2dist
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
    deallocate(selfl21)
    deallocate(selfl22)
    deallocate(ksi1)
    deallocate(ksi2)
    deallocate(cosp1)
    deallocate(cosp2)
    deallocate(sinp1)
    deallocate(sinp2)

end subroutine fget_kernels_fchl


subroutine fget_symmetric_kernels_fchl(x1, n1, nneigh1, sigmas, nm1, nsigmas, &
       & t_width, d_width, cut_distance, order, pd, &
       & distance_scale, angular_scale, kernels)

    use fchl_utils, only: atomic_distl2, calc_angle, calc_ksi3

    implicit none

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

    double precision, intent(in) :: t_width
    double precision, intent(in) :: d_width 
    double precision, intent(in) :: cut_distance
    integer, intent(in) :: order
    double precision, intent(in) :: distance_scale
    double precision, intent(in) :: angular_scale

    ! -1.0 / sigma^2 for use in the kernel
    double precision, dimension(nsigmas) :: inv_sigma2

    double precision, dimension(:,:), intent(in) :: pd

    ! Resulting alpha vector
    double precision, dimension(nsigmas,nm1,nm1), intent(out) :: kernels

    ! Internal counters
    integer :: i, j, k, ni, nj
    integer :: a, b, m, n

    ! Temporary variables necessary for parallelization
    double precision :: l2dist
    double precision, allocatable, dimension(:,:) :: atomic_distance

    ! Pre-computed terms in the full distance matrix
    double precision, allocatable, dimension(:,:) :: selfl21

    ! Pre-computed terms
    double precision, allocatable, dimension(:,:,:) :: ksi1

    double precision, allocatable, dimension(:,:,:,:,:) :: sinp1
    double precision, allocatable, dimension(:,:,:,:,:) :: cosp1

    ! Value of PI at full FORTRAN precision.
    double precision, parameter :: pi = 4.0d0 * atan(1.0d0)

    ! counter for periodic distance
    integer :: pj, pk
    integer :: pmax1
    integer :: nneighi
    double precision :: theta
    double precision :: sin_m, cos_m

    double precision :: ang_norm2

    double precision :: ksi3

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

    inv_sigma2(:) = -1.0d0 / (sigmas(:))**2

    allocate(ksi1(nm1, maxval(n1), maxval(nneigh1)))

    ksi1 = 0.0d0

    !$OMP PARALLEL DO PRIVATE(ni, nneighi)
    do a = 1, nm1
        ni = n1(a)
        do i = 1, ni
            nneighi = nneigh1(a, i)
            do j = 2, nneighi
                ksi1(a, i, j) = 1.0d0 / x1(a,i,1,j)**6
            enddo
        enddo
    enddo
    !$OMP END PARALLEL DO

    allocate(cosp1(nm1, maxval(n1), pmax1, order, maxval(nneigh1)))
    allocate(sinp1(nm1, maxval(n1), pmax1, order, maxval(nneigh1)))

    cosp1 = 0.0d0
    sinp1 = 0.0d0

    !$OMP PARALLEL DO PRIVATE(ni, nneighi, ksi3, pj, pk, theta, cos_m, sin_m) REDUCTION(+:cosp1,sinp1)
    do a = 1, nm1
        ni = n1(a)

        do i = 1, ni
            nneighi = nneigh1(a, i)

            do j = 2, nneighi
                do k = j+1, nneighi

                    ksi3 = calc_ksi3(X1(a,i,:,:), j, k)
                    theta = calc_angle(x1(a, i, 3:5, j), &
                        &  x1(a, i, 3:5, 1), x1(a, i, 3:5, k))

                    pj = int(x1(a,i,2,k))
                    pk = int(x1(a,i,2,j))

                    do m = 1, order

                        cos_m = (cos(m * theta) - cos((theta + pi) * m))*ksi3
                        sin_m = (sin(m * theta) - sin((theta + pi) * m))*ksi3

                        cosp1(a, i, pj, m, j) = cosp1(a, i, pj, m, j) + cos_m
                        sinp1(a, i, pj, m, j) = sinp1(a, i, pj, m, j) + sin_m

                        cosp1(a, i, pk, m, k) = cosp1(a, i, pk, m, k) + cos_m
                        sinp1(a, i, pk, m, k) = sinp1(a, i, pk, m, k) + sin_m

                    enddo
                enddo
            enddo
        enddo
    enddo
    !$OMP END PARALLEL DO

    allocate(selfl21(nm1, maxval(n1)))

    !$OMP PARALLEL DO PRIVATE(ni)
    do a = 1, nm1
        ni = n1(a)
        do i = 1, ni
            selfl21(a,i) = atomic_distl2(x1(a,i,:,:), x1(a,i,:,:), &
                & nneigh1(a,i), nneigh1(a,i), ksi1(a,i,:), ksi1(a,i,:), &
                & sinp1(a,i,:,:,:), sinp1(a,i,:,:,:), &
                & cosp1(a,i,:,:,:), cosp1(a,i,:,:,:), &
                & t_width, d_width, cut_distance, order, & 
                & pd, ang_norm2,distance_scale, angular_scale)
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

                    l2dist = atomic_distl2(x1(a,i,:,:), x1(b,j,:,:), &
                        & nneigh1(a,i), nneigh1(b,j), ksi1(a,i,:), ksi1(b,j,:), &
                        & sinp1(a,i,:,:,:), sinp1(b,j,:,:,:), &
                        & cosp1(a,i,:,:,:), cosp1(b,j,:,:,:), &
                        & t_width, d_width, cut_distance, order, &
                        & pd, ang_norm2, distance_scale, angular_scale)

                    l2dist = selfl21(a,i) + selfl21(b,j) - 2.0d0 * l2dist
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
    deallocate(selfl21)
    deallocate(ksi1)
    deallocate(cosp1)
    deallocate(sinp1)

end subroutine fget_symmetric_kernels_fchl


subroutine fget_global_symmetric_kernels_fchl(x1, n1, nneigh1, sigmas, nm1, nsigmas, &
       & t_width, d_width, cut_distance, order, pd, &
       & distance_scale, angular_scale, kernels)

    use fchl_utils, only: atomic_distl2, calc_angle, calc_ksi3

    implicit none

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

    double precision, intent(in) :: t_width
    double precision, intent(in) :: d_width 
    double precision, intent(in) :: cut_distance
    integer, intent(in) :: order
    double precision, intent(in) :: distance_scale
    double precision, intent(in) :: angular_scale

    ! -1.0 / sigma^2 for use in the kernel
    double precision, dimension(nsigmas) :: inv_sigma2

    double precision, dimension(:,:), intent(in) :: pd

    ! Resulting alpha vector
    double precision, dimension(nsigmas,nm1,nm1), intent(out) :: kernels

    ! Internal counters
    integer :: i, j, k, ni, nj
    integer :: a, b, m, n

    ! Temporary variables necessary for parallelization
    double precision :: l2dist
    double precision, allocatable, dimension(:,:) :: atomic_distance

    ! Pre-computed terms in the full distance matrix
    double precision, allocatable, dimension(:) :: selfl21

    ! Pre-computed terms
    double precision, allocatable, dimension(:,:,:) :: ksi1

    double precision, allocatable, dimension(:,:,:,:,:) :: sinp1
    double precision, allocatable, dimension(:,:,:,:,:) :: cosp1

    ! Value of PI at full FORTRAN precision.
    double precision, parameter :: pi = 4.0d0 * atan(1.0d0)

    ! counter for periodic distance
    integer :: pj, pk
    integer :: pmax1
    integer :: nneighi
    double precision :: theta
    double precision :: sin_m, cos_m

    double precision :: ang_norm2

    double precision :: mol_dist

    double precision :: ksi3

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

    inv_sigma2(:) = -1.0d0 / (sigmas(:))**2

    allocate(ksi1(nm1, maxval(n1), maxval(nneigh1)))

    ksi1 = 0.0d0

    !$OMP PARALLEL DO PRIVATE(ni, nneighi)
    do a = 1, nm1
        ni = n1(a)
        do i = 1, ni
            nneighi = nneigh1(a, i)
            do j = 2, nneighi
                ksi1(a, i, j) = 1.0d0 / x1(a,i,1,j)**6
            enddo
        enddo
    enddo
    !$OMP END PARALLEL DO

    allocate(cosp1(nm1, maxval(n1), pmax1, order, maxval(nneigh1)))
    allocate(sinp1(nm1, maxval(n1), pmax1, order, maxval(nneigh1)))

    cosp1 = 0.0d0
    sinp1 = 0.0d0

    !$OMP PARALLEL DO PRIVATE(ni, nneighi, ksi3, pj, pk, theta, cos_m, sin_m) REDUCTION(+:cosp1,sinp1)
    do a = 1, nm1
        ni = n1(a)

        do i = 1, ni
            nneighi = nneigh1(a, i)

            do j = 2, nneighi
                do k = j+1, nneighi

                    ksi3 = calc_ksi3(X1(a,i,:,:), j, k)
                    theta = calc_angle(x1(a, i, 3:5, j), &
                        &  x1(a, i, 3:5, 1), x1(a, i, 3:5, k))

                    pj = int(x1(a,i,2,k))
                    pk = int(x1(a,i,2,j))

                    do m = 1, order

                        cos_m = (cos(m * theta) - cos((theta + pi) * m))*ksi3
                        sin_m = (sin(m * theta) - sin((theta + pi) * m))*ksi3

                        cosp1(a, i, pj, m, j) = cosp1(a, i, pj, m, j) + cos_m
                        sinp1(a, i, pj, m, j) = sinp1(a, i, pj, m, j) + sin_m

                        cosp1(a, i, pk, m, k) = cosp1(a, i, pk, m, k) + cos_m
                        sinp1(a, i, pk, m, k) = sinp1(a, i, pk, m, k) + sin_m

                    enddo
                enddo
            enddo
        enddo
    enddo
    !$OMP END PARALLEL DO

    allocate(selfl21(nm1))

    selfl21 = 0.0d0

    !$OMP PARALLEL DO PRIVATE(ni) REDUCTION(+:selfl21)
    do a = 1, nm1
        ni = n1(a)
        do i = 1, ni
            do j = 1, ni
            
            selfl21(a) = selfl21(a) + atomic_distl2(x1(a,i,:,:), x1(a,j,:,:), &
                & nneigh1(a,i), nneigh1(a,j), ksi1(a,i,:), ksi1(a,j,:), &
                & sinp1(a,i,:,:,:), sinp1(a,j,:,:,:), &
                & cosp1(a,i,:,:,:), cosp1(a,j,:,:,:), &
                & t_width, d_width, cut_distance, order, & 
                & pd, ang_norm2,distance_scale, angular_scale)
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

                    l2dist = atomic_distl2(x1(a,i,:,:), x1(b,j,:,:), &
                        & nneigh1(a,i), nneigh1(b,j), ksi1(a,i,:), ksi1(b,j,:), &
                        & sinp1(a,i,:,:,:), sinp1(b,j,:,:,:), &
                        & cosp1(a,i,:,:,:), cosp1(b,j,:,:,:), &
                        & t_width, d_width, cut_distance, order, &
                        & pd, ang_norm2, distance_scale, angular_scale)

                    atomic_distance(i,j) = l2dist

                enddo
            enddo

            mol_dist = selfl21(a) + selfl21(b) - 2.0d0 * sum(atomic_distance(:ni,:nj))

            do k = 1, nsigmas
                kernels(k, a, b) = exp(mol_dist * inv_sigma2(k))
                kernels(k, b, a) = kernels(k, a, b)
            enddo
        enddo
    enddo
    !$OMP END PARALLEL DO

    deallocate(atomic_distance)
    deallocate(selfl21)
    deallocate(ksi1)
    deallocate(cosp1)
    deallocate(sinp1)

end subroutine fget_global_symmetric_kernels_fchl


subroutine fget_global_kernels_fchl(x1, x2, n1, n2, nneigh1, nneigh2, &
       & sigmas, nm1, nm2, nsigmas, &
       & t_width, d_width, cut_distance, order, pd, &
       & distance_scale, angular_scale, kernels)

    use fchl_utils, only: atomic_distl2, calc_angle, calc_ksi3, calc_ksi4, calc_dihedral

    implicit none

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

    double precision, intent(in) :: t_width
    double precision, intent(in) :: d_width 
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
    integer :: i, j, k
    integer :: ni, nj
    integer :: a, b, m, n

    ! Temporary variables necessary for parallelization
    double precision :: l2dist
    double precision, allocatable, dimension(:,:) :: atomic_distance

    ! Pre-computed terms in the full distance matrix
    double precision, allocatable, dimension(:) :: selfl21
    double precision, allocatable, dimension(:) :: selfl22

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
    integer :: pj, pk
    integer :: pmax1
    integer :: pmax2
    integer :: nneighi
    double precision :: theta

    double precision :: ksi3
    double precision :: cos_m, sin_m
    double precision :: ang_norm2

    double precision :: mol_dist

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

    inv_sigma2(:) = -1.0d0 / (sigmas(:))**2

    allocate(ksi1(nm1, maxval(n1), maxval(nneigh1)))
    allocate(ksi2(nm2, maxval(n2), maxval(nneigh2)))

    ksi1 = 0.0d0
    ksi2 = 0.0d0

    !$OMP PARALLEL DO PRIVATE(ni, nneighi)
    do a = 1, nm1
        ni = n1(a)
        do i = 1, ni
            nneighi = nneigh1(a, i)
            do j = 2, nneighi
                ksi1(a, i, j) = 1.0d0 / x1(a, i, 1, j)**6
            enddo
        enddo
    enddo
    !$OMP END PARALLEL do

    !$OMP PARALLEL DO PRIVATE(ni, nneighi)
    do a = 1, nm2
        ni = n2(a)
        do i = 1, ni
            nneighi = nneigh2(a, i)
            do j = 2, nneighi
                ksi2(a, i, j) = 1.0d0 / x2(a, i, 1, j)**6
            enddo
        enddo
    enddo
    !$OMP END PARALLEL do


    allocate(cosp1(nm1, maxval(n1), pmax1, order, maxval(nneigh1)))
    allocate(sinp1(nm1, maxval(n1), pmax1, order, maxval(nneigh1)))

    cosp1 = 0.0d0
    sinp1 = 0.0d0

    !$OMP PARALLEL DO PRIVATE(ni, nneighi, ksi3, pj, pk, theta, sin_m, cos_m) REDUCTION(+:cosp1,sinp1)
    do a = 1, nm1
        ni = n1(a)

        do i = 1, ni
            nneighi = nneigh1(a, i)

            do j = 2, nneighi
                do k = j+1, nneighi

                    ksi3 = calc_ksi3(X1(a,i,:,:), j, k)
                    theta = calc_angle(x1(a, i, 3:5, j), &
                        &  x1(a, i, 3:5, 1), x1(a, i, 3:5, k))

                    pj =  int(x1(a,i,2,k))
                    pk =  int(x1(a,i,2,j))

                    do m = 1, order

                        cos_m = (cos(m * theta) - cos((theta + pi) * m))*ksi3
                        sin_m = (sin(m * theta) - sin((theta + pi) * m))*ksi3

                        cosp1(a, i, pj, m, j) = cosp1(a, i, pj, m, j) + cos_m
                        sinp1(a, i, pj, m, j) = sinp1(a, i, pj, m, j) + sin_m

                        cosp1(a, i, pk, m, k) = cosp1(a, i, pk, m, k) + cos_m
                        sinp1(a, i, pk, m, k) = sinp1(a, i, pk, m, k) + sin_m

                    enddo
                enddo
            enddo
        enddo
    enddo
    !$OMP END PARALLEL do

    allocate(cosp2(nm2, maxval(n2), pmax2, order, maxval(nneigh2)))
    allocate(sinp2(nm2, maxval(n2), pmax2, order, maxval(nneigh2)))

    cosp2 = 0.0d0
    sinp2 = 0.0d0

    !$OMP PARALLEL DO PRIVATE(ni, nneighi, ksi3, pj, pk, theta, cos_m, sin_m) REDUCTION(+:cosp2,sinp2)
    do a = 1, nm2
        ni = n2(a)

        do i = 1, ni
            nneighi = nneigh2(a, i)

            do j = 2, nneighi
                do k = j+1, nneighi

                    ksi3 = calc_ksi3(X2(a,i,:,:), j, k)
                    theta = calc_angle(x2(a, i, 3:5, j), &
                        &  x2(a, i, 3:5, 1), x2(a, i, 3:5, k))

                    pj =  int(x2(a,i,2,k))
                    pk =  int(x2(a,i,2,j))

                    do m = 1, order

                        cos_m = (cos(m * theta) - cos((theta + pi) * m))*ksi3
                        sin_m = (sin(m * theta) - sin((theta + pi) * m))*ksi3

                        cosp2(a, i, pj, m, j) = cosp2(a, i, pj, m, j) + cos_m
                        sinp2(a, i, pj, m, j) = sinp2(a, i, pj, m, j) + sin_m

                        cosp2(a, i, pk, m, k) = cosp2(a, i, pk, m, k) + cos_m
                        sinp2(a, i, pk, m, k) = sinp2(a, i, pk, m, k) + sin_m

                    enddo
                enddo
            enddo
        enddo
    enddo
    !$OMP END PARALLEL do

    allocate(selfl21(nm1))
    allocate(selfl22(nm2))

    selfl21 = 0.0d0
    selfl22 = 0.0d0

    !$OMP PARALLEL DO PRIVATE(ni) REDUCTION(+:selfl21)
    do a = 1, nm1
        ni = n1(a)
        do i = 1, ni
            do j= 1, ni
            
            selfl21(a) = selfl21(a) + atomic_distl2(x1(a,i,:,:), x1(a,j,:,:), &
                & nneigh1(a,i), nneigh1(a,j), ksi1(a,i,:), ksi1(a,j,:), &
                & sinp1(a,i,:,:,:), sinp1(a,j,:,:,:), &
                & cosp1(a,i,:,:,:), cosp1(a,j,:,:,:), &
                & t_width, d_width, cut_distance, order, & 
                & pd, ang_norm2,distance_scale, angular_scale)
            enddo
        enddo
    enddo
    !$OMP END PARALLEL DO

    !$OMP PARALLEL DO PRIVATE(ni) REDUCTION(+:selfl22)
    do a = 1, nm2
        ni = n2(a)
        do i = 1, ni
            do j= 1, ni
            selfl22(a) = selfl22(a) + atomic_distl2(x2(a,i,:,:), x2(a,j,:,:), &
                & nneigh2(a,i), nneigh2(a,j), ksi2(a,i,:), ksi2(a,j,:), &
                & sinp2(a,i,:,:,:), sinp2(a,j,:,:,:), &
                & cosp2(a,i,:,:,:), cosp2(a,j,:,:,:), &
                & t_width, d_width, cut_distance, order, &
                & pd, ang_norm2, distance_scale, angular_scale)
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

                    l2dist = atomic_distl2(x1(a,i,:,:), x2(b,j,:,:), & 
                        & nneigh1(a,i), nneigh2(b,j), ksi1(a,i,:), ksi2(b,j,:), &
                        & sinp1(a,i,:,:,:), sinp2(b,j,:,:,:), &
                        & cosp1(a,i,:,:,:), cosp2(b,j,:,:,:), &
                        & t_width, d_width, cut_distance, order, &
                        & pd, ang_norm2, distance_scale, angular_scale)

                    atomic_distance(i,j) = l2dist

                enddo
            enddo

            mol_dist = selfl21(a) + selfl22(b) - 2.0d0 * sum(atomic_distance(:ni,:nj))

            do k = 1, nsigmas
                kernels(k, a, b) = exp(mol_dist * inv_sigma2(k))
            enddo

        enddo
    enddo
    !$OMP END PARALLEL DO

    deallocate(atomic_distance)
    deallocate(selfl21)
    deallocate(selfl22)
    deallocate(ksi1)
    deallocate(ksi2)
    deallocate(cosp1)
    deallocate(cosp2)
    deallocate(sinp1)
    deallocate(sinp2)

end subroutine fget_global_kernels_fchl
