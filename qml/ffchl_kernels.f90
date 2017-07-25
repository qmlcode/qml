module fchl_utils

    implicit none

contains

! Calculate the Fourier terms for the FCHL three-body expansion
function get_threebody_fourier(x, neighbors, order, cut_distance, dim1, dim2, dim3) result(fourier)

    implicit none

    ! Input representation, dimension=(5,n).
    double precision, intent(in), dimension(:,:) :: x

    ! Number of neighboring atoms to iterate over.
    integer, intent(in) :: neighbors

    ! Fourier-expansion order.
    integer, intent(in) :: order

    ! Cut-off distance.
    double precision, intent(in) :: cut_distance

    ! Dimensions or the output array.
    integer, intent(in) :: dim1, dim2, dim3 

    ! dim(1,:,:,:) are cos terms, dim(2,:,:,:) are sine terms.
    double precision, dimension(2,dim1,dim2,dim3) :: fourier
  
    ! Pi at double precision. 
    double precision, parameter :: pi = 4.0d0 * atan(1.0d0)
    
    ! Internal counters.
    integer :: j, k, m

    ! Indexes for the periodic-table distance matrix.
    integer :: pj, pk

    ! Angle between atoms for the three-body term.
    double precision :: theta

    ! Three-body weight
    double precision :: ksi3

    ! Temporary variables for cos and sine Fourier terms.
    double precision :: cos_m, sin_m

    fourier = 0.0d0

    do j = 2, neighbors
        if (x(1, j) > cut_distance) exit

        do k = j+1, neighbors
            if (x(1, k) > cut_distance) exit

            ksi3 = calc_ksi3(X(:,:), j, k)
            theta = calc_angle(x(3:5, j), x(3:5, 1), x(3:5, k))

            pj =  int(x(2,k))
            pk =  int(x(2,j))

            do m = 1, order

                cos_m = (cos(m * theta) - cos((theta + pi) * m))*ksi3
                sin_m = (sin(m * theta) - sin((theta + pi) * m))*ksi3

                ! cosp1(pj, m, j) = cosp1(pj, m, j) + cos_m
                ! sinp1(pj, m, j) = sinp1(pj, m, j) + sin_m
                fourier(1, pj, m, j) = fourier(1, pj, m, j) + cos_m
                fourier(2, pj, m, j) = fourier(2, pj, m, j) + sin_m

                ! cosp1(pk, m, k) = cosp1(pk, m, k) + cos_m
                ! sinp1(pk, m, k) = sinp1(pk, m, k) + sin_m
                fourier(1, pk, m, k) = fourier(1, pk, m, k) + cos_m
                fourier(2, pk, m, k) = fourier(2, pk, m, k) + sin_m

            enddo
        enddo
    enddo

    return

end function get_threebody_fourier



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


function scalar(X1, X2, N1, N2, ksi1, ksi2, sin1, sin2, cos1, cos2, &
    & t_width, d_width, cut_distance, order, pd, ang_norm2, &
    & distance_scale, angular_scale, alchemy) result(aadist)

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

    double precision, intent(in):: ang_norm2

    double precision :: aadist

    logical, intent(in) :: alchemy

    if (alchemy) then
        aadist = scalar_alchemy(X1, X2, N1, N2, ksi1, ksi2, sin1, sin2, cos1, cos2, &
            & t_width, d_width, cut_distance, order, pd, ang_norm2, &
            & distance_scale, angular_scale)
    else
        aadist = scalar_noalchemy(X1, X2, N1, N2, ksi1, ksi2, sin1, sin2, cos1, cos2, &
            & t_width, d_width, cut_distance, order, ang_norm2, &
            & distance_scale, angular_scale)
    endif


end function scalar

function scalar_noalchemy(X1, X2, N1, N2, ksi1, ksi2, sin1, sin2, cos1, cos2, &
    & t_width, d_width, cut_distance, order, ang_norm2, &
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
    ! double precision, dimension(:,:), intent(in) :: pd
    double precision, intent(in) :: angular_scale
    double precision, intent(in) :: distance_scale

    double precision :: aadist

    double precision :: d
    
    integer :: i, j, p
    double precision :: angular 
    double precision :: maxgausdist2

    integer :: pmax1
    integer :: pmax2
    integer :: pmax

    double precision :: inv_width
    double precision :: r2

    double precision :: s
    
    double precision, parameter :: pi = 4.0d0 * atan(1.0d0)

    double precision :: g1 

    logical, allocatable, dimension(:) :: mask1
    logical, allocatable, dimension(:) :: mask2

    double precision, intent(in):: ang_norm2

    ! cached sine and cos terms
    double precision, allocatable, dimension(:,:) :: cos1c, cos2c, sin1c, sin2c

    if (int(x1(2,1)) /= int(x2(2,1))) then
        aadist = 0.0d0
        return
    endif

    pmax1 = int(maxval(x1(2,:n1)))
    pmax2 = int(maxval(x2(2,:n2)))

    pmax = min(pmax1,pmax2)

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
    
    allocate(cos1c(pmax,n1))
    allocate(cos2c(pmax,n2))
    allocate(sin1c(pmax,n1))
    allocate(sin2c(pmax,n2))

    cos1c = 0.0d0
    cos2c = 0.0d0
    sin1c = 0.0d0
    sin2c = 0.0d0

    p = 0

    do i = 1, pmax
        if (mask1(i)) cycle
        if (mask2(i)) cycle

        p = p + 1

        cos1c(p,:n1) = cos1(i,1,:n1)
        cos2c(p,:n2) = cos2(i,1,:n2)
        sin1c(p,:n1) = sin1(i,1,:n1)
        sin2c(p,:n2) = sin2(i,1,:n2)

    enddo

    pmax = p

    ! Pre-computed constants
    g1 = sqrt(2.0d0 * pi)/ang_norm2
    s = g1 * exp(-(t_width)**2 / 2.0d0)
    inv_width = -1.0d0 / (4.0d0 * d_width**2)
    maxgausdist2 = (8.0d0 * d_width)**2

    ! Initialize scalar product
    aadist = 1.0d0

    do i = 2, n1

        ! if (x1(1, i) > cut_distance) exit

        do j = 2, n2

            ! if (x2(1, j) > cut_distance) exit
            if (int(x1(2,i)) /= int(x2(2,j))) cycle

            r2 = (x2(1,j) - x1(1,i))**2
            if (r2 >= maxgausdist2) cycle

            d = exp(r2 * inv_width)

            angular = (sum(cos1c(:pmax,i) * cos2c(:pmax,j)) &
                   & + sum(sin1c(:pmax,i) * sin2c(:pmax,j))) * s

            aadist = aadist + d * (ksi1(i) * ksi2(j) * distance_scale &
                & + angular * angular_scale)

        end do
    end do

    deallocate(mask1)
    deallocate(mask2)
    deallocate(cos1c)
    deallocate(cos2c)
    deallocate(sin1c)
    deallocate(sin2c)


end function scalar_noalchemy

pure function scalar_alchemy(X1, X2, N1, N2, ksi1, ksi2, sin1, sin2, cos1, cos2, &
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

        ! if (X1(1, m_1) > cut_distance) exit

        do m_2 = 2, N2

            ! if (X2(1, m_2) > cut_distance) exit

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
end function scalar_alchemy


end module fchl_utils


subroutine fget_kernels_fchl(x1, x2, n1, n2, nneigh1, nneigh2, &
       & sigmas, nm1, nm2, nsigmas, &
       & t_width, d_width, cut_distance, order, pd, &
       & distance_scale, angular_scale, alchemy, kernels)

    use fchl_utils, only: scalar, get_threebody_fourier

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
    integer :: nneighi
    
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
   
    !$OMP PARALLEL DO PRIVATE(ni, fourier) schedule(dynamic) 
    do a = 1, nm1
        ni = n1(a)
        do i = 1, ni

            fourier = get_threebody_fourier(x1(a,i,:,:), & 
                & nneigh1(a, i), order, cut_distance, pmax1, order, maxval(nneigh1))

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
                & nneigh2(a, i), order, cut_distance, pmax2, order, maxval(nneigh2))

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
       & t_width, d_width, cut_distance, order, pd, &
       & distance_scale, angular_scale, alchemy, kernels)

    use fchl_utils, only: scalar, get_threebody_fourier

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

    double precision, intent(in) :: t_width
    double precision, intent(in) :: d_width 
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
    integer :: nneighi

    double precision :: ang_norm2

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

    !$OMP PARALLEL DO PRIVATE(ni, fourier)
    do a = 1, nm1
        ni = n1(a)
        do i = 1, ni

            fourier = get_threebody_fourier(x1(a,i,:,:), & 
                & nneigh1(a, i), order, cut_distance, pmax1, order, maxval(nneigh1))

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
       & t_width, d_width, cut_distance, order, pd, &
       & distance_scale, angular_scale, alchemy, kernels)

    use fchl_utils, only: scalar, get_threebody_fourier

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

    double precision, intent(in) :: t_width
    double precision, intent(in) :: d_width 
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
    integer :: nneighi

    double precision :: ang_norm2

    double precision :: mol_dist

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

    !$OMP PARALLEL DO PRIVATE(ni, fourier)
    do a = 1, nm1
        ni = n1(a)
        do i = 1, ni

            fourier = get_threebody_fourier(x1(a,i,:,:), & 
                & nneigh1(a, i), order, cut_distance, pmax1, order, maxval(nneigh1))

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
       & t_width, d_width, cut_distance, order, pd, &
       & distance_scale, angular_scale, alchemy, kernels)

    use fchl_utils, only: scalar, get_threebody_fourier

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

    double precision, intent(in) :: t_width
    double precision, intent(in) :: d_width 
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
    ! integer :: pj, pk
    integer :: pmax1
    integer :: pmax2
    integer :: nneighi
    ! double precision :: theta

    ! double precision :: ksi3
    ! double precision :: cos_m, sin_m
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

    !$OMP PARALLEL DO PRIVATE(ni, fourier)
    do a = 1, nm1
        ni = n1(a)
        do i = 1, ni

            fourier = get_threebody_fourier(x1(a,i,:,:), & 
                & nneigh1(a, i), order, cut_distance, pmax1, order, maxval(nneigh1))

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
                & nneigh2(a, i), order, cut_distance, pmax2, order, maxval(nneigh2))

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
       & t_width, d_width, cut_distance, order, pd, &
       & distance_scale, angular_scale, alchemy, kernels)

    use fchl_utils, only: scalar, get_threebody_fourier

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

    double precision, intent(in) :: t_width
    double precision, intent(in) :: d_width 
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
    integer :: i, j, k
    integer :: ni, nj
    integer :: a, b, n

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
    ! integer :: pj, pk
    integer :: pmax1
    integer :: pmax2
    integer :: nneighi
    ! double precision :: theta

    ! double precision :: ksi3
    ! double precision :: cos_m, sin_m
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

    do a = 1, na1
        pmax1 = max(pmax1, int(maxval(x1(a,2,:nneigh1(a)))))
    enddo
    do a = 1, na2
        pmax2 = max(pmax2, int(maxval(x2(a,2,:nneigh2(a)))))
    enddo

    inv_sigma2(:) = -1.0d0 / (sigmas(:))**2

    allocate(ksi1(na1, maxval(nneigh1)))
    allocate(ksi2(na2, maxval(nneigh2)))

    ksi1 = 0.0d0
    ksi2 = 0.0d0

    !$OMP PARALLEL DO PRIVATE(nneighi)
    do i = 1, na1
        nneighi = nneigh1(i)
        do j = 2, nneighi
            ksi1(i, j) = 1.0d0 / x1(i, 1, j)**6
        enddo
    enddo
    !$OMP END PARALLEL do
    
    !$OMP PARALLEL DO PRIVATE(nneighi)
    do i = 1, na2
        nneighi = nneigh2(i)
        do j = 2, nneighi
            ksi2(i, j) = 1.0d0 / x2(i, 1, j)**6
        enddo
    enddo
    !$OMP END PARALLEL do

    allocate(cosp1(na1, pmax1, order, maxval(nneigh1)))
    allocate(sinp1(na1, pmax1, order, maxval(nneigh1)))

    cosp1 = 0.0d0
    sinp1 = 0.0d0

    !$OMP PARALLEL DO PRIVATE(fourier)
    do i = 1, na1

        fourier = get_threebody_fourier(x1(i,:,:), & 
            & nneigh1(i), order, cut_distance, pmax1, order, maxval(nneigh1))

        cosp1(i,:,:,:) = fourier(1,:,:,:)
        sinp1(i,:,:,:) = fourier(2,:,:,:)

    enddo
    !$OMP END PARALLEL DO

    allocate(cosp2(na2, pmax2, order, maxval(nneigh2)))
    allocate(sinp2(na2, pmax2, order, maxval(nneigh2)))

    cosp2 = 0.0d0
    sinp2 = 0.0d0

    !$OMP PARALLEL DO PRIVATE(fourier)
    do i = 1, na2

        fourier = get_threebody_fourier(x2(i,:,:), & 
            & nneigh2(i), order, cut_distance, pmax2, order, maxval(nneigh2))

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
       & t_width, d_width, cut_distance, order, pd, &
       & distance_scale, angular_scale, alchemy, kernels)

    use fchl_utils, only: scalar, get_threebody_fourier

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

    double precision, intent(in) :: t_width
    double precision, intent(in) :: d_width 
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
    integer :: i, j, k
    integer :: ni, nj
    integer :: a, b, n

    ! Temporary variables necessary for parallelization
    double precision :: l2dist

    ! Pre-computed terms in the full distance matrix
    double precision, allocatable, dimension(:) :: self_scalar1

    ! Pre-computed terms
    double precision, allocatable, dimension(:,:) :: ksi1
    double precision, allocatable, dimension(:,:) :: ksi2

    double precision, allocatable, dimension(:,:,:,:) :: sinp1
    double precision, allocatable, dimension(:,:,:,:) :: cosp1

    ! Value of PI at full FORTRAN precision.
    double precision, parameter :: pi = 4.0d0 * atan(1.0d0)

    ! counter for periodic distance
    ! integer :: pj, pk
    integer :: pmax1
    integer :: nneighi
    ! double precision :: theta

    ! double precision :: ksi3
    ! double precision :: cos_m, sin_m
    double precision :: ang_norm2

    double precision :: mol_dist

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

    inv_sigma2(:) = -1.0d0 / (sigmas(:))**2

    allocate(ksi1(na1, maxval(nneigh1)))

    ksi1 = 0.0d0

    !$OMP PARALLEL DO PRIVATE(nneighi)
    do i = 1, na1
        nneighi = nneigh1(i)
        do j = 2, nneighi
            ksi1(i, j) = 1.0d0 / x1(i, 1, j)**6
        enddo
    enddo
    !$OMP END PARALLEL do
    
    allocate(cosp1(na1, pmax1, order, maxval(nneigh1)))
    allocate(sinp1(na1, pmax1, order, maxval(nneigh1)))

    cosp1 = 0.0d0
    sinp1 = 0.0d0

    !$OMP PARALLEL DO PRIVATE(fourier)
    do i = 1, na1

        fourier = get_threebody_fourier(x1(i,:,:), & 
            & nneigh1(i), order, cut_distance, pmax1, order, maxval(nneigh1))

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
