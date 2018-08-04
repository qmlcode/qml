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

module ffchl_module

    implicit none

contains

pure function cut_function(r, cut_start, cut_distance) result(f)

    implicit none

    ! Distance
    double precision, intent(in) :: r

    ! Lower limit of damping
    double precision, intent(in) :: cut_start

    ! Upper limit of damping
    double precision, intent(in) :: cut_distance

    ! Intermediate variables
    double precision :: x
    double precision :: rl
    double precision :: ru

    ! Damping function at distance r
    double precision :: f

    ru = cut_distance
    rl = cut_start * cut_distance

    if (r > ru) then

        f = 0.0d0

    else if (r < rl) then

        f = 1.0d0

    else

        x = (ru - r) / (ru - rl)
        f = (10.0d0 * x**3) - (15.0d0 * x**4) + (6.0d0 * x**5)

    endif

end function cut_function


pure function get_angular_norm2(t_width) result(ang_norm2)

    implicit none

    ! Theta-width
    double precision, intent(in) :: t_width

    ! The resulting angular norm (squared)
    double precision ang_norm2

    ! Integration limit - bigger than 100 should suffice.
    integer, parameter :: limit = 10000

    ! Pi at double precision.
    double precision, parameter :: pi = 4.0d0 * atan(1.0d0)

    integer :: n

    ang_norm2 = 0.0d0

    do n = -limit, limit
        ang_norm2 = ang_norm2 + exp(-((t_width * n)**2)) &
            & * (2.0d0 - 2.0d0 * cos(n * pi))
    end do

    ang_norm2 = sqrt(ang_norm2 * pi) * 2.0d0

end function

function get_twobody_weights(x, neighbors, power, cut_start, cut_distance, dim1) result(ksi)

    implicit none

    double precision, dimension(:,:), intent(in) :: x
    integer, intent(in) :: neighbors
    double precision, intent(in) :: power
    double precision, intent(in) :: cut_start
    double precision, intent(in) :: cut_distance
    integer, intent(in) :: dim1

    double precision, dimension(dim1) :: ksi
    integer :: i


    ksi = 0.0d0

    do i = 2, neighbors

        ksi(i) = cut_function(x(1, i), cut_start, cut_distance) / x(1, i)**power

    enddo

end function get_twobody_weights

! Calculate the Fourier terms for the FCHL three-body expansion
function get_threebody_fourier(x, neighbors, order, power, cut_start, cut_distance, &
    & dim1, dim2, dim3) result(fourier)

    implicit none

    ! Input representation, dimension=(5,n).
    double precision, dimension(:,:), intent(in) :: x

    ! Number of neighboring atoms to iterate over.
    integer, intent(in) :: neighbors

    ! Fourier-expansion order.
    integer, intent(in) :: order

    ! Power law
    double precision, intent(in) :: power

    ! Lower limit of damping function
    double precision, intent(in) :: cut_start

    ! Upper limit of damping function
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
        do k = j+1, neighbors

            ksi3 = calc_ksi3(X(:,:), j, k, power, cut_start, cut_distance)
            theta = calc_angle(x(3:5, j), x(3:5, 1), x(3:5, k))

            pj =  int(x(2,k))
            pk =  int(x(2,j))

            do m = 1, order

                cos_m = (cos(m * theta) - cos((theta + pi) * m))*ksi3
                sin_m = (sin(m * theta) - sin((theta + pi) * m))*ksi3

                fourier(1, pj, m, j) = fourier(1, pj, m, j) + cos_m
                fourier(2, pj, m, j) = fourier(2, pj, m, j) + sin_m

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


function calc_ksi3(X, j, k, power, cut_start, cut_distance) result(ksi3)

    implicit none

    double precision, dimension(:,:), intent(in) :: X

    integer, intent(in) :: j
    integer, intent(in) :: k
    double precision, intent(in) :: power
    double precision, intent(in) :: cut_start
    double precision, intent(in) :: cut_distance

    double precision :: cos_i, cos_j, cos_k
    double precision :: di, dj, dk

    double precision :: ksi3
    double precision :: cut

    cos_i = calc_cos_angle(x(3:5, k), x(3:5, 1), x(3:5, j))
    cos_j = calc_cos_angle(x(3:5, j), x(3:5, k), x(3:5, 1))
    cos_k = calc_cos_angle(x(3:5, 1), x(3:5, j), x(3:5, k))

    dk = x(1, j)
    dj = x(1, k)
    di = norm2(x(3:5, j) - x(3:5, k))


    cut = cut_function(dk, cut_start, cut_distance) * &
        & cut_function(dj, cut_start, cut_distance) * &
        & cut_function(di, cut_start, cut_distance)

    ksi3 = cut * (1.0d0 + 3.0d0 * cos_i*cos_j*cos_k) / (di * dj * dk)**power

end function calc_ksi3


pure function scalar(X1, X2, N1, N2, ksi1, ksi2, sin1, sin2, cos1, cos2, &
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

    double precision :: true_angular_scale
    double precision :: true_distance_scale
    
    double precision, intent(in):: ang_norm2

    double precision :: aadist

    logical, intent(in) :: alchemy

    ! We changed the convention for the scaling factors when the paper was under review
    ! so this is a quick fix
    true_angular_scale = angular_scale / sqrt(8.0d0)
    true_distance_scale = distance_scale / 16.0d0

    if (alchemy) then
        aadist = scalar_alchemy(X1, X2, N1, N2, ksi1, ksi2, sin1, sin2, cos1, cos2, &
            & t_width, d_width, order, pd, ang_norm2, &
            & true_distance_scale, true_angular_scale)
    else
        aadist = scalar_noalchemy(X1, X2, N1, N2, ksi1, ksi2, sin1, sin2, cos1, cos2, &
            & t_width, d_width, ang_norm2, &
            & true_distance_scale, true_angular_scale)
    endif


end function scalar

pure function scalar_noalchemy(X1, X2, N1, N2, ksi1, ksi2, sin1, sin2, cos1, cos2, &
    & t_width, d_width, ang_norm2, &
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
        do j = 2, n2

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
    & t_width, d_width, order, pd, ang_norm2, &
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
    ! double precision, intent(in) :: cut_distance
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


end module ffchl_module
