! MIT License
!
! Copyright (c) 2016 Anders Steen Christensen
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

module funcs

contains

function stoch_dist(D1, D2, W1, W2) result(a)

    implicit none

    double precision, intent(in) :: D1
    double precision, intent(in) :: D2
    double precision, intent(in) :: W1
    double precision, intent(in) :: W2

    double precision :: a

    a = W1**2/(W1**2 + D1**2) * W2**2/(W2**2 + D2**2)

end function stoch_dist


function dist(x1, x2, w1, w2) result(a)

    implicit none

    double precision, intent(in) :: x1
    double precision, intent(in) :: x2
    double precision, intent(in) :: w1
    double precision, intent(in) :: w2

    double precision, parameter :: pi = 4.0d0 * atan(1.0d0)

    double precision :: a

    a = exp(-((x1-x2)**2)/(4.0d0*w1**2)) *  (1.0d0 - sin(pi * x1/(2.0d0 * w2))) * &
        & (1.0d0 - sin(pi * x2/(2.0d0 * w2)))

end function dist


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





function m_dist(X1, X2, N1, N2, width, cut_distance, r_width, c_width) result(aadist)

    implicit none

    double precision, dimension(:,:), intent(in) :: X1
    double precision, dimension(:,:), intent(in) :: X2

    integer, intent(in) :: N1
    integer, intent(in) :: N2

    double precision, intent(in) :: width
    double precision, intent(in) :: cut_distance
    double precision, intent(in) :: r_width
    double precision, intent(in) :: c_width

    double precision :: aadist

    double precision :: maxgausdist
    double precision :: d

    double precision :: r_dist
    double precision :: c_dist

    integer :: m_1, m_2

    maxgausdist = 8.0d0*width
    aadist = 0.0d0

    do m_1 = 1, N1

        if (X1(1, m_1) > cut_distance) exit

        do m_2 = 1, N2

            if (X2(1, m_2) > cut_distance) exit

            if (abs(X2(1,m_2) - X1(1,m_1)) < maxgausdist) then

                r_dist = abs(x1(2,m_1) - x2(2,m_2))
                c_dist = abs(x1(3,m_1) - x2(3,m_2))

                d = dist(x1(1,m_1), x2(1,m_2),width,cut_distance)
                d = d * stoch_dist(r_dist,c_dist,r_width,c_width)
                aadist = aadist + d * (1.0d0 + x1(4,m_1)*x2(4,m_2) + x1(5,m_1)*x2(5,m_2))

            end if
        end do
    end do

end function m_dist

function distl2(X1, X2, Z1, Z2, N1, N2, width, &
    & cut_distance, r_width, c_width, sin1, sin2) result(D12)

    implicit none

    double precision, dimension(:,:,:), intent(in) :: X1
    double precision, dimension(:,:,:), intent(in) :: X2

    integer, dimension(:,:), intent(in) :: Z1
    integer, dimension(:,:), intent(in) :: Z2

    integer, intent(in) :: N1
    integer, intent(in) :: N2

    double precision, intent(in) :: width
    double precision, intent(in) :: cut_distance
    double precision, intent(in) :: r_width
    double precision, intent(in) :: c_width

    double precision, dimension(:,:), intent(in) :: sin1, sin2

    double precision :: D12

    integer :: j_1, j_2

    integer :: m_1, m_2

    double precision :: r_dist, c_dist, aadist, d, maxgausdist2

    double precision :: inv_cut, inv_width
    double precision :: c_width2, r_width2, r2

    double precision, parameter :: pi = 4.0d0 * atan(1.0d0)

    double precision :: killer

    inv_cut = pi / (2.0d0 * cut_distance)
    inv_width = -1.0d0 / (4.0d0 * width**2)
    maxgausdist2 = (8.0d0 * width)**2
    r_width2 = r_width**2
    c_width2 = c_width**2

    D12 = 0.0d0

    do j_1 = 1, N1

!$OMP PARALLEL DO PRIVATE(aadist, r2, d, killer) REDUCTION(+:D12)
        do j_2 = 1, N2

            aadist = 0.0d0

            do m_1 = 1, N1

                killer = 1.0d0
                if (X1(j_1, 1,m_1) > cut_distance) killer = 0.0d0 !exit

                do m_2 = 1, N2

                    if (X2(j_2,1, m_2) > cut_distance) killer = 0.0d0 !exit

                    r2 =  (x1(j_1,1,m_1)-x2(j_2,1,m_2))**2

                    if (r2 < maxgausdist2) then

                        d = exp(r2 * inv_width )  * sin1(j_1, m_1) * sin2(j_2, m_2)

                        d = d * (r_width2/(r_width2 + (x1(j_1,2,m_1) - x2(j_2,2,m_2))**2) * &
                            & c_width2/(c_width2 + (x1(j_1,3,m_1) - x2(j_2,3,m_2))**2))

                        aadist = aadist + d * (1.0d0 + x1(j_1,4,m_1)*x2(j_2,4,m_2) + &
                            & x1(j_1,5,m_1)*x2(j_2,5,m_2)) * killer

                    end if
                end do
            end do

            r_dist = abs(z1(j_1,1) - z2(j_2,1))
            c_dist = abs(z1(j_1,2) - z2(j_2,2))

            D12 = D12 + aadist * (r_width2/(r_width2 + r_dist**2) * c_width2/(c_width2 + c_dist**2))

        enddo
!$OMP END PARALLEL DO
    enddo

end function distl2

end module funcs

subroutine molecular_arad_l2_distance_all(X1, X2, Z1, Z2, N1, N2, nmol1, nmol2, width, &
    & cut_distance, r_width, c_width, D12)

    use funcs, only: distl2, m_dist, stoch_dist

    implicit none

    double precision, dimension(:,:,:,:), intent(in) :: X1
    double precision, dimension(:,:,:,:), intent(in) :: X2

    integer, dimension(:,:,:), intent(in) :: Z1
    integer, dimension(:,:,:), intent(in) :: Z2

    integer, dimension(:), intent(in) :: N1
    integer, dimension(:), intent(in) :: N2

    integer, intent(in) :: nmol1
    integer, intent(in) :: nmol2

    double precision, intent(in) :: width
    double precision, intent(in) :: cut_distance
    double precision, intent(in) :: r_width
    double precision, intent(in) :: c_width

    double precision, dimension(nmol1, nmol2), intent(out) :: D12

    double precision, dimension(nmol1) :: D11
    double precision, dimension(nmol2) :: D22

    double precision, allocatable, dimension(:,:,:) :: sin1, sin2
    double precision :: inv_cut

    integer :: i, j, j_1, m_1

    double precision, parameter :: pi = 4.0d0 * atan(1.0d0)

    double precision, parameter :: eps = 1.0e-11

    inv_cut = pi / (2.0d0 * cut_distance)

    allocate(sin1(nmol1, maxval(N1), maxval(N1)))
    allocate(sin2(nmol2, maxval(N2), maxval(N2)))

    !$OMP PARALLEL DO
    do i = 1, nmol1
        do m_1 = 1, N1(i)
            do j_1 = 1, N1(i)
            sin1(i, j_1, m_1) = 1.0d0 - sin(x1(i,j_1,1,m_1) * inv_cut)
            enddo
        enddo
    enddo
    !$OMP END PARALLEL DO

    !$OMP PARALLEL DO
    do i = 1, nmol2
        do m_1 = 1, N2(i)
            do j_1 = 1, N2(i)
            sin2(i, j_1, m_1) = 1.0d0 - sin(x2(i,j_1,1,m_1) * inv_cut)
            enddo
        enddo
    enddo
    !$OMP END PARALLEL DO


    !$OMP PARALLEL DO
    do i = 1, nmol1
        D11(i) = distl2(X1(i,:,:,:), X1(i,:,:,:), Z1(i,:,:), Z1(i,:,:), N1(i), N1(i), &
            & width, cut_distance, r_width, c_width, sin1(i,:,:), sin1(i,:,:))
    enddo
    !$OMP END PARALLEL DO

    !$OMP PARALLEL DO
    do i = 1, nmol2
        D22(i) = distl2(X2(i,:,:,:), X2(i,:,:,:), Z2(i,:,:), Z2(i,:,:), N2(i), N2(i), &
            & width, cut_distance, r_width, c_width, sin2(i,:,:), sin2(i,:,:))
    enddo
    !$OMP END PARALLEL DO

    !$OMP PARALLEL DO
    do j = 1, nmol2
        do i = 1, nmol1
            D12(i,j) = distl2(X1(i,:,:,:), X2(j,:,:,:), Z1(i,:,:), Z2(j,:,:), N1(i), N2(j), &
                & width, cut_distance, r_width, c_width, sin1(i,:,:), sin2(j,:,:))

            D12(i,j) = D11(i) + D22(j) - 2.0d0 * D12(i,j)

            if (abs(D12(i,j)) < eps) D12(i,j) = 0.0d0

        enddo
    enddo
    !$OMP END PARALLEL DO

    deallocate(sin1)
    deallocate(sin2)

end subroutine molecular_arad_l2_distance_all

subroutine molecular_arad_l2_distance(X1, X2, Z1, Z2, N1, N2, width, &
    & cut_distance, r_width, c_width, distance)

    use funcs, only: m_dist, stoch_dist

    implicit none

    double precision, dimension(:,:,:), intent(in) :: X1
    double precision, dimension(:,:,:), intent(in) :: X2

    integer, dimension(:,:), intent(in) :: Z1
    integer, dimension(:,:), intent(in) :: Z2

    integer, intent(in) :: N1
    integer, intent(in) :: N2

    double precision, intent(in) :: width
    double precision, intent(in) :: cut_distance
    double precision, intent(in) :: r_width
    double precision, intent(in) :: c_width

    double precision, intent(out) :: distance

    integer :: j_1, j_2
    double precision :: D11, D22, D12

    double precision :: dd

    double precision :: rdist
    double precision :: cdist

    D11 = 0.0d0
    write (*,*) "N1:", N1
    write (*,*) "N2:", N2

    do j_1 = 1, N1
        write(*,*) "Z1:", "i", j_1, Z1(j_1,1), Z1(j_1,2)
    enddo


    do j_1 = 1, N1
        do j_2 = 1, N1

            rdist = abs(z1(j_1,1) - z1(j_2,1))
            CDist = abs(Z1(j_1,2) - Z1(j_2,2))

            dd = M_Dist(X1(j_1,:,:),X1(j_2,:,:),n1,n1,width,cut_distance,R_Width,C_width)
            D11 = D11 + dd * stoch_dist(RDist,CDist,R_Width,C_width)

        enddo
    enddo

    write(*,*) "D11:", "i", D11
    D22 = 0.0d0

    do j_1 = 1, N2
        do j_2 = 1, N2

            rdist = abs(z2(j_1,1) - z2(j_2,1))
            CDist = abs(Z2(j_1,2) - Z2(j_2,2))

            dd = M_Dist(X2(j_1,:,:),X2(j_2,:,:),n2,n2,width,cut_distance,R_Width,C_width)
            D22 = D22 + dd * stoch_dist(RDist,CDist,R_Width,C_width)

        enddo
    enddo

    D12 = 0.0d0

    do j_1 = 1, N1
        do j_2 = 1, N2

            rdist = abs(z1(j_1,1) - z2(j_2,1))
            CDist = abs(Z1(j_1,2) - Z2(j_2,2))

            dd = M_Dist(X1(j_1,:,:),X2(j_2,:,:),n1,n2,width,cut_distance,R_Width,C_width)
            D12 = D12 + dd * stoch_dist(RDist,CDist,R_Width,C_width)

        enddo
    enddo

    distance = D11 + D22 - 2.0d0 * D12

end subroutine molecular_arad_l2_distance


subroutine atomic_arad_l2_distance_all(q1, q2, z1, z2, n1, n2, nm1, nm2, width, &
    & cut_distance, r_width, c_width, amax, dmatrix)

    use funcs, only: m_dist, stoch_dist, atomic_distl2

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

    ! Number of molecules
    integer, intent(in) :: nm1
    integer, intent(in) :: nm2
    
    ! Max number of atoms in one descriptor
    integer, intent(in) :: amax

    ! ARAD parameters
    double precision, intent(in) :: width
    double precision, intent(in) :: cut_distance
    double precision, intent(in) :: r_width
    double precision, intent(in) :: c_width

    ! Resulting distance matrix
    double precision, dimension(nm1,nm2,amax,amax), intent(out) :: dmatrix

    ! Pre-computed sine terms
    double precision, allocatable, dimension(:,:,:) :: sin1
    double precision, allocatable, dimension(:,:,:) :: sin2

    ! Pre-computed terms in the full distance matrix
    double precision, allocatable, dimension(:,:) :: selfl21
    double precision, allocatable, dimension(:,:) :: selfl22

    ! Internal counters
    integer :: i, j, k, ni, nj
    integer :: m_1, i_1, j_1

    ! Pre-computed constants
    double precision :: r_width2
    double precision :: c_width2
    double precision :: inv_cut
    double precision :: l2dist

    ! Value of PI at full FORTRAN precision.
    double precision, parameter :: pi = 4.0d0 * atan(1.0d0)

    ! Small number // to test for numerical stability
    double precision, parameter :: eps = 5.0e-12


    r_width2 = r_width**2
    c_width2 = c_width**2

    inv_cut = pi / (2.0d0 * cut_distance)

    allocate(sin1(nm1, maxval(n1), maxval(n1)))
    allocate(sin2(nm2, maxval(n2), maxval(n2)))

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

    !$OMP PARALLEL DO PRIVATE(ni)
    do i = 1, nm2
        ni = n2(i)
        do m_1 = 1, ni
            do i_1 = 1, ni
                sin2(i, i_1, m_1) = 1.0d0 - sin(q2(i,i_1,1,m_1) * inv_cut)
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

    dmatrix = 0.0d0

    !$OMP PARALLEL DO PRIVATE(l2dist,ni,nj)
    do j = 1, nm2
        nj = n2(j)
        do i = 1, nm1
            ni = n1(i)

            ! atomic_distance(:,:) = 0.0d0

            do i_1 = 1, ni
                do j_1 = 1, nj

                    l2dist = atomic_distl2(q1(i,i_1,:,:), q2(j,j_1,:,:), n1(i), n2(j), &
                        & sin1(i,i_1,:), sin2(j,j_1,:), width, cut_distance, r_width, c_width)

                    l2dist = selfl21(i,i_1) + selfl22(j,j_1) - 2.0d0 * l2dist &
                        & * (r_width2/(r_width2 + (z1(i,i_1,1) - z2(j,j_1,1))**2) * &
                        & c_width2/(c_width2 + (z1(i,i_1,2) - z2(j,j_1,2))**2))

                    if (abs(l2dist) < eps) l2dist = 0.0d0

                    dmatrix(i,j,i_1,j_1) = l2dist

                enddo
            enddo

        enddo
    enddo
    !$OMP END PARALLEL DO

    deallocate(selfl21)
    deallocate(selfl22)
    deallocate(sin1)
    deallocate(sin2)
end subroutine atomic_arad_l2_distance_all
