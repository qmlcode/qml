! MIT License
!
! Copyright (c) 2017 Anders Steen Christensen, Bing Huang
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

module slatm_utils

    implicit none

contains

function linspace(x0, x1, nx) result(xs)

    double precision, intent(in) :: x0
    double precision, intent(in) :: x1
    integer, intent(in) :: nx

    double precision, dimension(nx) :: xs

    integer :: i
    double precision :: step

    step = (x1 - x0) / (nx - 1)

    do i = 1, nx
        xs(i) = x0 + (i - 1) * step
    enddo

end function linspace

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

end module slatm_utils

subroutine fget_sbot(coordinates, nuclear_charges, z1, z2, z3, rcut, nx, dgrid, sigma, ys)

    use slatm_utils, only: linspace, calc_angle, calc_cos_angle

    implicit none

    double precision, dimension(:,:), intent(in) :: coordinates
    double precision, dimension(:), intent(in) :: nuclear_charges
    double precision, intent(in) :: rcut
    integer, intent(in) :: nx
    double precision, intent(in) :: dgrid
    double precision, intent(in) :: sigma
    double precision, dimension(nx), intent(out) :: ys
    
    ! MBtype
    integer, intent(in) :: z1, z2, z3

    ! Unique three-body atoms array
    integer, dimension(:,:), allocatable :: tas
    integer :: ntas

    ! 
    integer, dimension(:), allocatable :: ias1, ias2, ias3
    integer :: nias1, nias2, nias3
    
    integer :: ia1, ia2, ia3, itas

    double precision, allocatable, dimension(:, :) :: distance_matrix

    double precision :: norm

    integer :: i, j, k, idx
    integer :: natoms

    double precision, parameter :: eps = epsilon(0.0d0)
    double precision, parameter :: pi = 4.0d0 * atan(1.0d0)
    double precision :: d2r
    double precision :: coeff
    double precision :: c0
    double precision :: ang
    double precision :: cak
    double precision :: cai
    double precision :: prefactor
    double precision :: r
    double precision :: a0, a1
    double precision, dimension(nx) :: xs
    double precision, dimension(nx) :: cos_xs
    double precision :: inv_sigma

    natoms = size(coordinates, dim=1)
    if (size(coordinates, dim=1) /= size(nuclear_charges, dim=1)) then
        write(*,*) "ERROR: Coulomb matrix generation"
        write(*,*) size(coordinates, dim=1), "coordinates, but", &
            & size(nuclear_charges, dim=1), "atom_types!"
        stop
    endif

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

    allocate(ias1(natoms))
    allocate(ias2(natoms))
    allocate(ias3(natoms))
   
    ias1 = 0
    ias2 = 0
    ias3 = 0

    nias1 = 0
    do i = 1, natoms
        if (int(nuclear_charges(i)).eq.z1) then
            nias1 = nias1 + 1
            ias1(nias1) = i
        endif
    enddo

    nias2 = 0
    do i = 1, natoms
        if (int(nuclear_charges(i)).eq.z2) then
            nias2 = nias2 + 1
            ias2(nias2) = i
        endif
    enddo 

    nias3 = 0
    do i = 1, natoms
        if (int(nuclear_charges(i)).eq.z3) then
            nias3 = nias3 + 1
            ias3(nias3) = i
        endif
    enddo 
    
    allocate(tas(3, nias1*nias2*nias3))

    ntas = 0
    do ia1 = 1, nias1
        do ia2 = 1, nias2
            if (.not. ((distance_matrix(ias1(ia1),ias2(ia2)) > eps) .and. &
                     & (distance_matrix(ias1(ia1),ias2(ia2)) <= rcut))) cycle
            do ia3 = 1, nias3
                if ((z1 == z3) .and. (ias1(ia1) < ias3(ia3))) cycle
                if (.not. ((distance_matrix(ias1(ia1),ias3(ia3)) > eps) .and. &
                         & (distance_matrix(ias1(ia1),ias3(ia3)) <= rcut))) cycle
                if (.not. ((distance_matrix(ias2(ia2),ias3(ia3)) > eps) .and. &
                         & (distance_matrix(ias2(ia2),ias3(ia3)) <= rcut))) cycle
                ntas = ntas + 1
                tas(1, ntas) = ias1(ia1)
                tas(2, ntas) = ias2(ia2)
                tas(3, ntas) = ias3(ia3)
            enddo
        enddo
    enddo

    d2r = pi/180.0d0
    a0 = -20.0d0 * d2r
    a1 = pi + 20.0d0 * d2r

    xs = linspace(a0, a1, nx)

    prefactor = 1.0d0 / 3.0d0

    coeff = 1.0d0 / sqrt(2*sigma**2*pi)
    c0 = prefactor * (mod(z1,1000)*mod(z2,1000)*mod(z3,1000)) * coeff


    ys = 0.0d0
    inv_sigma = 1.0d0 / (2*sigma**2)

    !$OMP PARALLEL DO
    do i = 1, nx
        cos_xs(i) = cos(xs(i))
    enddo
    !$OMP END PARALLEL do

    
    !$OMP PARALLEL DO PRIVATE(i,j,k,ang,cai,cak,r) REDUCTION(+:ys)
    do itas = 1, ntas

        i = tas(1, itas)
        j = tas(2, itas)
        k = tas(3, itas)

        ang = calc_angle(coordinates(i, :), coordinates(j, :), coordinates(k, :))
        cak = calc_cos_angle(coordinates(i, :), coordinates(k, :), coordinates(j, :))
        cai = calc_cos_angle(coordinates(k, :), coordinates(i, :), coordinates(j, :))

        r = distance_matrix(i,j) * distance_matrix(i,k) * distance_matrix(k,j)

        ys = ys + c0 *( (1.0d0 + cos_xs*cak*cai)/(r**3 ) ) * ( exp(-(xs-ang)**2 * inv_sigma) )

    enddo
    !$OMP END PARALLEL do

    ys = ys * dgrid

    deallocate(tas)
    deallocate(ias1)
    deallocate(ias2)
    deallocate(ias3)
    deallocate(distance_matrix)
    
end subroutine fget_sbot

