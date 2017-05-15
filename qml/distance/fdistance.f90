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

subroutine fmanhattan_distance(A, B, D)
    
    implicit none

    double precision, dimension(:,:), intent(in) :: A
    double precision, dimension(:,:), intent(in) :: B
    double precision, dimension(:,:), intent(inout) :: D

    integer :: na, nb 
    integer :: i, j

    na = size(A, dim=2)
    nb = size(B, dim=2)

!$OMP PARALLEL DO
    do i = 1, nb
        do j = 1, na
            D(j,i) = sum(abs(a(:,j) - b(:,i)))
        enddo
    enddo
!$OMP END PARALLEL DO


end subroutine fmanhattan_distance


subroutine fl2_distance(A, B, D)

    implicit none

    double precision, dimension(:,:), intent(in) :: A
    double precision, dimension(:,:), intent(in) :: B
    double precision, dimension(:,:), intent(inout) :: D

    integer :: na, nb, nv
    integer :: i, j

    double precision, allocatable, dimension(:) :: temp

    nv = size(A, dim=1)

    na = size(A, dim=2)
    nb = size(B, dim=2)

    allocate(temp(nv))

!$OMP PARALLEL DO PRIVATE(temp)
    do i = 1, nb
        do j = 1, na
            temp(:) = A(:,j) - B(:,i)
            D(j,i) = sqrt(sum(temp*temp))
        enddo
    enddo
!$OMP END PARALLEL DO

    deallocate(temp)

end subroutine fl2_distance


subroutine fp_distance_double(A, B, D, p)

    implicit none

    double precision, dimension(:,:), intent(in) :: A
    double precision, dimension(:,:), intent(in) :: B
    double precision, dimension(:,:), intent(inout) :: D
    double precision, intent(in) :: p

    integer :: na, nb, nv
    integer :: i, j

    double precision, allocatable, dimension(:) :: temp
    double precision :: inv_p

    nv = size(A, dim=1)

    na = size(A, dim=2)
    nb = size(B, dim=2)

    inv_p = 1.0d0 / p

    allocate(temp(nv))

!$OMP PARALLEL DO PRIVATE(temp)
    do i = 1, nb
        do j = 1, na
            temp(:) = abs(A(:,j) - B(:,i))
            D(j,i) = (sum(temp**p))**inv_p
        enddo
    enddo
!$OMP END PARALLEL DO

    deallocate(temp)

end subroutine fp_distance_double

subroutine fp_distance_integer(A, B, D, p)

    implicit none

    double precision, dimension(:,:), intent(in) :: A
    double precision, dimension(:,:), intent(in) :: B
    double precision, dimension(:,:), intent(inout) :: D
    integer, intent(in) :: p

    integer :: na, nb, nv
    integer :: i, j

    double precision, allocatable, dimension(:) :: temp
    double precision :: inv_p

    nv = size(A, dim=1)

    na = size(A, dim=2)
    nb = size(B, dim=2)

    inv_p = 1.0d0 / dble(p)

    allocate(temp(nv))

!$OMP PARALLEL DO PRIVATE(temp)
    do i = 1, nb
        do j = 1, na
            temp(:) = abs(A(:,j) - B(:,i))
            D(j,i) = (sum(temp**p))**inv_p
        enddo
    enddo
!$OMP END PARALLEL DO

    deallocate(temp)

end subroutine fp_distance_integer
