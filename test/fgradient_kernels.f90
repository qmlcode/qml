module fgradient_kernel_module

contains

subroutine outer(q1, q2, res)

    implicit none

    double precision, dimension(:), intent(in) :: q1
    double precision, dimension(:), intent(in) :: q2

    double precision, dimension(:,:), intent(out) :: res

    integer :: i, j, rep_size    

    rep_size = size(q1, dim=1)

    do i = 1, rep_size
        do j = 1, rep_size

            res(i,j) = q1(i) * q2(j)

        enddo
    enddo

end subroutine outer


end module fgradient_kernel_module


subroutine fatomic_local_gradient_kernel(x1, x2, dx2, n1, n2, nm1, nm2, na1, naq2, sigma, kernel)

    implicit none

    double precision, dimension(:,:,:), intent(in) :: x1
    double precision, dimension(:,:,:), intent(in) :: x2

    double precision, dimension(:,:,:,:,:), intent(in) :: dx2

    integer, dimension(:), intent(in) :: n1
    integer, dimension(:), intent(in) :: n2

    integer, intent(in) :: nm1
    integer, intent(in) :: nm2
    integer, intent(in) :: na1
    integer, intent(in) :: naq2

    double precision, intent(in) :: sigma

    double precision, dimension(naq2,na1), intent(out) :: kernel

    integer :: i2, j1, j2
    integer :: na, nb, xyz2
    integer :: a, b
    integer :: idx1_end, idx1_start, idx2_end, idx2_start, idx2, idx1

    integer :: rep_size

    double precision :: expd

    double precision, allocatable, dimension(:) :: d

    rep_size = size(x1, dim=3)
    allocate(d(rep_size))

    kernel = 0.0d0

    ! Molecule 1
    do a = 1, nm1

        na = n1(a)

        idx1_end = sum(n1(:a))
        idx1_start = idx1_end - na + 1


        ! Atom in Molecule 1
        do j1 = 1, na
            idx1 = idx1_start - 1 + j1

            ! Molecule 2
            do b = 1, nm2
                nb = n2(b)

                idx2_end = sum(n2(:b))
                idx2_start = idx2_end - nb + 1

                !Atom in Molecule2
                do j2 = 1, nb

                    d(:) = x1(a,j1,:)- x2(b,j2,:)
                    expd = -1.0d0/sigma**2 * exp(-(norm2(d)**2) / (2 * sigma**2))

                    ! Derivative WRT this atom in Molecule 2
                    do i2 = 1, nb

                        ! Loop over XYZ
                        do xyz2 = 1, 3

                            idx2 = (idx2_start-1)*3 + (i2-1)*3 + xyz2

                            kernel(idx2, idx1) = kernel(idx2, idx1) +  expd * dot_product(d, dx2(b, j2,:,i2,xyz2))

                        enddo
                    enddo
                enddo
            enddo
        enddo
    enddo




end subroutine fatomic_local_gradient_kernel


subroutine fgdml_kernel(x1, x2, dx1, dx2, n1, n2, nm1, nm2, na1, na2, sigma, kernel)


    use fgradient_kernel_module, only: outer

    implicit none

    double precision, dimension(:,:,:), intent(in) :: x1
    double precision, dimension(:,:,:), intent(in) :: x2

    double precision, dimension(:,:,:,:,:), intent(in) :: dx1
    double precision, dimension(:,:,:,:,:), intent(in) :: dx2

    integer, dimension(:), intent(in) :: n1
    integer, dimension(:), intent(in) :: n2

    double precision, external :: ddot

    integer, intent(in) :: nm1
    integer, intent(in) :: nm2
    integer, intent(in) :: na1
    integer, intent(in) :: na2

    double precision, intent(in) :: sigma

    double precision, dimension(na1*3,na2*3), intent(out) :: kernel

    integer :: i1, i2, j1, j2, k
    integer :: xyz1, xyz2
    integer :: a, b
    integer :: idx1_end, idx1_start, idx2_end, idx2_start, idx2, idx1

    integer :: rep_size

    double precision :: expd, expdiag

    double precision, allocatable, dimension(:) :: d

    double precision, allocatable, dimension(:,:) :: hess
    double precision, allocatable, dimension(:) :: partial

    rep_size = size(x1, dim=3)
    allocate(d(rep_size))
    allocate(partial(rep_size))
    allocate(hess(rep_size, rep_size))

    kernel = 0.0d0

    ! write(*,*) shape(x1)
    ! write(*,*) shape(x2)
    ! write(*,*) shape(dx1)
    ! write(*,*) shape(dx2)

    ! Molecules A and B
    do a = 1, nm1
        do b = 1, nm2
        
            idx1_end = sum(n1(:a))
            idx1_start = (idx1_end - n1(a))*3

            idx2_end = sum(n2(:b))
            idx2_start = (idx2_end - n2(b))*3 


            write(*,*) a, b
            ! Atoms A and B
            do i1 = 1, n1(a)
                do i2 = 1, n2(b)

                    d(:) = x1(a,i1,:)- x2(b,i2,:)
                    expd = -1.0d0/sigma**4 * exp(-(norm2(d)**2) / (2 * sigma**2))
                    expdiag = -1.0d0 * sigma**2 * expd
                    call outer(d(:), d(:), hess)

                    hess(:,:) = hess(:,:) * expd

                    do k = 1, rep_size
                       hess(k,k) = hess(k,k) + expdiag
                    enddo

                    ! Derivative wrt atoms A and B
                    do j1 = 1, n1(a)

                        ! Derivative wrt XYZ
                        do xyz1 = 1, 3
                                
                            idx1 = idx1_start + (j1-1)*3 + xyz1
                            partial = matmul(dx1(a, i1,:,j1,xyz1),hess)
                            ! partial = matmul(dx1(:,xyz1,j1,i1,a),hess)

                            ! partial = matmul(dx1(a, i1,:,j1,xyz1),hess)
                            ! call dgemv("N", rep_size, rep_size, 1.0d0, hess, rep_size, dx1(a, i1,:,j1,xyz1), 1, 0.0d0, partial, 1)


                            do j2 = 1, n2(b)
                                do xyz2 = 1, 3


                                    idx2 = idx2_start + (j2-1)*3 + xyz2

                                    kernel(idx1,idx2) = kernel(idx1,idx2) &
                                        ! & + dot_product(partial, dx2(:,xyz2,j2,i2,b))
                                        & + dot_product(partial, dx2(b, i2,:,j2,xyz2))

                                    ! kernel(idx1,idx2) = kernel(idx1,idx2) &
                                    !     & + ddot(rep_size, partial, 1, dx2(b, i2,:,j2,xyz2),1)


                                enddo
                            enddo

                        enddo
                    enddo

                enddo
            enddo

        enddo
    enddo


    deallocate(hess)
    deallocate(d)


end subroutine fgdml_kernel
