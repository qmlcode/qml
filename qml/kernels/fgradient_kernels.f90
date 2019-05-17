! MIT License
!
! Copyright (c) 2018-2019 Anders Steen Christensen
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

subroutine flocal_kernel_dpp(x1, q1, n1, nm1, sigma, kernel)

    implicit none

    double precision, dimension(:,:,:), intent(in) :: x1
    integer, dimension(:,:), intent(in) :: q1
    integer, dimension(:), intent(in) :: n1
    integer, intent(in) :: nm1

    double precision, intent(in) :: sigma

    double precision, dimension((nm1*(nm1+1))/2), intent(out) :: kernel

    integer :: j1, j2
    integer :: a, b

    integer :: rep_size
    double precision :: inv_sigma2

    integer :: idx
    double precision, allocatable, dimension(:) :: d

    kernel = 0.0d0

    rep_size = size(x1, dim=3)
    allocate(d(rep_size))

    inv_sigma2 = -1.0d0 / (2 * sigma**2)

    !$OMP PARALLEL DO private(d,idx) schedule(dynamic)
    do a = 1, nm1

        ! Molecule 2
        do b = a, nm1

            ! Atom in Molecule 1
            do j1 = 1, n1(a)

                !Atom in Molecule2
                do j2 = 1, n1(b)

                    if (q1(j1,a) == q1(j2,b)) then

                        d(:) = x1(a,j1,:)- x1(b,j2,:)

                        ! Follows UPLO = "U" convention
                        idx = a+(b*(b-1))/2

                        kernel(idx) = kernel(idx) + exp(sum(d**2) * inv_sigma2)

                    endif

                enddo
            enddo

        enddo
    enddo
    !$OMP END PARALLEL do

    deallocate(d)

end subroutine flocal_kernel_dpp


subroutine fglobal_kernel(x1, x2, q1, q2, n1, n2, nm1, nm2, sigma, kernel)

    implicit none

    double precision, dimension(:,:,:), intent(in) :: x1
    double precision, dimension(:,:,:), intent(in) :: x2

    integer, dimension(:,:), intent(in) :: q1
    integer, dimension(:,:), intent(in) :: q2

    integer, dimension(:), intent(in) :: n1
    integer, dimension(:), intent(in) :: n2

    integer, intent(in) :: nm1
    integer, intent(in) :: nm2

    double precision, intent(in) :: sigma

    double precision, dimension(nm2,nm1), intent(out) :: kernel

    integer :: i1, i2
    integer :: j1, j2

    integer :: a, b

    integer :: rep_size
    double precision :: inv_sigma2

    double precision, allocatable, dimension(:) :: d
    double precision, dimension(nm1) :: s11
    double precision, dimension(nm2) :: s22
    double precision :: s12

    double precision :: l2

    kernel = 0.0d0

    rep_size = size(x1, dim=3)
    allocate(d(rep_size))

    inv_sigma2 = -1.0d0 / (2 * sigma**2)

    s11(:) = 0.0d0

    !$OMP PARALLEL DO REDUCTION(+:s11)
    do a = 1, nm1
        do j1 = 1, n1(a)
            do i1 = 1, n1(a)

                if (q1(j1,a) == q1(i1,a)) then
                    s11(a) = s11(a) + dot_product(x1(a,j1,:), x1(a,i1,:))
                endif

            enddo
        enddo
    enddo
    !$OMP END PARALLEL DO


    s22(:) = 0.0d0

    !$OMP PARALLEL DO REDUCTION(+:s22)
    do b = 1, nm2
        do j2 = 1, n2(b)
            do i2 = 1, n2(b)

                if (q2(j2,b) == q2(i2,b)) then
                    s22(b) = s22(b) + dot_product(x2(b,j2,:), x1(b,i2,:))
                endif

            enddo
        enddo
    enddo
    !$OMP END PARALLEL DO

    !$OMP PARALLEL DO private(s12, l2) schedule(dynamic)
    do a = 1, nm1

        ! Molecule 2
        do b = 1, nm2

            s12 = 0.0d0
            ! Atom in Molecule 1
            do j1 = 1, n1(a)

                !Atom in Molecule2
                do j2 = 1, n2(b)

                    if (q1(j1,a) == q2(j2,b)) then

                        s12 = s12 + dot_product(x1(a,j1,:), x2(b,j2,:))

                    endif

                enddo
            enddo

            l2 = s11(a) + s22(b) - 2.0d0*s12
            kernel(b, a) = exp(l2 * inv_sigma2)

        enddo
    enddo
    !$OMP END PARALLEL DO

    deallocate(d)

end subroutine fglobal_kernel


subroutine flocal_kernels(x1, x2, q1, q2, n1, n2, nm1, nm2, sigmas, nsigmas, kernel)

    use omp_lib, only: omp_get_thread_num, omp_get_wtime

    implicit none

    double precision, dimension(:,:,:), intent(in) :: x1
    double precision, dimension(:,:,:), intent(in) :: x2

    integer, dimension(:,:), intent(in) :: q1
    integer, dimension(:,:), intent(in) :: q2

    integer, dimension(:), intent(in) :: n1
    integer, dimension(:), intent(in) :: n2

    integer, intent(in) :: nm1
    integer, intent(in) :: nm2

    double precision, dimension(:), intent(in) :: sigmas
    integer, intent(in) :: nsigmas

    double precision, dimension(nsigmas,nm2,nm1), intent(out) :: kernel

    ! integer, external :: omp_get_thread_num

    integer :: j1, j2
    integer :: a, b
    integer :: i

    integer :: work_done
    integer :: work_total

    ! integer :: rep_size
    double precision, dimension(nsigmas) :: inv_sigma2
    double precision :: l2
    double precision :: t_start
    double precision :: t_elapsed
    double precision :: t_eta

    kernel = 0.0d0

    do i =1, nsigmas
       inv_sigma2(i) = -1.0d0 / (2 * sigmas(i)**2)
    enddo


    ! !$OMP PARALLEL private(tid)
    ! tid = OMP_get_thread_num()
    ! write(*,*) tid
    ! !$OMP END PARALLEL

    work_total = nm1 * nm2
    work_done = 0

    t_start = omp_get_wtime()

    write(*,*) "QML: Non-alchemical Gaussian kernel progess:"
    !$OMP PARALLEL DO private(l2) shared(work_total) schedule(dynamic)
    do a = 1, nm1

        if (omp_get_thread_num() == 0) then

            t_elapsed = omp_get_wtime() - t_start
            t_eta = t_elapsed * work_total / work_done - t_elapsed

         write(*,"(F10.1, A, F10.1, A, F10.1, A, F10.1, A)") dble(work_done) / dble(work_total) * 100.0d0 , " %", &
             & t_eta, " s", t_elapsed, " s", t_elapsed + t_eta, " s"
          ! write(*,"(A, F10.1, A)") "elapsed", omp_get_wtime() - t_start, "s"




          ! write(*,"(A, F10.1, A)") "eta", (omp_get_wtime() - t_start) * dble(work_total) / dble(work_done) &
          !    & - (omp_get_wtime() - t_start), "s"
        endif

        ! Molecule 2
        do b = 1, nm2

            ! Atom in Molecule 1
            do j1 = 1, n1(a)

                !Atom in Molecule2
                do j2 = 1, n2(b)

                    if (q1(j1,a) == q2(j2,b)) then

                        l2 = sum((x1(a,j1,:) - x2(b,j2,:))**2)
                        kernel(:, b, a) = kernel(:, b, a) + exp(l2 * inv_sigma2(:))

                    endif

                enddo
            enddo

        enddo

        !$OMP ATOMIC
        work_done = work_done + nm2
        !$OMP END ATOMIC

    enddo
    !$OMP END PARALLEL do

    write(*,"(F10.1, A)") dble(work_done) / dble(work_total) * 100.0d0 , " %"
    write(*,*) "QML: Non-alchemical Gaussian kernel completed!"

end subroutine flocal_kernels


subroutine fsymmetric_local_kernels(x1, q1, n1, nm1, sigmas, nsigmas, kernel)

    use omp_lib, only: omp_get_thread_num, omp_get_wtime

    implicit none

    double precision, dimension(:,:,:), intent(in) :: x1
    integer, dimension(:,:), intent(in) :: q1
    integer, dimension(:), intent(in) :: n1
    integer, intent(in) :: nm1

    double precision, dimension(:), intent(in) :: sigmas
    integer, intent(in) :: nsigmas

    double precision, dimension(nsigmas,nm1,nm1), intent(out) :: kernel

    integer :: j1, j2
    integer :: a, b
    integer :: i

    integer :: work_done
    integer :: work_total

    integer :: rep_size
    double precision, allocatable, dimension(:) :: inv_sigma2
    double precision :: l2

    double precision :: t_start
    double precision :: t_elapsed
    double precision :: t_eta

    kernel = 0.0d0

    rep_size = size(x1, dim=3)
    allocate(inv_sigma2(nsigmas))

    do i =1, nsigmas
       inv_sigma2(i) = -1.0d0 / (2 * sigmas(i)**2)
    enddo

    work_total = (nm1 * (nm1 + 1)) / 2
    work_done = 0

    t_start = omp_get_wtime()

    write(*,*) "QML: Non-alchemical Gaussian kernel progess:"
    !$OMP PARALLEL DO private(l2) shared(work_done) schedule(dynamic)
    do a = 1, nm1

        ! if (omp_get_thread_num() == 0) then
        !     write(*,"(F10.1, A)") dble(work_done) / dble(work_total) * 100.0d0 , " %"
        ! endif

        if (omp_get_thread_num() == 0) then

            t_elapsed = omp_get_wtime() - t_start
            t_eta = t_elapsed * work_total / work_done - t_elapsed

         write(*,"(F10.1, A, F10.1, A, F10.1, A, F10.1, A)") dble(work_done) / dble(work_total) * 100.0d0 , " %", &
             & t_eta, " s", t_elapsed, " s", t_elapsed + t_eta, " s"
          ! write(*,"(A, F10.1, A)") "elapsed", omp_get_wtime() - t_start, "s"




          ! write(*,"(A, F10.1, A)") "eta", (omp_get_wtime() - t_start) * dble(work_total) / dble(work_done) &
          !    & - (omp_get_wtime() - t_start), "s"
        endif

        ! Molecule 2
        do b = a, nm1

            ! Atom in Molecule 1
            do j1 = 1, n1(a)

                !Atom in Molecule2
                do j2 = 1, n1(b)

                    if (q1(j1,a) == q1(j2,b)) then

                        l2 = sum((x1(a,j1,:) - x1(b,j2,:))**2)
                        kernel(:, b, a) = kernel(:, b, a) + exp(l2 * inv_sigma2(:))

                    endif

                enddo
            enddo

            if (b > a) then
                kernel(:, a, b) = kernel(:, b, a)

            endif

        enddo

        !$OMP ATOMIC
        work_done = work_done + (nm1 - a + 1)
        !$OMP END ATOMIC

    enddo
    !$OMP END PARALLEL do

    write(*,"(F10.1, A)") dble(work_done) / dble(work_total) * 100.0d0 , " %"
    write(*,*) "QML: Non-alchemical Gaussian kernel completed!"

    deallocate(inv_sigma2)

end subroutine fsymmetric_local_kernels



subroutine flocal_kernel(x1, x2, q1, q2, n1, n2, nm1, nm2, sigma, kernel)

    implicit none

    double precision, dimension(:,:,:), intent(in) :: x1
    double precision, dimension(:,:,:), intent(in) :: x2

    integer, dimension(:,:), intent(in) :: q1
    integer, dimension(:,:), intent(in) :: q2

    integer, dimension(:), intent(in) :: n1
    integer, dimension(:), intent(in) :: n2

    integer, intent(in) :: nm1
    integer, intent(in) :: nm2

    double precision, intent(in) :: sigma

    double precision, dimension(nm2,nm1), intent(out) :: kernel

    integer :: j1, j2
    integer :: a, b

    integer :: rep_size
    double precision :: inv_sigma2
    double precision :: l2

    kernel = 0.0d0

    rep_size = size(x1, dim=3)
    inv_sigma2 = -1.0d0 / (2 * sigma**2)

    !$OMP PARALLEL DO private(l2) schedule(dynamic)
    do a = 1, nm1

        ! Molecule 2
        do b = 1, nm2

            ! Atom in Molecule 1
            do j1 = 1, n1(a)

                !Atom in Molecule2
                do j2 = 1, n2(b)

                    if (q1(j1,a) == q2(j2,b)) then

                       l2 = sum((x1(a,j1,:) - x2(b,j2,:))**2)
                       kernel(b, a) = kernel(b, a) + exp(l2 * inv_sigma2)

                    endif

                enddo
            enddo

        enddo
    enddo
    !$OMP END PARALLEL do

end subroutine flocal_kernel


subroutine fsymmetric_local_kernel(x1, q1, n1, nm1, sigma, kernel)

    implicit none

    double precision, dimension(:,:,:), intent(in) :: x1

    integer, dimension(:,:), intent(in) :: q1

    integer, dimension(:), intent(in) :: n1

    integer, intent(in) :: nm1

    double precision, intent(in) :: sigma

    double precision, dimension(nm1,nm1), intent(out) :: kernel

    integer :: j1, j2
    integer :: a, b

    integer :: rep_size
    double precision :: inv_sigma2
    double precision :: l2

    kernel = 0.0d0

    rep_size = size(x1, dim=3)
    inv_sigma2 = -1.0d0 / (2 * sigma**2)

    !$OMP PARALLEL DO private(l2) schedule(dynamic)
    do a = 1, nm1

        ! Molecule 2
        do b = a, nm1

            ! Atom in Molecule 1
            do j1 = 1, n1(a)

                !Atom in Molecule2
                do j2 = 1, n1(b)

                    if (q1(j1,a) == q1(j2,b)) then

                        l2 = sum((x1(a,j1,:) - x1(b,j2,:))**2)
                        kernel(a, b) = kernel(a, b) + exp(l2 * inv_sigma2)

                    endif

                enddo
            enddo

            if (b > a) then
                kernel(b, a) = kernel(a, b)

            endif

        enddo
    enddo
    !$OMP END PARALLEL do

end subroutine fsymmetric_local_kernel


subroutine fatomic_local_kernel(x1, x2, q1, q2, n1, n2, nm1, nm2, na1, sigma, kernel)

    implicit none

    double precision, dimension(:,:,:), intent(in) :: x1
    double precision, dimension(:,:,:), intent(in) :: x2

    integer, dimension(:,:), intent(in) :: q1
    integer, dimension(:,:), intent(in) :: q2

    integer, dimension(:), intent(in) :: n1
    integer, dimension(:), intent(in) :: n2

    integer, intent(in) :: nm1
    integer, intent(in) :: nm2
    integer, intent(in) :: na1

    double precision, intent(in) :: sigma

    double precision, dimension(nm2,na1), intent(out) :: kernel

    integer :: j1, j2
    integer :: a, b
    integer :: idx1_start, idx1

    integer :: rep_size
    double precision :: inv_sigma2
    double precision :: l2

    kernel = 0.0d0

    rep_size = size(x1, dim=3)

    inv_sigma2 = -1.0d0 / (2 * sigma**2)

    !$OMP PARALLEL DO private(idx1_start, idx1, l2) schedule(dynamic)
    do a = 1, nm1

        idx1_start = sum(n1(:a)) - n1(a)

        ! Molecule 2
        do b = 1, nm2

            ! Atom in Molecule 1
            do j1 = 1, n1(a)

                idx1 = idx1_start + j1

                !Atom in Molecule2
                do j2 = 1, n2(b)

                    if (q1(j1,a) == q2(j2,b)) then

                        l2 = sum((x1(a,j1,:) - x2(b,j2,:))**2)
                        kernel(b,idx1) = kernel(b,idx1) + exp(l2 * inv_sigma2)

                    endif

                enddo
            enddo

        enddo
    enddo
    !$OMP END PARALLEL do

end subroutine fatomic_local_kernel


subroutine fatomic_local_gradient_kernel(x1, x2, dx2, q1, q2, n1, n2, nm1, nm2, na1, naq2, sigma, kernel)

    implicit none

    double precision, dimension(:,:,:), intent(in) :: x1
    double precision, dimension(:,:,:), intent(in) :: x2

    double precision, dimension(:,:,:,:,:), intent(in) :: dx2

    integer, dimension(:,:), intent(in) :: q1
    integer, dimension(:,:), intent(in) :: q2

    integer, dimension(:), intent(in) :: n1
    integer, dimension(:), intent(in) :: n2

    integer, intent(in) :: nm1
    integer, intent(in) :: nm2
    integer, intent(in) :: na1
    integer, intent(in) :: naq2

    double precision, intent(in) :: sigma

    double precision, dimension(naq2,na1), intent(out) :: kernel

    integer :: i2, j1, j2
    integer :: xyz2
    integer :: a, b
    integer :: idx1_start, idx2_end, idx2_start, idx2, idx1

    integer :: rep_size

    double precision :: expd
    double precision :: inv_2sigma2
    double precision :: inv_sigma2

    double precision, allocatable, dimension(:) :: d
    double precision, allocatable, dimension(:,:,:,:) :: sorted_derivs


    rep_size = size(x1, dim=3)
    allocate(d(rep_size))

    inv_2sigma2 = -1.0d0 / (2 * sigma**2)
    inv_sigma2 = -1.0d0 / (sigma**2)

    allocate(sorted_derivs(rep_size,maxval(n2)*3,maxval(n2),nm2))

    sorted_derivs = 0.0d0

    ! Presort the representation derivatives
    do b = 1, nm2
        do i2 = 1, n2(b)
            idx2 = 0

            do j2 = 1, n2(b)

                do xyz2 = 1, 3
                    idx2 = idx2 + 1

                    sorted_derivs(:,idx2,i2,b) = dx2(b, i2,:,j2,xyz2)

                enddo
            enddo
        enddo
    enddo

    kernel = 0.0d0

    !$OMP PARALLEL DO PRIVATE(idx2_end,idx2_start,d,expd,idx1_start,idx1) schedule(dynamic)
    do a = 1, nm1

        idx1_start = sum(n1(:a)) - n1(a) + 1

        ! Atom in Molecule 1
        do j1 = 1, n1(a)

            idx1 = idx1_start - 1 + j1

            ! Molecule 2
            do b = 1, nm2

                idx2_start = (sum(n2(:b)) - n2(b))*3 + 1
                idx2_end = sum(n2(:b))*3

                !Atom in Molecule2
                do j2 = 1, n2(b)

                    if (q1(j1,a) == q2(j2,b)) then
                        ! Calculate the distance vector, and some intermediate results
                        d(:) = x1(a,j1,:)- x2(b,j2,:)
                        expd = inv_sigma2 * exp(sum(d**2) * inv_2sigma2)

                        ! Add the dot product to the kernel in one BLAS call
                        call dgemv("T", rep_size, n2(b)*3, expd, sorted_derivs(:,:n2(b)*3,j2,b), &
                            & rep_size, d, 1, 1.0d0, kernel(idx2_start:idx2_end,idx1), 1)
                    endif

                enddo
            enddo
        enddo
    enddo
    !$OMP END PARALLEL do

    deallocate(sorted_derivs)
    deallocate(d)

end subroutine fatomic_local_gradient_kernel


! NOTE: Legacy code without any fancy BLAS calls, for reference
!
! subroutine fatomic_local_gradient_kernel(x1, x2, dx2, q1, q2, n1, n2, nm1, nm2, na1, naq2, sigma, kernel)
!
!     implicit none
!
!     double precision, dimension(:,:,:), intent(in) :: x1
!     double precision, dimension(:,:,:), intent(in) :: x2
!
!     double precision, dimension(:,:,:,:,:), intent(in) :: dx2
!
!     integer, dimension(:,:), intent(in) :: q1
!     integer, dimension(:,:), intent(in) :: q2
!
!     integer, dimension(:), intent(in) :: n1
!     integer, dimension(:), intent(in) :: n2
!
!     integer, intent(in) :: nm1
!     integer, intent(in) :: nm2
!     integer, intent(in) :: na1
!     integer, intent(in) :: naq2
!
!     double precision, intent(in) :: sigma
!
!     double precision, dimension(naq2,na1), intent(out) :: kernel
!
!     integer :: i2, j1, j2
!     integer :: na, nb, xyz2
!     integer :: a, b
!     integer :: idx1_end, idx1_start, idx2_end, idx2_start, idx2, idx1
!
!     integer :: rep_size
!
!     double precision :: expd
!     double precision :: inv_2sigma2
!     double precision :: inv_sigma2
!
!     double precision, allocatable, dimension(:) :: d
!
!     rep_size = size(x1, dim=3)
!     allocate(d(rep_size))
!
!     inv_2sigma2 = -1.0d0 / (2 * sigma**2)
!     inv_sigma2 = -1.0d0 / (sigma**2)
!
!     kernel = 0.0d0
!
!     ! Molecule 1
!     do a = 1, nm1
!
!         na = n1(a)
!
!         idx1_end = sum(n1(:a))
!         idx1_start = idx1_end - na + 1
!
!
!         ! Atom in Molecule 1
!         do j1 = 1, na
!             idx1 = idx1_start - 1 + j1
!
!             ! Molecule 2
!             do b = 1, nm2
!                 nb = n2(b)
!
!                 idx2_end = sum(n2(:b))
!                 idx2_start = idx2_end - nb + 1
!
!                 !Atom in Molecule2
!                 do j2 = 1, nb
!
!                     if (q1(j1,a) == q2(j2,b)) then
!
!                         d(:) = x1(a,j1,:)- x2(b,j2,:)
!                         ! expd = -1.0d0/sigma**2 * exp(-(norm2(d)**2) / (2 * sigma**2))
!                         expd = inv_sigma2 * exp((norm2(d)**2) * inv_2sigma2) ! Possibly 4??
!
!                         ! Derivative WRT this atom in Molecule 2
!                         do i2 = 1, nb
!
!                                 ! Loop over XYZ
!                                 do xyz2 = 1, 3
!
!                                     idx2 = (idx2_start-1)*3 + (i2-1)*3 + xyz2
!                                     kernel(idx2, idx1) = kernel(idx2, idx1) +  expd * dot_product(d, dx2(b, j2,:,i2,xyz2))
!
!                                 enddo
!
!                             endif
!                         enddo
!
!                     endif
!
!                 enddo
!             enddo
!         enddo
!     enddo
!
!
!
!
! end subroutine fatomic_local_gradient_kernel


subroutine flocal_gradient_kernel(x1, x2, dx2, q1, q2, n1, n2, nm1, nm2, naq2, sigma, kernel)

    implicit none

    double precision, dimension(:,:,:), intent(in) :: x1
    double precision, dimension(:,:,:), intent(in) :: x2

    double precision, dimension(:,:,:,:,:), intent(in) :: dx2

    integer, dimension(:,:), intent(in) :: q1
    integer, dimension(:,:), intent(in) :: q2

    integer, dimension(:), intent(in) :: n1
    integer, dimension(:), intent(in) :: n2

    integer, intent(in) :: nm1
    integer, intent(in) :: nm2
    integer, intent(in) :: naq2

    double precision, intent(in) :: sigma

    double precision, dimension(naq2,nm1), intent(out) :: kernel

    integer :: i2, j1, j2
    integer :: nb, xyz2
    integer :: a, b
    integer :: idx2_end, idx2_start, idx2

    integer :: rep_size

    double precision :: expd, inv_2sigma2, inv_sigma2

    double precision, allocatable, dimension(:) :: d
    double precision, allocatable, dimension(:,:,:,:) :: sorted_derivs

    rep_size = size(x1, dim=3)
    allocate(d(rep_size))

    inv_2sigma2 = -1.0d0 / (2 * sigma**2)
    inv_sigma2 = -1.0d0 / (sigma**2)

    kernel = 0.0d0


    allocate(sorted_derivs(rep_size,maxval(n2)*3,maxval(n2),nm2))

    sorted_derivs = 0.0d0

    ! Presort the representation derivatives
    do b = 1, nm2
        do i2 = 1, n2(b)
            idx2 = 0

            do j2 = 1, n2(b)

                do xyz2 = 1, 3
                    idx2 = idx2 + 1

                    sorted_derivs(:,idx2,i2,b) = dx2(b, i2,:,j2,xyz2)

                enddo
            enddo
        enddo
    enddo

    ! Molecule 1
    !$OMP PARALLEL DO PRIVATE(idx2_end,idx2_start,d,expd) schedule(dynamic)
    do a = 1, nm1

        ! Atom in Molecule 1
        do j1 = 1, n1(a)

            ! Molecule 2
            do b = 1, nm2
                nb = n2(b)

                idx2_start = (sum(n2(:b)) - n2(b))*3 + 1
                idx2_end = sum(n2(:b))*3

                !Atom in Molecule2
                do j2 = 1, n2(b)

                    if (q1(j1,a) == q2(j2,b)) then
                        ! Calculate the distance vector, and some intermediate results
                        d(:) = x1(a,j1,:)- x2(b,j2,:)
                        expd = inv_sigma2 * exp(sum(d**2) * inv_2sigma2)

                        ! Add the dot products to the kernel in one BLAS call
                        call dgemv("T", rep_size, n2(b)*3, expd, sorted_derivs(:,:n2(b)*3,j2,b), &
                            & rep_size, d, 1, 1.0d0, kernel(idx2_start:idx2_end,a), 1)
                    endif

                enddo
            enddo
        enddo
    enddo
    !$OMP END PARALLEL do

    deallocate(sorted_derivs)
    deallocate(d)

end subroutine flocal_gradient_kernel


subroutine fgdml_kernel(x1, x2, dx1, dx2, q1, q2, n1, n2, nm1, nm2, na1, na2, sigma, kernel)

    implicit none

    double precision, dimension(:,:,:), intent(in) :: x1
    double precision, dimension(:,:,:), intent(in) :: x2

    double precision, dimension(:,:,:,:,:), intent(in) :: dx1
    double precision, dimension(:,:,:,:,:), intent(in) :: dx2

    integer, dimension(:,:), intent(in) :: q1
    integer, dimension(:,:), intent(in) :: q2

    integer, dimension(:), intent(in) :: n1
    integer, dimension(:), intent(in) :: n2

    integer, intent(in) :: nm1
    integer, intent(in) :: nm2
    integer, intent(in) :: na1
    integer, intent(in) :: na2

    double precision, intent(in) :: sigma

    double precision, dimension(na2*3,na1*3), intent(out) :: kernel

    integer :: i1, i2, j2, k
    integer :: xyz2
    integer :: a, b
    integer :: idx1_end, idx1_start, idx2_end, idx2_start, idx2

    integer :: rep_size

    double precision :: expd, expdiag

    double precision :: inv_2sigma2
    double precision :: inv_sigma4
    double precision :: sigma2

    double precision, allocatable, dimension(:) :: d

    double precision, allocatable, dimension(:,:) :: hess
    double precision, allocatable, dimension(:,:) :: partial

    double precision, allocatable, dimension(:,:,:,:) :: sorted_derivs1
    double precision, allocatable, dimension(:,:,:,:) :: sorted_derivs2

    rep_size = size(x1, dim=3)
    allocate(d(rep_size))
    allocate(partial(rep_size,maxval(n2)*3))
    partial = 0.0d0
    allocate(hess(rep_size, rep_size))

    allocate(sorted_derivs1(rep_size,maxval(n1)*3,maxval(n1),nm1))
    allocate(sorted_derivs2(rep_size,maxval(n2)*3,maxval(n2),nm2))

    sorted_derivs1 = 0.0d0
    sorted_derivs2 = 0.0d0

    ! Presort the representation derivatives
    do b = 1, nm2
        do i2 = 1, n2(b)
            idx2 = 0

            do j2 = 1, n2(b)

                do xyz2 = 1, 3
                    idx2 = idx2 + 1

                    sorted_derivs2(:,idx2,i2,b) = dx2(b, i2,:,j2,xyz2)

                enddo
            enddo
        enddo
    enddo

    ! Presort the representation derivatives
    do b = 1, nm1
        do i2 = 1, n1(b)
            idx2 = 0

            do j2 = 1, n1(b)

                do xyz2 = 1, 3
                    idx2 = idx2 + 1

                    sorted_derivs1(:,idx2,i2,b) = dx1(b, i2,:,j2,xyz2)

                enddo
            enddo
        enddo
    enddo

    ! Reset kernel
    kernel = 0.0d0

    ! Calculate these only once
    inv_2sigma2 = -1.0d0 / (2 * sigma**2)
    inv_sigma4 = -1.0d0 / (sigma**4)
    sigma2 = -1.0d0 * sigma**2

    !$OMP PARALLEL DO PRIVATE(idx1_start,idx2_start,d,expd,expdiag,hess,idx1_end,idx2_end,partial) schedule(dynamic)
    do a = 1, nm1
        idx1_start = (sum(n1(:a)) - n1(a))*3 + 1
        idx1_end = sum(n1(:a))*3

        do b = 1, nm2
            idx2_start = (sum(n2(:b)) - n2(b))*3 + 1
            idx2_end = sum(n2(:b))*3

            ! Atoms A and B
            do i1 = 1, n1(a)
                do i2 = 1, n2(b)

                    if (q1(i1,a) == q2(i2,b)) then

                        ! Calculate the distance vector, and some intermediate results
                        d(:) = x1(a,i1,:)- x2(b,i2,:)
                        expd = inv_sigma4 * exp(sum(d**2) * inv_2sigma2)
                        expdiag = sigma2 * expd

                        ! Calculate the outer product of the distance
                        hess = 0.0d0
                        call dsyr("U", rep_size, expd, d, 1, hess, rep_size)

                        do k = 1, rep_size
                           hess(k,k) = hess(k,k) + expdiag
                        enddo

                        ! ! Do the first half of the dot product, save in partial(:,:)
                        ! call dsymm("L", "U", rep_size, n1(a)*3, 1.0d0, hess(:,:), &
                        !     & rep_size, sorted_derivs1(:,:n1(a)*3,i1,a), rep_size, &
                        !     & 0.0d0, partial(:,:n1(a)*3), rep_size)

                        ! ! Add the dot product to the kernel in one BLAS call
                        ! call dgemm("T", "N", n1(a)*3, n2(b)*3, rep_size, 1.0d0, &
                        !     & partial(:,:n1(a)*3), rep_size, &
                        !     & sorted_derivs2(:,:n2(b)*3,i2,b), rep_size, 1.0d0, &
                        !     & kernel(idx1_start:idx1_end,idx2_start:idx2_end), n1(a)*3, 1)

                        ! Do the first half of the dot product, save in partial(:,:)
                        call dsymm("L", "U", rep_size, n2(b)*3, 1.0d0, hess(:,:), &
                            & rep_size, sorted_derivs2(:,:n2(b)*3,i2,b), rep_size, &
                            & 0.0d0, partial(:,:n2(b)*3), rep_size)

                        ! Add the dot product to the kernel in one BLAS call
                        call dgemm("T", "N", n2(b)*3, n1(a)*3, rep_size, 1.0d0, &
                            & partial(:,:n2(b)*3), rep_size, &
                            & sorted_derivs1(:,:n1(a)*3,i1,a), rep_size, 1.0d0, &
                            & kernel(idx2_start:idx2_end,idx1_start:idx1_end), n2(b)*3, 1)

                    endif

                enddo
            enddo

        enddo
    enddo
    !$OMP END PARALLEL do

    deallocate(hess)
    deallocate(sorted_derivs1)
    deallocate(sorted_derivs2)
    deallocate(partial)
    deallocate(d)

end subroutine fgdml_kernel


subroutine fsymmetric_gdml_kernel(x1, dx1, q1, n1, nm1, na1, sigma, kernel)

    implicit none

    double precision, dimension(:,:,:), intent(in) :: x1

    double precision, dimension(:,:,:,:,:), intent(in) :: dx1

    integer, dimension(:,:), intent(in) :: q1

    integer, dimension(:), intent(in) :: n1

    integer, intent(in) :: nm1
    integer, intent(in) :: na1

    double precision, intent(in) :: sigma

    double precision, dimension(na1*3,na1*3), intent(out) :: kernel

    integer :: i1, i2, j2, k
    integer :: xyz2
    integer :: a, b
    integer :: idx1_end, idx1_start, idx2_end, idx2_start, idx2

    integer :: rep_size

    double precision :: expd, expdiag

    double precision :: inv_2sigma2
    double precision :: inv_sigma4
    double precision :: sigma2

    double precision, allocatable, dimension(:) :: d

    double precision, allocatable, dimension(:,:) :: hess
    double precision, allocatable, dimension(:,:) :: partial

    double precision, allocatable, dimension(:,:,:,:) :: sorted_derivs1

    rep_size = size(x1, dim=3)
    allocate(d(rep_size))
    allocate(partial(rep_size,maxval(n1)*3))
    partial = 0.0d0
    allocate(hess(rep_size, rep_size))

    allocate(sorted_derivs1(rep_size,maxval(n1)*3,maxval(n1),nm1))

    sorted_derivs1 = 0.0d0

    ! Presort the representation derivatives
    do b = 1, nm1
        do i2 = 1, n1(b)
            idx2 = 0

            do j2 = 1, n1(b)

                do xyz2 = 1, 3
                    idx2 = idx2 + 1

                    sorted_derivs1(:,idx2,i2,b) = dx1(b, i2,:,j2,xyz2)

                enddo
            enddo
        enddo
    enddo

    ! Reset kernel
    kernel = 0.0d0

    ! Calculate these only once
    inv_2sigma2 = -1.0d0 / (2 * sigma**2)
    inv_sigma4 = -1.0d0 / (sigma**4)
    sigma2 = -1.0d0 * sigma**2

    !$OMP PARALLEL DO PRIVATE(idx1_start,idx2_start,d,expd,expdiag,hess,idx1_end,idx2_end,partial) schedule(dynamic)
    do a = 1, nm1
        idx1_start = (sum(n1(:a)) - n1(a))*3 + 1
        idx1_end = sum(n1(:a))*3

        do b = a, nm1
            idx2_start = (sum(n1(:b)) - n1(b))*3 + 1
            idx2_end = sum(n1(:b))*3

            ! Atoms A and B
            do i1 = 1, n1(a)
                do i2 = 1, n1(b)

                    if (q1(i1,a) == q1(i2,b)) then
                        ! Calculate the distance vector, and some intermediate results
                        d(:) = x1(a,i1,:)- x1(b,i2,:)
                        expd = inv_sigma4 * exp(sum(d**2) * inv_2sigma2)
                        expdiag = sigma2 * expd

                        ! Calculate the outer product of the distance
                        hess = 0.0d0
                        call dsyr("U", rep_size, expd, d, 1, hess, rep_size)

                        do k = 1, rep_size
                           hess(k,k) = hess(k,k) + expdiag
                        enddo

                        ! Do the first half of the dot product, save in partial(:,:)
                        call dsymm("L", "U", rep_size, n1(a)*3, 1.0d0, hess(:,:), &
                            & rep_size, sorted_derivs1(:,:n1(a)*3,i1,a), rep_size, &
                            & 0.0d0, partial(:,:n1(a)*3), rep_size)

                        ! Add the dot product to the kernel in one BLAS call
                        call dgemm("T", "N", n1(a)*3, n1(b)*3, rep_size, 1.0d0, &
                            & partial(:,:n1(a)*3), rep_size, &
                            & sorted_derivs1(:,:n1(b)*3,i2,b), rep_size, 1.0d0, &
                            & kernel(idx1_start:idx1_end,idx2_start:idx2_end), n1(a)*3, 1)
                    endif
                enddo
            enddo

            if (b > a) then
               kernel(idx2_start:idx2_end,idx1_start:idx1_end) &
                & = transpose(kernel(idx1_start:idx1_end,idx2_start:idx2_end))
            endif

        enddo
    enddo
    !$OMP END PARALLEL do

    deallocate(hess)
    deallocate(sorted_derivs1)
    deallocate(partial)
    deallocate(d)

end subroutine fsymmetric_gdml_kernel


subroutine fgaussian_process_kernel(x1, x2, dx1, dx2, q1, q2, n1, n2, nm1, nm2, na1, na2, sigma, kernel)

    implicit none

    double precision, dimension(:,:,:), intent(in) :: x1
    double precision, dimension(:,:,:), intent(in) :: x2

    double precision, dimension(:,:,:,:,:), intent(in) :: dx1
    double precision, dimension(:,:,:,:,:), intent(in) :: dx2

    integer, dimension(:,:), intent(in) :: q1
    integer, dimension(:,:), intent(in) :: q2

    integer, dimension(:), intent(in) :: n1
    integer, dimension(:), intent(in) :: n2

    integer, intent(in) :: nm1
    integer, intent(in) :: nm2
    integer, intent(in) :: na1
    integer, intent(in) :: na2

    double precision, intent(in) :: sigma

    double precision, dimension(na2*3+nm2,na1*3+nm1), intent(out) :: kernel

    integer :: i1, i2, j1, j2, k
    integer :: xyz2
    integer :: a, b
    integer :: idx1_end, idx1_start, idx2_end, idx2_start, idx2

    integer :: rep_size

    double precision :: expd, expdiag

    double precision :: inv_2sigma2
    double precision :: inv_sigma2
    double precision :: inv_sigma4
    double precision :: sigma2

    double precision, allocatable, dimension(:) :: d

    double precision, allocatable, dimension(:,:) :: hess
    double precision, allocatable, dimension(:,:) :: partial

    double precision, allocatable, dimension(:,:,:,:) :: sorted_derivs1
    double precision, allocatable, dimension(:,:,:,:) :: sorted_derivs2

    ! Reset kernel
    kernel = 0.0d0

    ! Calculate these only once
    inv_2sigma2 = -1.0d0 / (2 * sigma**2)
    inv_sigma2 = -1.0d0 / (sigma**2)
    inv_sigma4 = -1.0d0 / (sigma**4)
    sigma2 = -1.0d0 * sigma**2

    rep_size = size(x1, dim=3)
    allocate(d(rep_size))
    allocate(partial(rep_size,maxval(n2)*3))
    partial = 0.0d0
    allocate(hess(rep_size, rep_size))

    allocate(sorted_derivs1(rep_size,maxval(n1)*3,maxval(n1),nm1))
    allocate(sorted_derivs2(rep_size,maxval(n2)*3,maxval(n2),nm2))

    sorted_derivs1 = 0.0d0
    sorted_derivs2 = 0.0d0


    ! Presort the representation derivatives
    do b = 1, nm1
        do i2 = 1, n1(b)
            idx2 = 0

            do j2 = 1, n1(b)

                do xyz2 = 1, 3
                    idx2 = idx2 + 1

                    sorted_derivs1(:,idx2,i2,b) = dx1(b, i2,:,j2,xyz2)

                enddo
            enddo
        enddo
    enddo


    ! Presort the representation derivatives
    do b = 1, nm2
        do i2 = 1, n2(b)
            idx2 = 0

            do j2 = 1, n2(b)

                do xyz2 = 1, 3
                    idx2 = idx2 + 1

                    sorted_derivs2(:,idx2,i2,b) = dx2(b, i2,:,j2,xyz2)

                enddo
            enddo
        enddo
    enddo

    ! write (*,*) "Kernel"

    !$OMP PARALLEL DO private(d) schedule(dynamic)
    do a = 1, nm1

        ! Molecule 2
        do b = 1, nm2

            ! Atom in Molecule 1
            do j1 = 1, n1(a)

                !Atom in Molecule2
                do j2 = 1, n2(b)

                    if (q1(j1,a) == q2(j2,b)) then
                        d(:) = x1(a,j1,:)- x2(b,j2,:)
                        !kernel(a, b) = kernel(a, b) + exp((norm2(d)**2) * inv_2sigma2)
                        kernel(b, a) = kernel(b, a) + exp(sum(d**2) * inv_2sigma2)
                    endif

                enddo
            enddo

        enddo
    enddo
    !$OMP END PARALLEL do

    ! write (*,*) "Kernel grad 1"

    ! Molecule 1
    !$OMP PARALLEL DO PRIVATE(idx2_end,idx2_start,d,expd) schedule(dynamic)
    do a = 1, nm1

        ! Atom in Molecule 1
        do j1 = 1, n1(a)

            ! Molecule 2
            do b = 1, nm2

                idx2_start = (sum(n2(:b)) - n2(b))*3 + 1
                idx2_end = sum(n2(:b))*3

                !Atom in Molecule2
                do j2 = 1, n2(b)

                    if (q1(j1,a) == q2(j2,b)) then
                        ! Calculate the distance vector, and some intermediate results
                        d(:) = x1(a,j1,:)- x2(b,j2,:)
                        expd = inv_sigma2 * exp(sum(d**2) * inv_2sigma2)

                        ! Add the dot products to the kernel in one BLAS call
                        call dgemv("T", rep_size, n2(b)*3, expd, sorted_derivs2(:,:n2(b)*3,j2,b), &
                            ! & rep_size, d, 1, 1.0d0, kernel(a,idx2_start+nm2:idx2_end+nm2), 1)
                            & rep_size, d, 1, 1.0d0, kernel(idx2_start+nm2:idx2_end+nm2,a), 1)

                    endif

                enddo
            enddo
        enddo
    enddo
    !$OMP END PARALLEL do

    ! write (*,*) "Kernel grad 2"

    ! Molecule 1
    !$OMP PARALLEL DO PRIVATE(idx1_end,idx1_start,d,expd) schedule(dynamic)
    do a = 1, nm2

        ! Atom in Molecule 1
        do j1 = 1, n2(a)

            ! Molecule 2
            do b = 1, nm1

                idx1_start = (sum(n1(:b)) - n1(b))*3 + 1
                idx1_end = sum(n1(:b))*3

                !Atom in Molecule2
                do j2 = 1, n1(b)

                    if (q2(j1,a) == q1(j2,b)) then

                        ! Calculate the distance vector, and some intermediate results
                        d(:) = x2(a,j1,:)- x1(b,j2,:)
                        expd = inv_sigma2 * exp(sum(d**2) * inv_2sigma2)

                        ! write(*,*) nm1,idx1_start+nm1,idx1_end+nm1,a

                        ! Add the dot products to the kernel in one BLAS call
                        call dgemv("T", rep_size, n1(b)*3, expd, sorted_derivs1(:,:n1(b)*3,j2,b), &
                            ! & rep_size, d, 1, 1.0d0, kernel(idx1_start+nm1:idx1_end+nm1,a), 1)
                            & rep_size, d, 1, 1.0d0, kernel(a,idx1_start+nm1:idx1_end+nm1), 1)

                   endif

                enddo
            enddo
        enddo
    enddo
    !$OMP END PARALLEL do

    ! write (*,*) "Kernel hessian"

    !$OMP PARALLEL DO PRIVATE(idx1_start,idx2_start,d,expd,expdiag,hess,idx1_end,idx2_end,partial) schedule(dynamic)
    do a = 1, nm1
        idx1_start = (sum(n1(:a)) - n1(a))*3 + 1 + nm1
        idx1_end = sum(n1(:a))*3 + nm1

        do b = 1, nm2
            idx2_start = (sum(n2(:b)) - n2(b))*3 + 1 + nm2
            idx2_end = sum(n2(:b))*3 + nm2

            ! Atoms A and B
            do i1 = 1, n1(a)
                do i2 = 1, n2(b)

                    if (q1(i1,a) == q2(i2,b)) then

                        ! Calculate the distance vector, and some intermediate results
                        d(:) = x1(a,i1,:)- x2(b,i2,:)
                        expd = inv_sigma4 * exp(sum(d**2) * inv_2sigma2)
                        expdiag = sigma2 * expd

                        ! Calculate the outer product of the distance
                        hess = 0.0d0
                        call dsyr("U", rep_size, expd, d, 1, hess, rep_size)

                        do k = 1, rep_size
                           hess(k,k) = hess(k,k) + expdiag
                        enddo

                        ! ! Do the first half of the dot product, save in partial(:,:)
                        ! call dsymm("L", "U", rep_size, n1(a)*3, 1.0d0, hess(:,:), &
                        !     & rep_size, sorted_derivs1(:,:n1(a)*3,i1,a), rep_size, &
                        !     & 0.0d0, partial(:,:n1(a)*3), rep_size)

                        ! ! Add the dot product to the kernel in one BLAS call
                        ! call dgemm("T", "N", n1(a)*3, n2(b)*3, rep_size, 1.0d0, &
                        !     & partial(:,:n1(a)*3), rep_size, &
                        !     & sorted_derivs2(:,:n2(b)*3,i2,b), rep_size, 1.0d0, &
                        !     & kernel(idx1_start:idx1_end,idx2_start:idx2_end), n1(a)*3, 1)

                        ! Do the first half of the dot product, save in partial(:,:)
                        call dsymm("L", "U", rep_size, n2(b)*3, 1.0d0, hess(:,:), &
                            & rep_size, sorted_derivs2(:,:n2(b)*3,i2,b), rep_size, &
                            & 0.0d0, partial(:,:n2(b)*3), rep_size)

                        ! Add the dot product to the kernel in one BLAS call
                        call dgemm("T", "N", n2(b)*3, n1(a)*3, rep_size, 1.0d0, &
                            & partial(:,:n2(b)*3), rep_size, &
                            & sorted_derivs1(:,:n1(a)*3,i1,a), rep_size, 1.0d0, &
                            & kernel(idx2_start:idx2_end,idx1_start:idx1_end), n2(b)*3, 1)

                    endif

                enddo
            enddo

        enddo
    enddo
    !$OMP END PARALLEL do

    deallocate(hess)
    deallocate(sorted_derivs1)
    deallocate(sorted_derivs2)
    deallocate(partial)
    deallocate(d)

end subroutine fgaussian_process_kernel


subroutine fsymmetric_gaussian_process_kernel(x1, dx1, q1, n1, nm1, na1, sigma, kernel)

    implicit none

    double precision, dimension(:,:,:), intent(in) :: x1

    double precision, dimension(:,:,:,:,:), intent(in) :: dx1

    integer, dimension(:,:), intent(in) :: q1

    integer, dimension(:), intent(in) :: n1

    integer, intent(in) :: nm1
    integer, intent(in) :: na1

    double precision, intent(in) :: sigma

    double precision, dimension(na1*3+nm1,na1*3+nm1), intent(out) :: kernel

    integer :: i1, i2, j1, j2, k
    integer :: xyz2
    integer :: a, b
    integer :: idx1_end, idx1_start, idx2_end, idx2_start, idx2

    integer :: rep_size

    double precision :: expd, expdiag

    double precision :: inv_2sigma2
    double precision :: inv_sigma2
    double precision :: inv_sigma4
    double precision :: sigma2

    double precision, allocatable, dimension(:) :: d

    double precision, allocatable, dimension(:,:) :: hess
    double precision, allocatable, dimension(:,:) :: partial

    double precision, allocatable, dimension(:,:,:,:) :: sorted_derivs1

    ! Reset kernel
    kernel(:,:) = 0.0d0

    ! Calculate these only once
    inv_2sigma2 = -1.0d0 / (2 * sigma**2)
    inv_sigma2 = -1.0d0 / (sigma**2)
    inv_sigma4 = -1.0d0 / (sigma**4)
    sigma2 = -1.0d0 * sigma**2
    rep_size = size(x1, dim=3)

    allocate(d(rep_size))

    allocate(partial(rep_size,maxval(n1)*3))
    partial(:,:) = 0.0d0

    allocate(hess(rep_size, rep_size))

    allocate(sorted_derivs1(rep_size,maxval(n1)*3,maxval(n1),nm1))
    sorted_derivs1(:,:,:,:)= 0.0d0

    ! Presort the representation derivatives
    do b = 1, nm1
        do i2 = 1, n1(b)
            idx2 = 0

            do j2 = 1, n1(b)

                do xyz2 = 1, 3
                    idx2 = idx2 + 1

                    sorted_derivs1(:,idx2,i2,b) = dx1(b, i2,:,j2,xyz2)

                enddo
            enddo
        enddo
    enddo

    ! write (*,*) "Kernel"

    !$OMP PARALLEL DO private(d) schedule(dynamic)
    do a = 1, nm1

        ! Molecule 2
        do b = a, nm1

            ! Atom in Molecule 1
            do j1 = 1, n1(a)

                !Atom in Molecule2
                do j2 = 1, n1(b)

                    if (q1(j1,a) == q1(j2,b)) then
                        d(:) = x1(a,j1,:)- x1(b,j2,:)
                        kernel(a, b) = kernel(a, b) + exp(sum(d**2) * inv_2sigma2)
                    endif

                enddo
            enddo

            if (b > a) then
                kernel(b, a) = kernel(a, b)

            endif

        enddo
    enddo
    !$OMP END PARALLEL do

    ! write (*,*) "Kernel grad 1"

    ! Molecule 1
    !$OMP PARALLEL DO PRIVATE(idx2_end,idx2_start,d,expd) schedule(dynamic)
    do a = 1, nm1

        ! Atom in Molecule 1
        do j1 = 1, n1(a)

            ! Molecule 2
            do b = 1, nm1

                idx2_start = (sum(n1(:b)) - n1(b))*3 + 1
                idx2_end = sum(n1(:b))*3

                !Atom in Molecule2
                do j2 = 1, n1(b)

                    if (q1(j1,a) == q1(j2,b)) then
                        ! Calculate the distance vector, and some intermediate results
                        d(:) = x1(a,j1,:)- x1(b,j2,:)
                        expd = inv_sigma2 * exp(sum(d**2) * inv_2sigma2)

                        ! Add the dot products to the kernel in one BLAS call
                        call dgemv("T", rep_size, n1(b)*3, expd, sorted_derivs1(:,:n1(b)*3,j2,b), &
                            & rep_size, d, 1, 1.0d0, kernel(a,idx2_start+nm1:idx2_end+nm1), 1)
                    endif

                enddo
            enddo
        enddo
    enddo
    !$OMP END PARALLEL do

    kernel(nm1:, :nm1) = transpose(kernel(:nm1, nm1:))

    ! write (*,*) "Kernel hessian"

    !$OMP PARALLEL DO PRIVATE(idx1_start,idx2_start,d,expd,expdiag,hess,idx1_end,idx2_end,partial) schedule(dynamic)
    do a = 1, nm1
        idx1_start = (sum(n1(:a)) - n1(a))*3 + 1 + nm1
        idx1_end = sum(n1(:a))*3 + nm1

        do b = a, nm1
            idx2_start = (sum(n1(:b)) - n1(b))*3 + 1 + nm1
            idx2_end = sum(n1(:b))*3 + nm1

            ! Atoms A and B
            do i1 = 1, n1(a)
                do i2 = 1, n1(b)

                    if (q1(i1,a) == q1(i2,b)) then

                        ! Calculate the distance vector, and some intermediate results
                        d(:) = x1(a,i1,:)- x1(b,i2,:)
                        expd = inv_sigma4 * exp(sum(d**2) * inv_2sigma2)
                        expdiag = sigma2 * expd

                        ! Calculate the outer product of the distance
                        hess = 0.0d0
                        call dsyr("U", rep_size, expd, d, 1, hess, rep_size)

                        do k = 1, rep_size
                           hess(k,k) = hess(k,k) + expdiag
                        enddo

                        ! Do the first half of the dot product, save in partial(:,:)
                        call dsymm("L", "U", rep_size, n1(a)*3, 1.0d0, hess(:,:), &
                            & rep_size, sorted_derivs1(:,:n1(a)*3,i1,a), rep_size, &
                            & 0.0d0, partial(:,:n1(a)*3), rep_size)

                        ! Add the dot product to the kernel in one BLAS call
                        call dgemm("T", "N", n1(a)*3, n1(b)*3, rep_size, 1.0d0, &
                            & partial(:,:n1(a)*3), rep_size, &
                            & sorted_derivs1(:,:n1(b)*3,i2,b), rep_size, 1.0d0, &
                            & kernel(idx1_start:idx1_end,idx2_start:idx2_end), n1(a)*3, 1)

                    endif

                enddo
            enddo

            if (b > a) then
               kernel(idx2_start:idx2_end,idx1_start:idx1_end) &
                & = transpose(kernel(idx1_start:idx1_end,idx2_start:idx2_end))
            endif

        enddo
    enddo
    !$OMP END PARALLEL do

    deallocate(hess)
    deallocate(sorted_derivs1)
    deallocate(partial)
    deallocate(d)

end subroutine fsymmetric_gaussian_process_kernel
