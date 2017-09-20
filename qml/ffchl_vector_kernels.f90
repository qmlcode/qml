subroutine fget_atomic_force_alphas_fchl(x1, forces, nneigh1, &
       & sigmas, lambda, na1, nsigmas, &
       & t_width, d_width, cut_distance, order, pd, &
       & distance_scale, angular_scale, alchemy, two_body_power, three_body_power, alphas)

    use ffchl_module, only: scalar, get_threebody_fourier, get_twobody_weights, &
                        & get_displaced_representaions, get_angular_norm2

    implicit none

    double precision, allocatable, dimension(:,:,:,:) :: fourier

    ! fchl descriptors for the training set, format (i,maxatoms,5,maxneighbors)
    double precision, dimension(:,:,:), intent(in) :: x1
    double precision, dimension(:,:), intent(in) :: forces

    double precision, allocatable, dimension(:,:,:,:,:) :: x1_displaced

    ! Number of neighbors for each atom in each compound
    integer, dimension(:), intent(in) :: nneigh1

    ! Sigma in the Gaussian kernel
    double precision, dimension(:), intent(in) :: sigmas
    double precision, intent(in) :: lambda

    ! Number of molecules
    integer, intent(in) :: na1

    ! Number of sigmas
    integer, intent(in) :: nsigmas

    double precision, intent(in) :: two_body_power
    double precision, intent(in) :: three_body_power

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
    double precision, dimension(nsigmas,na1), intent(out) :: alphas

    double precision, allocatable, dimension(:,:) :: y
    !DEC$ attributes align: 64:: y

    double precision, allocatable, dimension(:,:,:)  :: kernel_delta
    !DEC$ attributes align: 64:: kernel_delta

    double precision, allocatable, dimension(:,:,:)  :: kernel_scratch
    !DEC$ attributes align: 64:: kernel_scratch

    ! Internal counters
    integer :: i, j, k
    ! integer :: ni, nj
    integer :: a

    ! Temporary variables necessary for parallelization
    double precision :: l2dist

    ! Pre-computed terms in the full distance matrix
    double precision, allocatable, dimension(:) :: self_scalar1
    double precision :: self_scalar1_displaced

    ! Pre-computed terms
    double precision, allocatable, dimension(:,:) :: ksi1
    double precision, allocatable, dimension(:) :: ksi1_displaced

    double precision, allocatable, dimension(:,:,:,:) :: sinp1
    double precision, allocatable, dimension(:,:,:,:) :: cosp1

    double precision, allocatable, dimension(:,:,:,:) :: fourier_displaced

    ! Value of PI at full FORTRAN precision.
    double precision, parameter :: pi = 4.0d0 * atan(1.0d0)

    ! counter for periodic distance
    integer :: pmax1
    ! integer :: nneighi

    integer :: dim1, dim2, dim3
    integer :: xyz, pm
    integer :: info

    double precision :: ang_norm2

    double precision, parameter :: dx = 0.0005d0
    double precision, parameter :: inv_2dx = 1.0d0 / (2.0d0 * dx)
    double precision :: dx_sign

    integer :: maxneigh1

    write (*,*) "INIT"

    maxneigh1 = maxval(nneigh1)
    ang_norm2 = get_angular_norm2(t_width)

    pmax1 = 0
    do a = 1, na1
        pmax1 = max(pmax1, int(maxval(x1(a,2,:nneigh1(a)))))
    enddo

    inv_sigma2(:) = -1.0d0 / (sigmas(:))**2

    write (*,*) "DISPLACED REPS"

    dim1 = size(x1, dim=1)
    dim2 = size(x1, dim=2)
    dim3 = size(x1, dim=3)

    allocate(x1_displaced(dim1, dim2, dim3, 3, 2))

    !$OMP PARALLEL DO
    do i = 1, na1
        x1_displaced(i, :, :, :, :) = &
            & get_displaced_representaions(x1(i,:,:), nneigh1(i), dx, dim2, dim3)
    enddo
    !$OMP END PARALLEL do

    write (*,*) "KSI1"
    allocate(ksi1(na1, maxneigh1))

    ksi1 = 0.0d0

    !$OMP PARALLEL DO
    do i = 1, na1
        ksi1(i, :) = get_twobody_weights(x1(i,:,:), nneigh1(i), &
            & two_body_power, maxneigh1)
    enddo
    !$OMP END PARALLEL do

    write (*,*) "FOURIER"
    allocate(cosp1(na1, pmax1, order, maxneigh1))
    allocate(sinp1(na1, pmax1, order, maxneigh1))

    cosp1 = 0.0d0
    sinp1 = 0.0d0

    !$OMP PARALLEL DO PRIVATE(fourier)
    do i = 1, na1

        fourier = get_threebody_fourier(x1(i,:,:), &
            & nneigh1(i), order, three_body_power, pmax1, order, maxneigh1)

        cosp1(i,:,:,:) = fourier(1,:,:,:)
        sinp1(i,:,:,:) = fourier(2,:,:,:)

    enddo
    !$OMP END PARALLEL DO

    write (*,*) "SELF SCALAR"
    allocate(self_scalar1(na1))

    self_scalar1 = 0.0d0

    write (*,*) "SELF SCALAR"
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

    write (*,*) "ALLOCATE AND CLEAR"
    allocate(kernel_delta(na1,na1,nsigmas))

    allocate(kernel_scratch(na1,na1,nsigmas))
    kernel_scratch = 0.0d0

    allocate(ksi1_displaced(maxneigh1))
    ksi1_displaced = 0.0d0

    allocate(fourier_displaced(2, pmax1, order, maxneigh1))
    fourier_displaced = 0.0d0

    allocate(y(na1,nsigmas))
    y = 0.0d0

    alphas = 0.0d0

    do xyz = 1, 3

        kernel_delta = 0.0d0

        ! Plus/minus displacemenets
        do pm = 1, 2

            ! Get the sign and magnitude of displacement
            dx_sign = ((dble(pm) - 1.5d0) * 2.0d0) * inv_2dx

            write (*,*) "DERIVATIVE", xyz, ((dble(pm) - 1.5d0) * 2.0d0)

            !$OMP PARALLEL DO schedule(dynamic), &
            !$OMP& PRIVATE(l2dist,self_scalar1_displaced,ksi1_displaced,fourier_displaced)
            do i = 1, na1

                ksi1_displaced(:) = &
                    & get_twobody_weights(x1_displaced(i,:,:,xyz,pm), nneigh1(i), &
                    & two_body_power, maxneigh1)

                fourier_displaced(:,:,:,:) = get_threebody_fourier(x1_displaced(i,:,:,xyz,pm), &
                    & nneigh1(i), order, three_body_power, pmax1, order, maxneigh1)

                self_scalar1_displaced = scalar(x1_displaced(i,:,:,xyz,pm), &
                    & x1_displaced(i,:,:,xyz,pm), nneigh1(i), nneigh1(i), &
                    & ksi1_displaced(:), ksi1_displaced(:), &
                    & fourier_displaced(2,:,:,:), fourier_displaced(2,:,:,:), &
                    & fourier_displaced(1,:,:,:), fourier_displaced(1,:,:,:), &
                    & t_width, d_width, cut_distance, order, &
                    & pd, ang_norm2,distance_scale, angular_scale, alchemy)

                do j = 1, na1

                    l2dist = scalar(x1_displaced(i,:,:,xyz,pm), x1(j,:,:), &
                        & nneigh1(i), nneigh1(j), ksi1_displaced(:), ksi1(j,:), &
                        & fourier_displaced(2,:,:,:), sinp1(j,:,:,:), &
                        & fourier_displaced(1,:,:,:), cosp1(j,:,:,:), &
                        & t_width, d_width, cut_distance, order, &
                        & pd, ang_norm2, distance_scale, angular_scale, alchemy)

                    l2dist = self_scalar1_displaced &
                        & + self_scalar1(j) - 2.0d0 * l2dist

                    do k = 1, nsigmas
                        kernel_delta(i,j,k) = kernel_delta(i,j,k) + &
                            & exp(l2dist * inv_sigma2(k)) * dx_sign
                    enddo

                enddo
            enddo
            !$OMP END PARALLEL DO
        enddo

        do k = 1, nsigmas

            write (*,*) "    DSYRK"
            ! DSYRK call corresponds to: C := 1.0 *  K^T * K + 1.0 * C
            call dsyrk("U", "T", na1, na1, 1.0d0, kernel_delta(:,:,k), na1, &
                & 1.0d0, kernel_scratch(:,:,k), na1)

            write (*,*) "    DGEMV"
            ! DGEMV call corresponds to y := 1.0 * K^T * F + 1.0 * y
            call dgemv("T", na1, na1, 1.0d0, kernel_delta(:,:,k), na1, &
                            & forces(:,xyz), 1, 1.0d0, y(:,k), 1)
        enddo

    enddo

    do k = 1, nsigmas
        do i = 1, na1
            kernel_scratch(i,i,k) = kernel_scratch(i,i,k) + lambda
        enddo
    enddo

    do k = 1, nsigmas
        write (*,*) "  DPOTRF"
        call dpotrf("U", na1, kernel_scratch(:,:,k), na1, info)
        if (info > 0) then
            write (*,*) "QML WARNING: Error in LAPACK Cholesky decomposition DPOTRF()."
            write (*,*) "QML WARNING: The", info, "-th leading order is not positive definite."
        else if (info < 0) then
            write (*,*) "QML WARNING: Error in LAPACK Cholesky decomposition DPOTRF()."
            write (*,*) "QML WARNING: The", -info, "-th argument had an illegal value."
        endif

        write (*,*) "  DPOTRS"
        call dpotrs("U", na1, 1, kernel_scratch(:,:,k), na1, y(:,k), na1, info)
        if (info < 0) then
            write (*,*) "QML WARNING: Error in LAPACK Cholesky solver DPOTRS()."
            write (*,*) "QML WARNING: The", -info, "-th argument had an illegal value."
        endif

        alphas(k,:) = y(:,k)
    enddo

    deallocate(kernel_delta)
    deallocate(kernel_scratch)
    deallocate(self_scalar1)
    deallocate(cosp1)
    deallocate(sinp1)
    deallocate(ksi1)
    deallocate(x1_displaced)

end subroutine fget_atomic_force_alphas_fchl


subroutine fget_atomic_force_kernels_fchl(x1, x2, nneigh1, nneigh2, &
       & sigmas, na1, na2, nsigmas, &
       & t_width, d_width, cut_distance, order, pd, &
       & distance_scale, angular_scale, alchemy, two_body_power, three_body_power, kernels)

    use ffchl_module, only: scalar, get_threebody_fourier, get_twobody_weights, &
                        & get_displaced_representaions, get_angular_norm2

    implicit none

    double precision, allocatable, dimension(:,:,:,:) :: fourier

    ! fchl descriptors for the training set, format (na1,maxatoms,5,maxneighbors)
    double precision, dimension(:,:,:), intent(in) :: x1

    ! fchl descriptors for the prediction set, format (na2,maxatoms,5,maxneighbors)
    double precision, dimension(:,:,:), intent(in) :: x2

    double precision, allocatable, dimension(:,:,:,:,:) :: x2_displaced

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

    double precision, intent(in) :: two_body_power
    double precision, intent(in) :: three_body_power

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
    double precision, dimension(nsigmas,3,na2,na1), intent(out) :: kernels
    ! double precision, allocatable, dimension(:,:,:,:)  :: l2_displaced

    ! Internal counters
    integer :: i, j, k
    ! integer :: ni, nj
    integer :: a

    ! Temporary variables necessary for parallelization
    double precision :: l2dist

    ! Pre-computed terms in the full distance matrix
    double precision, allocatable, dimension(:) :: self_scalar1
    double precision :: self_scalar2_displaced

    ! Pre-computed terms
    double precision, allocatable, dimension(:,:) :: ksi1
    double precision, allocatable, dimension(:) :: ksi2_displaced

    double precision, allocatable, dimension(:,:,:,:) :: sinp1
    double precision, allocatable, dimension(:,:,:,:) :: cosp1

    double precision, allocatable, dimension(:,:,:,:) :: fourier_displaced

    ! Value of PI at full FORTRAN precision.
    double precision, parameter :: pi = 4.0d0 * atan(1.0d0)

    ! counter for periodic distance
    integer :: pmax1
    integer :: pmax2
    ! integer :: nneighi

    integer :: dim1, dim2, dim3
    integer :: xyz, pm

    double precision :: ang_norm2

    double precision, parameter :: dx = 0.0005d0
    double precision, parameter :: inv_2dx = 1.0d0 / (2.0d0 * dx)
    double precision :: dx_sign

    integer :: maxneigh1
    integer :: maxneigh2

    write (*,*) "INIT"

    write (*,*) "CLEARING KERNEL MEM"
    kernels = 0.0d0



    maxneigh1 = maxval(nneigh1(:))
    maxneigh2 = maxval(nneigh2(:))
    ang_norm2 = get_angular_norm2(t_width)

    pmax1 = 0
    do a = 1, na1
        pmax1 = max(pmax1, int(maxval(x1(a,2,:nneigh1(a)))))
    enddo

    pmax2 = 0
    do a = 1, na2
        pmax2 = max(pmax2, int(maxval(x2(a,2,:nneigh2(a)))))
    enddo

    inv_sigma2(:) = -1.0d0 / (sigmas(:))**2

    write (*,*) "DISPLACED REPS"

    dim1 = size(x2, dim=1)
    dim2 = size(x2, dim=2)
    dim3 = size(x2, dim=3)

    allocate(x2_displaced(dim1, dim2, dim3, 3, 2))

    !$OMP PARALLEL DO
    do i = 1, na2
        x2_displaced(i, :, :, :, :) = &
            & get_displaced_representaions(x2(i,:,:), nneigh2(i), dx, dim2, dim3)
    enddo
    !$OMP END PARALLEL do

    write (*,*) "KSI1"
    allocate(ksi1(na1, maxneigh1))

    ksi1 = 0.0d0

    !$OMP PARALLEL DO
    do i = 1, na1
        ksi1(i, :) = get_twobody_weights(x1(i,:,:), nneigh1(i), &
            & two_body_power, maxneigh1)
    enddo
    !$OMP END PARALLEL do

    write (*,*) "FOURIER"
    allocate(cosp1(na1, pmax1, order, maxneigh1))
    allocate(sinp1(na1, pmax1, order, maxneigh1))

    cosp1 = 0.0d0
    sinp1 = 0.0d0

    !$OMP PARALLEL DO PRIVATE(fourier)
    do i = 1, na1

        fourier = get_threebody_fourier(x1(i,:,:), &
            & nneigh1(i), order, three_body_power, pmax1, order, maxneigh1)

        cosp1(i,:,:,:) = fourier(1,:,:,:)
        sinp1(i,:,:,:) = fourier(2,:,:,:)

    enddo
    !$OMP END PARALLEL DO


    write (*,*) "SELF SCALAR"
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


    allocate(ksi2_displaced(maxneigh2))
    allocate(fourier_displaced(2, pmax2, order, maxneigh2))
    ksi2_displaced = 0.0d0
    fourier_displaced = 0.0d0

    write (*,*) "KERNEL DERIVATIVES"
    do pm = 1, 2

        ! Get the sign and magnitude of displacement
        dx_sign = ((dble(pm) - 1.5d0) * 2.0d0) * inv_2dx

        !$OMP PARALLEL DO schedule(dynamic), &
        !$OMP& PRIVATE(l2dist,self_scalar2_displaced,ksi2_displaced,fourier_displaced)
        do i = 1, na2
           do xyz = 1, 3

                ksi2_displaced(:) = &
                    & get_twobody_weights(x2_displaced(i,:,:,xyz,pm), nneigh2(i), &
                    & two_body_power, maxneigh2)

                fourier_displaced(:,:,:,:) = get_threebody_fourier(x2_displaced(i,:,:,xyz,pm), &
                    & nneigh2(i), order, three_body_power, pmax2, order, maxneigh2)

                self_scalar2_displaced = scalar(x2_displaced(i,:,:,xyz,pm), &
                    & x2_displaced(i,:,:,xyz,pm), nneigh2(i), nneigh2(i), &
                    & ksi2_displaced(:), ksi2_displaced(:), &
                    & fourier_displaced(2,:,:,:), fourier_displaced(2,:,:,:), &
                    & fourier_displaced(1,:,:,:), fourier_displaced(1,:,:,:), &
                    & t_width, d_width, cut_distance, order, &
                    & pd, ang_norm2,distance_scale, angular_scale, alchemy)

                do j = 1, na1

                    l2dist = scalar(x2_displaced(i,:,:,xyz,pm), x1(j,:,:), &
                        & nneigh2(i), nneigh1(j), ksi2_displaced(:), ksi1(j,:), &
                        & fourier_displaced(2,:,:,:), sinp1(j,:,:,:), &
                        & fourier_displaced(1,:,:,:), cosp1(j,:,:,:), &
                        & t_width, d_width, cut_distance, order, &
                        & pd, ang_norm2, distance_scale, angular_scale, alchemy)

                    l2dist = self_scalar2_displaced &
                        & + self_scalar1(j) - 2.0d0 * l2dist

                    ! write (*,*) "l2 old", i,j,nneigh2(i), nneigh1(j), self_scalar2_displaced, self_scalar1(j), l2dist
                    do k = 1, nsigmas
                        kernels(k,xyz,i,j) = kernels(k,xyz,i,j) + &
                            & exp(l2dist * inv_sigma2(k)) * dx_sign
                    enddo

                enddo
            enddo
        enddo
        !$OMP END PARALLEL DO
    enddo

    deallocate(self_scalar1)
    deallocate(cosp1)
    deallocate(sinp1)
    deallocate(ksi1)
    deallocate(x2_displaced)

end subroutine fget_atomic_force_kernels_fchl


subroutine fget_scalar_vector_alphas_fchl(x1, forces, energies, nneigh1, &
       & sigmas, lambda, nm1, na1, n1, nsigmas, &
       & t_width, d_width, cut_distance, order, pd, &
       & distance_scale, angular_scale, alchemy, two_body_power, three_body_power, alphas)

    use ffchl_module, only: scalar, get_threebody_fourier, get_twobody_weights, &
                        & get_displaced_representaions, get_angular_norm2

    implicit none

    double precision, allocatable, dimension(:,:,:,:) :: fourier

    ! fchl descriptors for the training set, format (i,maxatoms,5,maxneighbors)
    double precision, dimension(:,:,:), intent(in) :: x1
    double precision, dimension(:,:), intent(in) :: forces
    double precision, dimension(:), intent(in) :: energies

    double precision, allocatable, dimension(:,:,:,:,:) :: x1_displaced

    ! Number of neighbors for each atom in each compound
    integer, dimension(:), intent(in) :: nneigh1

    ! Sigma in the Gaussian kernel
    double precision, dimension(:), intent(in) :: sigmas
    double precision, intent(in) :: lambda

    ! Number of molecules
    integer, intent(in) :: nm1

    ! Number of atoms
    integer, intent(in) :: na1

    ! Number of atoms
    integer, dimension(:), intent(in) :: n1

    ! Number of sigmas
    integer, intent(in) :: nsigmas

    double precision, intent(in) :: two_body_power
    double precision, intent(in) :: three_body_power

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
    double precision, dimension(nsigmas,na1), intent(out) :: alphas

    double precision, allocatable, dimension(:,:) :: y
    !DEC$ attributes align: 64:: y

    double precision, allocatable, dimension(:,:,:)  :: kernel_delta
    !DEC$ attributes align: 64:: kernel_delta

    double precision, allocatable, dimension(:,:,:)  :: kernel_scratch
    !DEC$ attributes align: 64:: kernel_scratch

    ! Internal counters
    integer :: i, j, k
    ! integer :: ni, nj
    integer :: a! , b

    ! Temporary variables necessary for parallelization
    double precision :: l2dist
    ! double precision, allocatable, dimension(:,:) :: atomic_distance

    ! Pre-computed terms in the full distance matrix
    double precision, allocatable, dimension(:) :: self_scalar1
    double precision :: self_scalar1_displaced

    ! Pre-computed terms
    double precision, allocatable, dimension(:,:) :: ksi1
    double precision, allocatable, dimension(:) :: ksi1_displaced

    double precision, allocatable, dimension(:,:,:,:) :: sinp1
    double precision, allocatable, dimension(:,:,:,:) :: cosp1

    double precision, allocatable, dimension(:,:,:,:) :: fourier_displaced

    ! Value of PI at full FORTRAN precision.
    double precision, parameter :: pi = 4.0d0 * atan(1.0d0)

    ! counter for periodic distance
    integer :: pmax1
    ! integer :: nneighi

    integer :: dim1, dim2, dim3
    integer :: xyz, pm
    integer :: info

    double precision :: ang_norm2

    double precision, parameter :: dx = 0.0005d0
    double precision, parameter :: inv_2dx = 1.0d0 / (2.0d0 * dx)
    double precision :: dx_sign

    integer :: maxneigh1

    integer, dimension(nm1) :: istart, iend
    double precision, dimension(na1) :: e
    double precision, dimension(nm1) :: e_scratch
    double precision, allocatable, dimension(:,:,:) :: kernel_molecular
    double precision, allocatable, dimension(:,:) :: kernel_molecular_scratch

    write (*,*) "INIT"

    maxneigh1 = maxval(nneigh1)
    ang_norm2 = get_angular_norm2(t_width)

    pmax1 = 0
    do a = 1, na1
        pmax1 = max(pmax1, int(maxval(x1(a,2,:nneigh1(a)))))
    enddo

    inv_sigma2(:) = -1.0d0 / (sigmas(:))**2

    write (*,*) "DISPLACED REPS"

    dim1 = size(x1, dim=1)
    dim2 = size(x1, dim=2)
    dim3 = size(x1, dim=3)

    allocate(x1_displaced(dim1, dim2, dim3, 3, 2))

    !$OMP PARALLEL DO
    do i = 1, na1
        x1_displaced(i, :, :, :, :) = &
            & get_displaced_representaions(x1(i,:,:), nneigh1(i), dx, dim2, dim3)
    enddo
    !$OMP END PARALLEL do

    write (*,*) "KSI1"
    allocate(ksi1(na1, maxneigh1))

    ksi1 = 0.0d0

    !$OMP PARALLEL DO
    do i = 1, na1
        ksi1(i, :) = get_twobody_weights(x1(i,:,:), nneigh1(i), &
            & two_body_power, maxneigh1)
    enddo
    !$OMP END PARALLEL do

    write (*,*) "FOURIER"
    allocate(cosp1(na1, pmax1, order, maxneigh1))
    allocate(sinp1(na1, pmax1, order, maxneigh1))

    cosp1 = 0.0d0
    sinp1 = 0.0d0

    !$OMP PARALLEL DO PRIVATE(fourier)
    do i = 1, na1

        fourier = get_threebody_fourier(x1(i,:,:), &
            & nneigh1(i), order, three_body_power, pmax1, order, maxneigh1)

        cosp1(i,:,:,:) = fourier(1,:,:,:)
        sinp1(i,:,:,:) = fourier(2,:,:,:)

    enddo
    !$OMP END PARALLEL DO

    write (*,*) "SELF SCALAR"
    allocate(self_scalar1(na1))

    self_scalar1 = 0.0d0

    write (*,*) "SELF SCALAR"
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

    write (*,*) "ALLOCATE AND CLEAR"
    allocate(kernel_delta(na1,na1,nsigmas))

    allocate(kernel_scratch(na1,na1,nsigmas))
    kernel_scratch = 0.0d0

    allocate(ksi1_displaced(maxneigh1))
    ksi1_displaced = 0.0d0

    allocate(fourier_displaced(2, pmax1, order, maxneigh1))
    fourier_displaced = 0.0d0

    allocate(y(na1,nsigmas))
    y = 0.0d0

    alphas = 0.0d0

    do xyz = 1, 3

        kernel_delta = 0.0d0


        ! Plus/minus displacemenets
        do pm = 1, 2

            ! Get the sign and magnitude of displacement
            dx_sign = ((dble(pm) - 1.5d0) * 2.0d0) * inv_2dx

            write (*,*) "DERIVATIVE", xyz, nint(sign(1.0d0, dx_sign))

            !$OMP PARALLEL DO schedule(dynamic), &
            !$OMP& PRIVATE(l2dist,self_scalar1_displaced,ksi1_displaced,fourier_displaced)
            do i = 1, na1

                ksi1_displaced(:) = &
                    & get_twobody_weights(x1_displaced(i,:,:,xyz,pm), nneigh1(i), &
                    & two_body_power, maxneigh1)

                fourier_displaced(:,:,:,:) = get_threebody_fourier(x1_displaced(i,:,:,xyz,pm), &
                    & nneigh1(i), order, three_body_power, pmax1, order, maxneigh1)

                self_scalar1_displaced = scalar(x1_displaced(i,:,:,xyz,pm), &
                    & x1_displaced(i,:,:,xyz,pm), nneigh1(i), nneigh1(i), &
                    & ksi1_displaced(:), ksi1_displaced(:), &
                    & fourier_displaced(2,:,:,:), fourier_displaced(2,:,:,:), &
                    & fourier_displaced(1,:,:,:), fourier_displaced(1,:,:,:), &
                    & t_width, d_width, cut_distance, order, &
                    & pd, ang_norm2,distance_scale, angular_scale, alchemy)

                do j = 1, na1

                    l2dist = scalar(x1_displaced(i,:,:,xyz,pm), x1(j,:,:), &
                        & nneigh1(i), nneigh1(j), ksi1_displaced(:), ksi1(j,:), &
                        & fourier_displaced(2,:,:,:), sinp1(j,:,:,:), &
                        & fourier_displaced(1,:,:,:), cosp1(j,:,:,:), &
                        & t_width, d_width, cut_distance, order, &
                        & pd, ang_norm2, distance_scale, angular_scale, alchemy)

                    l2dist = self_scalar1_displaced &
                        & + self_scalar1(j) - 2.0d0 * l2dist

                    do k = 1, nsigmas
                        kernel_delta(i,j,k) = kernel_delta(i,j,k) + &
                            & exp(l2dist * inv_sigma2(k)) * dx_sign
                    enddo

                enddo
            enddo
            !$OMP END PARALLEL DO
        enddo

        do k = 1, nsigmas
            write (*,*) "    DSYRK", sigmas(k)
            ! DSYRK call corresponds to: C := 1.0 *  K^T * K + 1.0 * C
            call dsyrk("U", "T", na1, na1, 1.0d0, kernel_delta(:,:,k), na1, &
                & 1.0d0, kernel_scratch(:,:,k), na1)

            write (*,*) "    DGEMV", sigmas(k)
            ! DGEMV call corresponds to alphas := 1.0 * K^T * F + 1.0 * alphas
            call dgemv("T", na1, na1, 1.0d0, kernel_delta(:,:,k), na1, &
                            & forces(:,xyz), 1, 1.0d0, y(:,k), 1)

        enddo

    enddo

    kernel_delta = 0.0d0

    write (*,*) "NORMAL KERNEL"
    !$OMP PARALLEL DO schedule(dynamic) PRIVATE(l2dist)
    do j = 1, na1
        do i = j, na1
            l2dist = scalar(x1(i,:,:), x1(j,:,:), &
                & nneigh1(i), nneigh1(j), ksi1(i,:), ksi1(j,:), &
                & sinp1(i,:,:,:), sinp1(j,:,:,:), &
                & cosp1(i,:,:,:), cosp1(j,:,:,:), &
                & t_width, d_width, cut_distance, order, &
                & pd, ang_norm2, distance_scale, angular_scale, alchemy)

            l2dist = self_scalar1(i) + self_scalar1(j) - 2.0d0 * l2dist

            do k = 1, nsigmas
                kernel_delta(i, j, k) =  exp(l2dist * inv_sigma2(k))
                kernel_delta(j, i, k) =  kernel_delta(i, j, k)
            enddo
        enddo
    enddo
    !$OMP END PARALLEL DO


    istart(:) = 0
    iend(:) = 0

    istart(1) = 1
    iend(1) = n1(1)
    do i = 2, nm1
        istart(i) = istart(i-1) + n1(i)
        iend(i) = iend(i-1) + n1(i)
        write(*,*) i, istart(i), iend(i)
    enddo

    do i = 1, nm1
        e(istart(i):iend(i)) = energies(i) / n1(i)
    enddo

    ! ! explode first
    ! do k = 1, nsigmas
    !     do j = 1, nm1
    !         do i = 1, nm1

    !             l2dist = sum(kernel_delta(istart(i):iend(i),istart(j):iend(j),k))
    !             kernel_delta(istart(i):iend(i),istart(j):iend(j),k) = l2dist
    !         enddo
    !     enddo
    ! enddo

    ! do k = 1, nsigmas
    !     write (*,*) "    DSYRK", sigmas(k)
    !     ! DSYRK call corresponds to: C := 1.0 *  K^T * K + 1.0 * C
    !     call dsyrk("U", "T", na1, na1, 1.0d0, kernel_delta(:,:,k), na1, &
    !         & 1.0d0, kernel_scratch(:,:,k), na1)

    !     write (*,*) "    DGEMV", sigmas(k)
    !     ! DGEMV call corresponds to alphas := 1.0 * K^T * F + 1.0 * alphas
    !     call dgemv("T", na1, na1, 1.0d0, kernel_delta(:,:,k), na1, &
    !                     & e(:), 1, 1.0d0, y(:,k), 1)

    ! enddo

    
    
    
    ! Multiply first
    allocate(kernel_molecular(nm1,nm1,k))
    allocate(kernel_molecular_scratch(nm1,nm1))

    kernel_molecular = 0.0d0

    do k = 1, nsigmas
        do j = 1, nm1
            do i = 1, nm1

                l2dist = sum(kernel_delta(istart(i):iend(i), istart(j):iend(j), k))
                kernel_molecular(i, j, k) = l2dist
                kernel_molecular(j, i, k) = l2dist

                ! write(*,*) i, j, istart(i), iend(i), istart(j), iend(j)
            enddo
            write(*,*) kernel_molecular(j,:6,k)
        enddo
    enddo
    
    write(*,*) "Y"
    write(*,*) y(:38,1)

    e(:nm1) = energies(:nm1) / n1(:nm1)

    do k = 1, nsigmas

        kernel_molecular_scratch = 0.0d0
        e_scratch = 0.0d0

        ! DGEMM call corresponds to: C := 1.0 *  K^T * K + 1.0 * C
        call dgemm("t", "n", nm1, nm1, nm1, 1.0d0, kernel_molecular(:,:,k), nm1, &
            & kernel_molecular(:,:,k), nm1, 0.0d0, kernel_molecular_scratch(:,:), nm1)

        do j = 1, nm1
            do i = 1, nm1
                kernel_scratch(istart(i):iend(i), istart(j):iend(j), k) &
                    & = kernel_scratch(istart(i):iend(i), istart(j):iend(j), k) &
                    & + kernel_molecular_scratch(i,j)
            enddo
        enddo

        write (*,*) "    DGEMV"
        ! DGEMV call corresponds to alphas := 1.0 * K^T * F + 1.0 * alphas
        call dgemv("T", nm1, nm1, 1.0d0, kernel_molecular(:,:,k), nm1, &
                        & e(:), 1, 0.0d0, e_scratch(:), 1)
                        !& energies(:), 1, 0.0d0, e_scratch(:), 1)

        do i = 1, nm1
            y(istart(i):iend(i),k) = y(istart(i):iend(i),k) + e_scratch(i)
        enddo
    enddo

    write(*,*) "Y"
    write(*,*) y(:38,1)

    deallocate(kernel_molecular)
    deallocate(kernel_molecular_scratch)

    deallocate(kernel_delta)
    deallocate(self_scalar1)
    deallocate(cosp1)
    deallocate(sinp1)
    deallocate(ksi1)
    deallocate(ksi1_displaced)
    deallocate(fourier_displaced)
    deallocate(x1_displaced)

    do k = 1, nsigmas
        do i = 1, na1
            kernel_scratch(i,i,k) = kernel_scratch(i,i,k) + lambda
        enddo
    enddo

    do k = 1, nsigmas
        write (*,*) "  DPOTRF", sigmas(k)
        call dpotrf("U", na1, kernel_scratch(:,:,k), na1, info)
        if (info > 0) then
            write (*,*) "QML WARNING: Error in LAPACK Cholesky decomposition DPOTRF()."
            write (*,*) "QML WARNING: The", info, "-th leading order is not positive definite."
        else if (info < 0) then
            write (*,*) "QML WARNING: Error in LAPACK Cholesky decomposition DPOTRF()."
            write (*,*) "QML WARNING: The", -info, "-th argument had an illegal value."
        endif

        write (*,*) "  DPOTRS", sigmas(k)
        call dpotrs("U", na1, 1, kernel_scratch(:,:,k), na1, y(:,k), na1, info)
        if (info < 0) then
            write (*,*) "QML WARNING: Error in LAPACK Cholesky solver DPOTRS()."
            write (*,*) "QML WARNING: The", -info, "-th argument had an illegal value."
        endif

        alphas(k,:) = y(:,k)
    enddo

    deallocate(kernel_scratch)
    deallocate(y)

end subroutine fget_scalar_vector_alphas_fchl


subroutine fget_scalar_vector_kernels_fchl(x1, x2, nneigh1, nneigh2, &
       & sigmas, na1, na2, nm1, nm2, n1, n2, nsigmas, &
       & t_width, d_width, cut_distance, order, pd, &
       & distance_scale, angular_scale, alchemy, two_body_power, three_body_power, kernels)

    use ffchl_module, only: scalar, get_threebody_fourier, get_twobody_weights, &
                        & get_displaced_representaions, get_angular_norm2

    implicit none

    double precision, allocatable, dimension(:,:,:,:) :: fourier

    ! fchl descriptors for the training set, format (na1,maxatoms,5,maxneighbors)
    double precision, dimension(:,:,:), intent(in) :: x1

    ! fchl descriptors for the prediction set, format (na2,maxatoms,5,maxneighbors)
    double precision, dimension(:,:,:), intent(in) :: x2

    double precision, allocatable, dimension(:,:,:,:,:) :: x2_displaced

    ! Number of neighbors for each atom in each compound
    integer, dimension(:), intent(in) :: nneigh1
    integer, dimension(:), intent(in) :: nneigh2

    ! Sigma in the Gaussian kernel
    double precision, dimension(:), intent(in) :: sigmas

    ! Number of molecules
    integer, intent(in) :: na1
    integer, intent(in) :: na2
    
    ! Number of molecules
    integer, intent(in) :: nm1
    integer, intent(in) :: nm2
    
    ! Number of molecules
    integer, dimension(:), intent(in) :: n1
    integer, dimension(:), intent(in) :: n2

    ! Number of sigmas
    integer, intent(in) :: nsigmas

    double precision, intent(in) :: two_body_power
    double precision, intent(in) :: three_body_power

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
    double precision, dimension(nsigmas,3,na2,na1), intent(out) :: kernels
    ! double precision, allocatable, dimension(:,:,:,:)  :: l2_displaced

    ! Internal counters
    integer :: i, j, k
    ! integer :: ni, nj
    integer :: a

    ! Temporary variables necessary for parallelization
    double precision :: l2dist

    ! Pre-computed terms in the full distance matrix
    double precision, allocatable, dimension(:) :: self_scalar1
    double precision :: self_scalar2_displaced

    ! Pre-computed terms
    double precision, allocatable, dimension(:,:) :: ksi1
    double precision, allocatable, dimension(:) :: ksi2_displaced

    double precision, allocatable, dimension(:,:,:,:) :: sinp1
    double precision, allocatable, dimension(:,:,:,:) :: cosp1

    double precision, allocatable, dimension(:,:,:,:) :: fourier_displaced

    ! Value of PI at full FORTRAN precision.
    double precision, parameter :: pi = 4.0d0 * atan(1.0d0)

    ! counter for periodic distance
    integer :: pmax1
    integer :: pmax2
    ! integer :: nneighi

    integer :: dim1, dim2, dim3
    integer :: xyz, pm

    double precision :: ang_norm2

    double precision, parameter :: dx = 0.0005d0
    double precision, parameter :: inv_2dx = 1.0d0 / (2.0d0 * dx)
    double precision :: dx_sign

    integer :: maxneigh1
    integer :: maxneigh2

    write (*,*) "INIT"

    write (*,*) "CLEARING KERNEL MEM"
    kernels = 0.0d0

    maxneigh1 = maxval(nneigh1(:))
    maxneigh2 = maxval(nneigh2(:))
    ang_norm2 = get_angular_norm2(t_width)

    pmax1 = 0
    do a = 1, na1
        pmax1 = max(pmax1, int(maxval(x1(a,2,:nneigh1(a)))))
    enddo

    pmax2 = 0
    do a = 1, na2
        pmax2 = max(pmax2, int(maxval(x2(a,2,:nneigh2(a)))))
    enddo

    inv_sigma2(:) = -1.0d0 / (sigmas(:))**2

    write (*,*) "DISPLACED REPS"

    dim1 = size(x2, dim=1)
    dim2 = size(x2, dim=2)
    dim3 = size(x2, dim=3)

    allocate(x2_displaced(dim1, dim2, dim3, 3, 2))

    !$OMP PARALLEL DO
    do i = 1, na2
        x2_displaced(i, :, :, :, :) = &
            & get_displaced_representaions(x2(i,:,:), nneigh2(i), dx, dim2, dim3)
    enddo
    !$OMP END PARALLEL do

    write (*,*) "KSI1"
    allocate(ksi1(na1, maxneigh1))

    ksi1 = 0.0d0

    !$OMP PARALLEL DO
    do i = 1, na1
        ksi1(i, :) = get_twobody_weights(x1(i,:,:), nneigh1(i), &
            & two_body_power, maxneigh1)
    enddo
    !$OMP END PARALLEL do

    write (*,*) "FOURIER"
    allocate(cosp1(na1, pmax1, order, maxneigh1))
    allocate(sinp1(na1, pmax1, order, maxneigh1))

    cosp1 = 0.0d0
    sinp1 = 0.0d0

    !$OMP PARALLEL DO PRIVATE(fourier)
    do i = 1, na1

        fourier = get_threebody_fourier(x1(i,:,:), &
            & nneigh1(i), order, three_body_power, pmax1, order, maxneigh1)

        cosp1(i,:,:,:) = fourier(1,:,:,:)
        sinp1(i,:,:,:) = fourier(2,:,:,:)

    enddo
    !$OMP END PARALLEL DO


    write (*,*) "SELF SCALAR"
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


    allocate(ksi2_displaced(maxneigh2))
    allocate(fourier_displaced(2, pmax2, order, maxneigh2))
    ksi2_displaced = 0.0d0
    fourier_displaced = 0.0d0

    write (*,*) "KERNEL DERIVATIVES"
    do pm = 1, 2

        ! Get the sign and magnitude of displacement
        dx_sign = ((dble(pm) - 1.5d0) * 2.0d0) * inv_2dx

        !$OMP PARALLEL DO schedule(dynamic), &
        !$OMP& PRIVATE(l2dist,self_scalar2_displaced,ksi2_displaced,fourier_displaced)
        do i = 1, na2
           do xyz = 1, 3

                ksi2_displaced(:) = &
                    & get_twobody_weights(x2_displaced(i,:,:,xyz,pm), nneigh2(i), &
                    & two_body_power, maxneigh2)

                fourier_displaced(:,:,:,:) = get_threebody_fourier(x2_displaced(i,:,:,xyz,pm), &
                    & nneigh2(i), order, three_body_power, pmax2, order, maxneigh2)

                self_scalar2_displaced = scalar(x2_displaced(i,:,:,xyz,pm), &
                    & x2_displaced(i,:,:,xyz,pm), nneigh2(i), nneigh2(i), &
                    & ksi2_displaced(:), ksi2_displaced(:), &
                    & fourier_displaced(2,:,:,:), fourier_displaced(2,:,:,:), &
                    & fourier_displaced(1,:,:,:), fourier_displaced(1,:,:,:), &
                    & t_width, d_width, cut_distance, order, &
                    & pd, ang_norm2,distance_scale, angular_scale, alchemy)

                do j = 1, na1

                    l2dist = scalar(x2_displaced(i,:,:,xyz,pm), x1(j,:,:), &
                        & nneigh2(i), nneigh1(j), ksi2_displaced(:), ksi1(j,:), &
                        & fourier_displaced(2,:,:,:), sinp1(j,:,:,:), &
                        & fourier_displaced(1,:,:,:), cosp1(j,:,:,:), &
                        & t_width, d_width, cut_distance, order, &
                        & pd, ang_norm2, distance_scale, angular_scale, alchemy)

                    l2dist = self_scalar2_displaced &
                        & + self_scalar1(j) - 2.0d0 * l2dist

                    ! write (*,*) "l2 old", i,j,nneigh2(i), nneigh1(j), self_scalar2_displaced, self_scalar1(j), l2dist
                    do k = 1, nsigmas
                        kernels(k,xyz,i,j) = kernels(k,xyz,i,j) + &
                            & exp(l2dist * inv_sigma2(k)) * dx_sign
                    enddo

                enddo
            enddo
        enddo
        !$OMP END PARALLEL DO
    enddo

    deallocate(self_scalar1)
    deallocate(cosp1)
    deallocate(sinp1)
    deallocate(ksi1)
    deallocate(x2_displaced)

end subroutine fget_scalar_vector_kernels_fchl
