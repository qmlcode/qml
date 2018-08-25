subroutine fget_ef_gaussian_process_kernels_fchl(x1, x2, verbose, f1, f2, n1, n2, nneigh1, nneigh2, &
        & nm1, nm2, nsigmas, t_width, d_width, cut_start, cut_distance, order, pd, &
        & distance_scale, angular_scale, alchemy, two_body_power, three_body_power, ef_scale,&
        & df, kernel_idx, parameters, kernels)

    use ffchl_module, only: scalar, get_angular_norm2, get_pmax, get_ksi_ef_field, init_cosp_sinp_ef_field, &
        & get_selfscalar, get_ksi_ef, init_cosp_sinp_ef
    
    use ffchl_kernels, only: kernel

    use omp_lib, only: omp_get_wtime

    implicit none

    ! fchl descriptors for the training set, format (i,maxatoms,5,maxneighbors)
    double precision, dimension(:,:,:,:), intent(in) :: x1
    double precision, dimension(:,:,:,:), intent(in) :: x2

    ! Display output
    logical, intent(in) :: verbose

    ! List of numbers of atoms in each molecule
    integer, dimension(:), intent(in) :: n1
    integer, dimension(:), intent(in) :: n2
    
    ! Electric field perturbations for each molecule
    double precision, dimension(:,:), intent(in) :: f1
    double precision, dimension(:,:), intent(in) :: f2

    ! Number of molecules
    integer, intent(in) :: nm1
    integer, intent(in) :: nm2

    ! Number of sigmas
    integer, intent(in) :: nsigmas

    ! Number of neighbors for each atom in each compound
    integer, dimension(:,:), intent(in) :: nneigh1
    integer, dimension(:,:), intent(in) :: nneigh2

    ! Angular Gaussian width
    double precision, intent(in) :: t_width

    ! Distance Gaussian width
    double precision, intent(in) :: d_width

    ! Fraction of cut_distance at which cut-off starts
    double precision, intent(in) :: cut_start
    double precision, intent(in) :: cut_distance
    
    ! Truncation order for Fourier terms
    integer, intent(in) :: order

    ! Periodic table distance matrix
    double precision, dimension(:,:), intent(in) :: pd

    ! Scaling for angular and distance terms
    double precision, intent(in) :: distance_scale
    double precision, intent(in) :: angular_scale

    ! Switch alchemy on or off
    logical, intent(in) :: alchemy

    ! Decaying power laws for two- and three-body terms
    double precision, intent(in) :: two_body_power
    double precision, intent(in) :: three_body_power


    double precision, intent(in) :: ef_scale
    double precision, intent(in) :: df

    ! Kernel ID and corresponding parameters
    integer, intent(in) :: kernel_idx
    double precision, dimension(:,:), intent(in) :: parameters

    ! Resulting alpha vector
    double precision, dimension(nsigmas,4*nm1,4*nm2), intent(out) :: kernels

    ! Internal counters
    integer :: i, j
    integer :: ni, nj
    integer :: a, b
    integer :: xyz, pm
    integer :: xyz1, pm1
    integer :: xyz2, pm2
    integer :: idx_a
    integer :: idx_b

    ! Temporary variables necessary for parallelization
    double precision :: s12
    
    ! Pre-computed terms in the full distance matrix
    double precision, allocatable, dimension(:,:) :: self_scalar1
    double precision, allocatable, dimension(:,:) :: self_scalar2
    double precision, allocatable, dimension(:,:,:,:) :: self_scalar1_ef
    double precision, allocatable, dimension(:,:,:,:) :: self_scalar2_ef
    
    ! Pre-computed two-body weights for nummerical differentation of electric field
    double precision, allocatable, dimension(:,:,:) :: ksi1
    double precision, allocatable, dimension(:,:,:) :: ksi2
    double precision, allocatable, dimension(:,:,:,:,:) :: ksi1_ef
    double precision, allocatable, dimension(:,:,:,:,:) :: ksi2_ef

    ! Pre-computed terms for the Fourier expansion of the three-body term
    double precision, allocatable, dimension(:,:,:,:,:) :: sinp1
    double precision, allocatable, dimension(:,:,:,:,:) :: cosp1
    double precision, allocatable, dimension(:,:,:,:,:) :: sinp2
    double precision, allocatable, dimension(:,:,:,:,:) :: cosp2
    double precision, allocatable, dimension(:,:,:,:,:,:,:) :: sinp1_ef
    double precision, allocatable, dimension(:,:,:,:,:,:,:) :: cosp1_ef
    double precision, allocatable, dimension(:,:,:,:,:,:,:) :: sinp2_ef
    double precision, allocatable, dimension(:,:,:,:,:,:,:) :: cosp2_ef

    ! Max index in the periodic table
    integer :: pmax1
    integer :: pmax2

    ! Angular normalization constant
    double precision :: ang_norm2
   
    ! Max number of neighbors 
    integer :: maxneigh1
    integer :: maxneigh2

    ! Variables to calculate time 
    double precision :: t_start, t_end
    
    if (verbose) write (*,*) "CLEARING KERNEL MEM"
    kernels(:,:,:) = 0.0d0

    ! Get max number of neighbors
    maxneigh1 = maxval(nneigh1)
    maxneigh2 = maxval(nneigh2)

    ! Calculate angular normalization constant
    ang_norm2 = get_angular_norm2(t_width)

    ! pmax = max nuclear charge
    pmax1 = get_pmax(x1, n1)
    pmax2 = get_pmax(x2, n2)

    ! Get two-body weight function
    ksi1 = get_ksi_ef_field(x1, n1, nneigh1, two_body_power, cut_start, cut_distance, &
                & f1, ef_scale, verbose)
    ksi1_ef = get_ksi_ef(x1, n1, nneigh1, two_body_power, cut_start, cut_distance, ef_scale, df, verbose)
    
    ! Get two-body weight function
    ksi2 = get_ksi_ef_field(x2, n2, nneigh2, two_body_power, cut_start, cut_distance, &
                & f2, ef_scale, verbose)
    ksi2_ef = get_ksi_ef(x2, n2, nneigh2, two_body_power, cut_start, cut_distance, ef_scale, df, verbose)
   
    ! Allocate three-body Fourier terms
    allocate(cosp1(nm1, maxval(n1), pmax1, order, maxneigh1))
    allocate(sinp1(nm1, maxval(n1), pmax1, order, maxneigh1))

    ! Initialize and pre-calculate three-body Fourier terms
    call init_cosp_sinp_ef_field(x1, n1, nneigh1, three_body_power, order, cut_start,  cut_distance, &
        & cosp1, sinp1, f1, ef_scale, verbose)
    
    ! Allocate three-body Fourier terms
    allocate(cosp2(nm2, maxval(n2), pmax2, order, maxneigh2))
    allocate(sinp2(nm2, maxval(n2), pmax2, order, maxneigh2))

    ! Initialize and pre-calculate three-body Fourier terms
    call init_cosp_sinp_ef_field(x2, n2, nneigh2, three_body_power, order, cut_start,  cut_distance, &
        & cosp2, sinp2, f2, ef_scale, verbose)

    ! Allocate three-body Fourier terms
    allocate(cosp1_ef(nm1, 3, 2, maxval(n1), pmax1, order, maxneigh1))
    allocate(sinp1_ef(nm1, 3, 2, maxval(n1), pmax1, order, maxneigh1))
    
    ! Initialize and pre-calculate three-body Fourier terms
    call init_cosp_sinp_ef(x1, n1, nneigh1, three_body_power, order, cut_start, cut_distance, &
        & cosp1_ef, sinp1_ef, ef_scale, df, verbose)
    
    ! Allocate three-body Fourier terms
    allocate(cosp2_ef(nm2, 3, 2, maxval(n2), pmax2, order, maxneigh2))
    allocate(sinp2_ef(nm2, 3, 2, maxval(n2), pmax2, order, maxneigh2))
    
    ! Initialize and pre-calculate three-body Fourier terms
    call init_cosp_sinp_ef(x2, n2, nneigh2, three_body_power, order, cut_start, cut_distance, &
        & cosp2_ef, sinp2_ef, ef_scale, df, verbose)

    ! Pre-calculate self-scalar terms 
    self_scalar1 = get_selfscalar(x1, nm1, n1, nneigh1, ksi1, sinp1, cosp1, t_width, d_width, &
         & cut_distance, order, pd, ang_norm2,distance_scale, angular_scale, alchemy, verbose)

    ! Pre-calculate self-scalar terms 
    self_scalar2 = get_selfscalar(x2, nm2, n2, nneigh2, ksi2, sinp2, cosp2, t_width, d_width, &
         & cut_distance, order, pd, ang_norm2,distance_scale, angular_scale, alchemy, verbose)

    ! Self-scalar derivatives
    allocate(self_scalar1_ef(nm1, 3,2, maxval(n1)))
    do a = 1, nm1
        ni = n1(a)
        do xyz = 1, 3
            do pm = 1, 2
                do i = 1, ni

                    self_scalar1_ef(a,xyz,pm,i) = scalar(x1(a,i,:,:), x1(a,i,:,:), nneigh1(a,i), nneigh1(a,i), &
                        & ksi1_ef(a,xyz,pm,i,:), ksi1_ef(a,xyz,pm,i,:), &
                        & sinp1_ef(a,xyz,pm,i,:,:,:), sinp1_ef(a,xyz,pm,i,:,:,:), &
                        & cosp1_ef(a,xyz,pm,i,:,:,:), cosp1_ef(a,xyz,pm,i,:,:,:), &
                        & t_width, d_width, cut_distance, order, &
                        & pd, ang_norm2, distance_scale, angular_scale, alchemy)

                enddo
            enddo
        enddo
    enddo

    ! Self-scalar derivatives
    allocate(self_scalar2_ef(nm2, 3,2, maxval(n2)))
    do a = 1, nm2
        ni = n2(a)
        do xyz = 1, 3
            do pm = 1, 2
                do i = 1, ni

                    self_scalar2_ef(a,xyz,pm,i) = scalar(x2(a,i,:,:), x2(a,i,:,:), nneigh2(a,i), nneigh2(a,i), &
                        & ksi2_ef(a,xyz,pm,i,:), ksi2_ef(a,xyz,pm,i,:), &
                        & sinp2_ef(a,xyz,pm,i,:,:,:), sinp2_ef(a,xyz,pm,i,:,:,:), &
                        & cosp2_ef(a,xyz,pm,i,:,:,:), cosp2_ef(a,xyz,pm,i,:,:,:), &
                        & t_width, d_width, cut_distance, order, &
                        & pd, ang_norm2, distance_scale, angular_scale, alchemy)

                enddo
            enddo
        enddo
    enddo

    t_start = omp_get_wtime()
    if (verbose) write (*,"(A)", advance="no") "KERNEL WITH FIELDS"
    
    !$OMP PARALLEL DO schedule(dynamic) PRIVATE(s12,ni,nj)
    do a = 1, nm1
        ni = n1(a)
        do i = 1, ni
            do b = 1, nm2
                nj = n2(b)
                do j = 1, nj

                    s12 = scalar(x1(a,i,:,:), x2(b,j,:,:), &
                        & nneigh1(a,i), nneigh2(b,j), &
                        & ksi1(a,i,:), ksi2(b,j,:), &
                        & sinp1(a,i,:,:,:), sinp2(b,j,:,:,:), &
                        & cosp1(a,i,:,:,:), cosp2(b,j,:,:,:), &
                        & t_width, d_width, cut_distance, order, &
                        & pd, ang_norm2, distance_scale, angular_scale, alchemy)
                    
                        kernels(:, a, b) = kernels(:, a, b) &
                            & + kernel(self_scalar1(a,i),  self_scalar2(b,j), s12, &
                            & kernel_idx, parameters)

                enddo
            enddo
        enddo
    enddo
    !$OMP END PARALLEL DO

    t_end = omp_get_wtime()
    if (verbose) write (*,"(A,F12.4,A)") "                    Time = ", t_end - t_start, " s"
   
    t_start = omp_get_wtime()
    if (verbose) write (*,"(A)", advance="no") "KERNEL EF DERIVATIVE 1/2"
    
    !$OMP PARALLEL DO schedule(dynamic) PRIVATE(s12,ni,nj,idx_a,idx_b)
    do a = 1, nm1
        ni = n1(a)
        do i = 1, ni

            idx_a = a


            do b = 1, nm2
                nj = n2(b)
                do j = 1, nj
                do xyz = 1, 3
                idx_b = (b - 1) * 3 + xyz + nm2

                do pm = 1, 2

                    s12 = scalar(x1(a,i,:,:), x2(b,j,:,:), &
                        & nneigh1(a,i), nneigh2(b,j), &
                        & ksi1(a,i,:), ksi2_ef(b,xyz,pm,j,:), &
                        & sinp1(a,i,:,:,:), sinp2_ef(b,xyz,pm,j,:,:,:), &
                        & cosp1(a,i,:,:,:), cosp2_ef(b,xyz,pm,j,:,:,:), &
                        & t_width, d_width, cut_distance, order, &
                        & pd, ang_norm2, distance_scale, angular_scale, alchemy)
                    
                    if (pm == 1) then

                        kernels(:, idx_a, idx_b) = kernels(:, idx_a, idx_b) &
                            & + kernel(self_scalar1(a,i),  self_scalar2_ef(b,xyz,pm,j), s12, &
                            & kernel_idx, parameters) / (2 * df)

                    else
                        
                        kernels(:, idx_a, idx_b) = kernels(:, idx_a, idx_b) &
                            & - kernel(self_scalar1(a,i),  self_scalar2_ef(b,xyz,pm,j), s12, &
                            & kernel_idx, parameters) / (2 * df)

                    endif

                enddo
                enddo
                enddo
            enddo
        enddo
    enddo
    !$OMP END PARALLEL DO
    
    t_end = omp_get_wtime()
    if (verbose) write (*,"(A,F12.4,A)") "                Time = ", t_end - t_start, " s"

    t_start = omp_get_wtime()
    if (verbose) write (*,"(A)", advance="no") "KERNEL EF DERIVATIVE 2/2"
    !$OMP PARALLEL DO schedule(dynamic) PRIVATE(s12,ni,nj,idx_a,idx_b)

    do a = 1, nm2
        ni = n2(a)
        do i = 1, ni

            idx_a = a


            do b = 1, nm1
                nj = n1(b)
                do j = 1, nj
                do xyz = 1, 3
                idx_b = (b - 1) * 3 + xyz + nm1

                do pm = 1, 2

                    s12 = scalar(x2(a,i,:,:), x1(b,j,:,:), &
                        & nneigh2(a,i), nneigh1(b,j), &
                        & ksi2(a,i,:), ksi1_ef(b,xyz,pm,j,:), &
                        & sinp2(a,i,:,:,:), sinp1_ef(b,xyz,pm,j,:,:,:), &
                        & cosp2(a,i,:,:,:), cosp1_ef(b,xyz,pm,j,:,:,:), &
                        & t_width, d_width, cut_distance, order, &
                        & pd, ang_norm2, distance_scale, angular_scale, alchemy)
                    
                    if (pm == 1) then

                        ! kernels(:, idx_a, idx_b) = kernels(:, idx_a, idx_b) &
                        kernels(:, idx_b, idx_a) = kernels(:, idx_b, idx_a) &
                            & + kernel(self_scalar2(a,i),  self_scalar1_ef(b,xyz,pm,j), s12, &
                            & kernel_idx, parameters) / (2 * df)

                    else
                        
                        ! kernels(:, idx_a, idx_b) = kernels(:, idx_a, idx_b) &
                        kernels(:, idx_b, idx_a) = kernels(:, idx_b, idx_a) &
                            & - kernel(self_scalar2(a,i),  self_scalar1_ef(b,xyz,pm,j), s12, &
                            & kernel_idx, parameters) / (2 * df)

                    endif

                enddo
                enddo
                enddo
            enddo
        enddo
    enddo
    !$OMP END PARALLEL DO
    
    t_end = omp_get_wtime()
    if (verbose) write (*,"(A,F12.4,A)") "                Time = ", t_end - t_start, " s"

    t_start = omp_get_wtime()
    if (verbose) write (*,"(A)", advance="no") "KERNEL EF HESSIAN   "
    
    ! should be zero?

    !$OMP PARALLEL DO schedule(dynamic) PRIVATE(s12,ni,nj,idx_a,idx_b)
    do a = 1, nm1
        
        ni = n1(a)
        do i = 1, ni
        do xyz1 = 1, 3
        idx_a = (a - 1) * 3 + xyz1 + nm1
        do pm1 = 1, 2


            do b = 1, nm2

                nj = n2(b)
                do j = 1, nj
                do xyz2 = 1, 3
                idx_b = (b - 1) * 3 + xyz2 + nm2
                do pm2 = 1, 2

                    s12 = scalar(x1(a,i,:,:), x2(b,j,:,:), &
                        & nneigh1(a,i), nneigh2(b,j), &
                        & ksi1_ef(a,xyz1,pm1,i,:), ksi2_ef(b,xyz2,pm2,j,:), &
                        & sinp1_ef(a,xyz1,pm1,i,:,:,:), sinp2_ef(b,xyz2,pm2,j,:,:,:), &
                        & cosp1_ef(a,xyz1,pm1,i,:,:,:), cosp2_ef(b,xyz2,pm2,j,:,:,:), &
                        & t_width, d_width, cut_distance, order, &
                        & pd, ang_norm2, distance_scale, angular_scale, alchemy)
                if (pm1 == pm2) then

                        kernels(:, idx_a, idx_b) = kernels(:, idx_a, idx_b) &
                & + kernel(self_scalar1_ef(a,xyz1,pm1,i),  self_scalar2_ef(b,xyz2,pm2,j), s12, &
                            & kernel_idx, parameters) / (4 * df**2)
                else 
                        kernels(:, idx_a, idx_b) = kernels(:, idx_a, idx_b) &
                & - kernel(self_scalar1_ef(a,xyz1,pm1,i),  self_scalar2_ef(b,xyz2,pm2,j), s12, &
                            & kernel_idx, parameters) / (4 * df**2)

                endif

                enddo
                enddo
                enddo
            enddo
        enddo
        enddo
        enddo
    enddo
    !$OMP END PARALLEL DO

    t_end = omp_get_wtime()
    if (verbose) write (*,"(A,F12.4,A)") "                    Time = ", t_end - t_start, " s"

    deallocate(self_scalar1)
    deallocate(self_scalar1_ef)
    deallocate(ksi1)
    deallocate(ksi1_ef)
    deallocate(cosp1)
    deallocate(cosp1_ef)
    deallocate(sinp1)
    deallocate(sinp1_ef)

end subroutine fget_ef_gaussian_process_kernels_fchl


subroutine fget_ef_atomic_local_kernels_fchl(x1, x2, verbose, f2, n1, n2, nneigh1, nneigh2, nm1, nm2, nsigmas, na1, &
       & t_width, d_width, cut_start, cut_distance, order, pd, &
       & distance_scale, angular_scale, alchemy, two_body_power, three_body_power, ef_scale,&
       & kernel_idx, parameters, kernels)

    use ffchl_module, only: scalar, get_angular_norm2, get_pmax, get_ksi_ef_field, init_cosp_sinp_ef_field, &
        & get_selfscalar, get_ksi, init_cosp_sinp

    use ffchl_kernels, only: kernel

    use omp_lib, only: omp_get_wtime

    implicit none

    ! fchl descriptors for the training set, format (i,maxatoms,5,maxneighbors)
    double precision, dimension(:,:,:,:), intent(in) :: x1
    double precision, dimension(:,:,:,:), intent(in) :: x2

    ! List of numbers of atoms in each molecule
    integer, dimension(:), intent(in) :: n1
    integer, dimension(:), intent(in) :: n2

    ! Display output
    logical, intent(in) :: verbose

    ! Electric field perturbations for each molecule
    double precision, dimension(:,:), intent(in) :: f2

    ! Number of molecules
    integer, intent(in) :: nm1
    integer, intent(in) :: nm2

    ! Number of sigmas
    integer, intent(in) :: nsigmas

    integer, intent(in) :: na1

    ! Number of neighbors for each atom in each compound
    integer, dimension(:,:), intent(in) :: nneigh1
    integer, dimension(:,:), intent(in) :: nneigh2

    ! Angular Gaussian width
    double precision, intent(in) :: t_width

    ! Distance Gaussian width
    double precision, intent(in) :: d_width

    ! Fraction of cut_distance at which cut-off starts
    double precision, intent(in) :: cut_start
    double precision, intent(in) :: cut_distance

    ! Truncation order for Fourier terms
    integer, intent(in) :: order

    ! Periodic table distance matrix
    double precision, dimension(:,:), intent(in) :: pd

    ! Scaling for angular and distance terms
    double precision, intent(in) :: distance_scale
    double precision, intent(in) :: angular_scale

    ! Switch alchemy on or off
    logical, intent(in) :: alchemy

    ! Decaying power laws for two- and three-body terms
    double precision, intent(in) :: two_body_power
    double precision, intent(in) :: three_body_power


    double precision, intent(in) :: ef_scale

    ! Kernel ID and corresponding parameters
    integer, intent(in) :: kernel_idx
    double precision, dimension(:,:), intent(in) :: parameters

    ! Resulting alpha vector
    double precision, dimension(nsigmas,na1,nm2), intent(out) :: kernels

    ! Internal counters
    integer :: i, j
    integer :: ni, nj
    integer :: a, b
    integer :: idx1

    ! Temporary variables necessary for parallelization
    double precision :: s12

    ! Pre-computed terms in the full distance matrix
    double precision, allocatable, dimension(:,:) :: self_scalar1
    double precision, allocatable, dimension(:,:) :: self_scalar2

    ! Pre-computed two-body weights for nummerical differentation of electric field
    double precision, allocatable, dimension(:,:,:) :: ksi1
    double precision, allocatable, dimension(:,:,:) :: ksi2

    ! Pre-computed terms for the Fourier expansion of the three-body term
    double precision, allocatable, dimension(:,:,:,:,:) :: sinp1
    double precision, allocatable, dimension(:,:,:,:,:) :: cosp1
    double precision, allocatable, dimension(:,:,:,:,:) :: sinp2
    double precision, allocatable, dimension(:,:,:,:,:) :: cosp2

    ! Max index in the periodic table
    integer :: pmax1
    integer :: pmax2

    ! Angular normalization constant
    double precision :: ang_norm2

    ! Max number of neighbors
    integer :: maxneigh1
    integer :: maxneigh2

    ! Variables to calculate time
    double precision :: t_start, t_end

    if (verbose) write (*,*) "CLEARING KERNEL MEM"
    kernels(:,:,:) = 0.0d0

    ! Get max number of neighbors
    maxneigh1 = maxval(nneigh1)
    maxneigh2 = maxval(nneigh2)

    ! Calculate angular normalization constant
    ang_norm2 = get_angular_norm2(t_width)

    ! pmax = max nuclear charge
    pmax1 = get_pmax(x1, n1)
    pmax2 = get_pmax(x2, n2)

    ! Get two-body weight function
    ksi1 = get_ksi(x1, n1, nneigh1, two_body_power, cut_start, cut_distance, verbose)

    ! Get two-body weight function
    ksi2 = get_ksi_ef_field(x2, n2, nneigh2, two_body_power, cut_start, cut_distance, &
                & f2, ef_scale, verbose)

    ! Allocate three-body Fourier terms
    allocate(cosp1(nm1, maxval(n1), pmax1, order, maxneigh1))
    allocate(sinp1(nm1, maxval(n1), pmax1, order, maxneigh1))

    ! Initialize and pre-calculate three-body Fourier terms
    call init_cosp_sinp(x1, n1, nneigh1, three_body_power, order, cut_start,  cut_distance, &
        & cosp1, sinp1, verbose)

    ! Allocate three-body Fourier terms
    allocate(cosp2(nm2, maxval(n2), pmax2, order, maxneigh2))
    allocate(sinp2(nm2, maxval(n2), pmax2, order, maxneigh2))

    ! Initialize and pre-calculate three-body Fourier terms
    call init_cosp_sinp_ef_field(x2, n2, nneigh2, three_body_power, order, cut_start,  cut_distance, &
        & cosp2, sinp2, f2, ef_scale, verbose)

    ! Pre-calculate self-scalar terms
    self_scalar1 = get_selfscalar(x1, nm1, n1, nneigh1, ksi1, sinp1, cosp1, t_width, d_width, &
         & cut_distance, order, pd, ang_norm2,distance_scale, angular_scale, alchemy, verbose)

    ! Pre-calculate self-scalar terms
    self_scalar2 = get_selfscalar(x2, nm2, n2, nneigh2, ksi2, sinp2, cosp2, t_width, d_width, &
         & cut_distance, order, pd, ang_norm2,distance_scale, angular_scale, alchemy, verbose)


    kernels(:,:,:) = 0.0d0


    ! write (*,*) nm1, nm2, na1
    ! write (*,*) size(kernels,dim=1), size(kernels,dim=2), size(kernels,dim=3)


    t_start = omp_get_wtime()
    if (verbose)  write (*,"(A)", advance="no") "KERNEL"

    !$OMP PARALLEL DO schedule(dynamic) PRIVATE(ni,nj,idx1,s12)
    do a = 1, nm1
        ni = n1(a)
        do i = 1, ni

            idx1 = sum(n1(:a)) - ni + i

            do b = 1, nm2
                nj = n2(b)
                do j = 1, nj

                    s12 = scalar(x1(a,i,:,:), x2(b,j,:,:), &
                        & nneigh1(a,i), nneigh2(b,j), ksi1(a,i,:), ksi2(b,j,:), &
                        & sinp1(a,i,:,:,:), sinp2(b,j,:,:,:), &
                        & cosp1(a,i,:,:,:), cosp2(b,j,:,:,:), &
                        & t_width, d_width, cut_distance, order, &
                         & pd, ang_norm2, distance_scale, angular_scale, alchemy)

                    kernels(:, idx1, b) = kernels(:, idx1, b) &
                        & + kernel(self_scalar1(a,i), self_scalar2(b,j), s12, &
                         & kernel_idx, parameters)

                 enddo
             enddo

         enddo
    enddo
    !$OMP END PARALLEL DO

    t_end = omp_get_wtime()
    if (verbose)  write (*,"(A,F12.4,A)") "                                  Time = ", t_end - t_start, " s"

    deallocate(self_scalar1)
    deallocate(self_scalar2)
    deallocate(ksi1)
    deallocate(ksi2)
    deallocate(cosp1)
    deallocate(cosp2)
    deallocate(sinp1)
    deallocate(sinp2)
end subroutine fget_ef_atomic_local_kernels_fchl


subroutine fget_ef_atomic_local_gradient_kernels_fchl(x1, x2, verbose, n1, n2, nneigh1, nneigh2, nm1, nm2, na1, nsigmas, &
       & t_width, d_width, cut_start, cut_distance, order, pd, &
       & distance_scale, angular_scale, alchemy, two_body_power, three_body_power, ef_scale,&
       & df, kernel_idx, parameters, kernels)

    use ffchl_module, only: scalar, get_angular_norm2, get_pmax, get_ksi, init_cosp_sinp, &
        & get_selfscalar, get_ksi_ef, init_cosp_sinp_ef

    use ffchl_kernels, only: kernel

    use omp_lib, only: omp_get_wtime

    implicit none

    ! fchl descriptors for the training set, format (i,maxatoms,5,maxneighbors)
    double precision, dimension(:,:,:,:), intent(in) :: x1
    double precision, dimension(:,:,:,:), intent(in) :: x2

    ! Display output
    logical, intent(in) :: verbose

    ! List of numbers of atoms in each molecule
    integer, dimension(:), intent(in) :: n1
    integer, dimension(:), intent(in) :: n2

    ! Number of molecules
    integer, intent(in) :: nm1
    integer, intent(in) :: nm2

    ! Number of atoms in set 1
    integer, intent(in) :: na1

    ! Number of sigmas
    integer, intent(in) :: nsigmas

    ! Number of neighbors for each atom in each compound
    integer, dimension(:,:), intent(in) :: nneigh1
    integer, dimension(:,:), intent(in) :: nneigh2

    ! Angular Gaussian width
    double precision, intent(in) :: t_width

    ! Distance Gaussian width
    double precision, intent(in) :: d_width

    ! Fraction of cut_distance at which cut-off starts
    double precision, intent(in) :: cut_start
    double precision, intent(in) :: cut_distance

    ! Truncation order for Fourier terms
    integer, intent(in) :: order

    ! Periodic table distance matrix
    double precision, dimension(:,:), intent(in) :: pd

    ! Scaling for angular and distance terms
    double precision, intent(in) :: distance_scale
    double precision, intent(in) :: angular_scale

    ! Switch alchemy on or off
    logical, intent(in) :: alchemy

    ! Decaying power laws for two- and three-body terms
    double precision, intent(in) :: two_body_power
    double precision, intent(in) :: three_body_power


    double precision, intent(in) :: ef_scale
    double precision, intent(in) :: df

    ! Kernel ID and corresponding parameters
    integer, intent(in) :: kernel_idx
    double precision, dimension(:,:), intent(in) :: parameters

    ! Resulting alpha vector
    double precision, dimension(nsigmas,na1,nm2*3), intent(out) :: kernels

    ! Internal counters
    integer :: i, j
    integer :: ni, nj
    integer :: a, b
    integer :: xyz, pm
    integer :: idx_a
    integer :: idx_b

    ! Temporary variables necessary for parallelization
    double precision :: s12

    ! Pre-computed terms in the full distance matrix
    double precision, allocatable, dimension(:,:) :: self_scalar1
    double precision, allocatable, dimension(:,:,:,:) :: self_scalar2_ef

    ! Pre-computed two-body weights for nummerical differentation of electric field
    double precision, allocatable, dimension(:,:,:) :: ksi1
    double precision, allocatable, dimension(:,:,:,:,:) :: ksi2_ef

    ! Pre-computed terms for the Fourier expansion of the three-body term
    double precision, allocatable, dimension(:,:,:,:,:) :: sinp1
    double precision, allocatable, dimension(:,:,:,:,:) :: cosp1
    double precision, allocatable, dimension(:,:,:,:,:,:,:) :: sinp2_ef
    double precision, allocatable, dimension(:,:,:,:,:,:,:) :: cosp2_ef

    ! Max index in the periodic table
    integer :: pmax1
    integer :: pmax2

    ! Angular normalization constant
    double precision :: ang_norm2

    ! Max number of neighbors
    integer :: maxneigh1
    integer :: maxneigh2

    ! Variables to calculate time
    double precision :: t_start, t_end

    if (verbose)  write (*,*) "CLEARING KERNEL MEM"
    kernels(:,:,:) = 0.0d0

    ! Get max number of neighbors
    maxneigh1 = maxval(nneigh1)
    maxneigh2 = maxval(nneigh2)

    ! Calculate angular normalization constant
    ang_norm2 = get_angular_norm2(t_width)

    ! pmax = max nuclear charge
    pmax1 = get_pmax(x1, n1)
    pmax2 = get_pmax(x2, n2)

    ! Get two-body weight function
    ksi1 = get_ksi(x1, n1, nneigh1, two_body_power, cut_start, cut_distance, verbose)
    ksi2_ef = get_ksi_ef(x2, n2, nneigh2, two_body_power, cut_start, cut_distance, ef_scale, df, verbose)

    ! Allocate three-body Fourier terms
    allocate(cosp1(nm1, maxval(n1), pmax1, order, maxneigh1))
    allocate(sinp1(nm1, maxval(n1), pmax1, order, maxneigh1))

    ! Initialize and pre-calculate three-body Fourier terms
    call init_cosp_sinp(x1, n1, nneigh1, three_body_power, order, cut_start,  cut_distance, &
        & cosp1, sinp1, verbose)

    ! Allocate three-body Fourier terms
    allocate(cosp2_ef(nm2, 3, 2, maxval(n2), pmax2, order, maxneigh2))
    allocate(sinp2_ef(nm2, 3, 2, maxval(n2), pmax2, order, maxneigh2))

    ! Initialize and pre-calculate three-body Fourier terms
    call init_cosp_sinp_ef(x2, n2, nneigh2, three_body_power, order, cut_start, cut_distance, &
        & cosp2_ef, sinp2_ef, ef_scale, df, verbose)

    ! Pre-calculate self-scalar terms
    self_scalar1 = get_selfscalar(x1, nm1, n1, nneigh1, ksi1, sinp1, cosp1, t_width, d_width, &
         & cut_distance, order, pd, ang_norm2,distance_scale, angular_scale, alchemy, verbose)

    allocate(self_scalar2_ef(nm2, 3,2, maxval(n2)))
    do a = 1, nm2
        ni = n2(a)
        do xyz = 1, 3
            do pm = 1, 2
                do i = 1, ni

                    self_scalar2_ef(a,xyz,pm,i) = scalar(x2(a,i,:,:), x2(a,i,:,:), nneigh2(a,i), nneigh2(a,i), &
                        & ksi2_ef(a,xyz,pm,i,:), ksi2_ef(a,xyz,pm,i,:), &
                        & sinp2_ef(a,xyz,pm,i,:,:,:), sinp2_ef(a,xyz,pm,i,:,:,:), &
                        & cosp2_ef(a,xyz,pm,i,:,:,:), cosp2_ef(a,xyz,pm,i,:,:,:), &
                        & t_width, d_width, cut_distance, order, &
                        & pd, ang_norm2, distance_scale, angular_scale, alchemy)

                enddo
            enddo
        enddo
    enddo

    t_start = omp_get_wtime()
    if (verbose)  write (*,"(A)", advance="no") "KERNEL EF DERIVATIVE"

    !$OMP PARALLEL DO schedule(dynamic) PRIVATE(s12,ni,nj,idx_a,idx_b)

    do a = 1, nm1
        ni = n1(a)
        do i = 1, ni

            idx_a = sum(n1(:a)) - ni + i


            do b = 1, nm2
                nj = n2(b)
                do j = 1, nj
                do xyz = 1, 3
                idx_b = (b - 1) * 3 + xyz

                do pm = 1, 2

                    s12 = scalar(x1(a,i,:,:), x2(b,j,:,:), &
                        & nneigh1(a,i), nneigh2(b,j), &
                        & ksi1(a,i,:), ksi2_ef(b,xyz,pm,j,:), &
                        & sinp1(a,i,:,:,:), sinp2_ef(b,xyz,pm,j,:,:,:), &
                        & cosp1(a,i,:,:,:), cosp2_ef(b,xyz,pm,j,:,:,:), &
                        & t_width, d_width, cut_distance, order, &
                        & pd, ang_norm2, distance_scale, angular_scale, alchemy)

                    if (pm == 1) then

                        kernels(:, idx_a, idx_b) = kernels(:, idx_a, idx_b) &
                            & + kernel(self_scalar1(a,i),  self_scalar2_ef(b,xyz,pm,j), s12, &
                            & kernel_idx, parameters)

                    else

                        kernels(:, idx_a, idx_b) = kernels(:, idx_a, idx_b) &
                            & - kernel(self_scalar1(a,i),  self_scalar2_ef(b,xyz,pm,j), s12, &
                            & kernel_idx, parameters)

                    endif

                enddo
                enddo
                enddo
            enddo
        enddo
    enddo
    !$OMP END PARALLEL DO

    kernels = kernels / (2 * df)

    t_end = omp_get_wtime()
    if (verbose)  write (*,"(A,F12.4,A)") "                    Time = ", t_end - t_start, " s"

    deallocate(self_scalar1)
    deallocate(self_scalar2_ef)
    deallocate(ksi1)
    deallocate(ksi2_ef)
    deallocate(cosp1)
    deallocate(cosp2_ef)
    deallocate(sinp1)
    deallocate(sinp2_ef)

end subroutine fget_ef_atomic_local_gradient_kernels_fchl



! subroutine fget_ef_local_hessian_kernels_fchl(x1, x2, verbose, n1, n2, nneigh1, nneigh2, nm1, nm2, nsigmas, &
!        & t_width, d_width, cut_start, cut_distance, order, pd, &
!        & distance_scale, angular_scale, alchemy, two_body_power, three_body_power, &
!        & ef_scale, df, kernel_idx, parameters, kernels)
! 
!     use ffchl_module, only: scalar, get_angular_norm2, get_pmax, get_ksi, init_cosp_sinp, &
!         & get_selfscalar, get_ksi_ef, init_cosp_sinp_ef
! 
!     use ffchl_kernels, only: kernel
! 
!     use omp_lib, only: omp_get_wtime
! 
!     implicit none
! 
!     ! fchl descriptors for the training set, format (i,maxatoms,5,maxneighbors)
!     double precision, dimension(:,:,:,:), intent(in) :: x1
!     double precision, dimension(:,:,:,:), intent(in) :: x2
! 
!     ! Display output
!     logical, intent(in) :: verbose
! 
!     ! List of numbers of atoms in each molecule
!     integer, dimension(:), intent(in) :: n1
!     integer, dimension(:), intent(in) :: n2
! 
!     ! Number of molecules
!     integer, intent(in) :: nm1
!     integer, intent(in) :: nm2
! 
!     ! Number of sigmas
!     integer, intent(in) :: nsigmas
! 
!     ! Number of neighbors for each atom in each compound
!     integer, dimension(:,:), intent(in) :: nneigh1
!     integer, dimension(:,:), intent(in) :: nneigh2
! 
!     ! Angular Gaussian width
!     double precision, intent(in) :: t_width
! 
!     ! Distance Gaussian width
!     double precision, intent(in) :: d_width
! 
!     ! Fraction of cut_distance at which cut-off starts
!     double precision, intent(in) :: cut_start
!     double precision, intent(in) :: cut_distance
! 
!     ! Truncation order for Fourier terms
!     integer, intent(in) :: order
! 
!     ! Periodic table distance matrix
!     double precision, dimension(:,:), intent(in) :: pd
! 
!     ! Scaling for angular and distance terms
!     double precision, intent(in) :: distance_scale
!     double precision, intent(in) :: angular_scale
! 
!     ! Switch alchemy on or off
!     logical, intent(in) :: alchemy
! 
!     ! Decaying power laws for two- and three-body terms
!     double precision, intent(in) :: two_body_power
!     double precision, intent(in) :: three_body_power
! 
!     double precision, intent(in) :: ef_scale
!     double precision, intent(in) :: df
! 
!     ! Kernel ID and corresponding parameters
!     integer, intent(in) :: kernel_idx
!     double precision, dimension(:,:), intent(in) :: parameters
! 
!     ! Resulting alpha vector
!     ! double precision, dimension(nsigmas,nm1,nm2), intent(out) :: kernels
!     double precision, dimension(nsigmas,nm1*3,nm2*3), intent(out) :: kernels
! 
!     double precision, dimension(nsigmas):: kernel_sum1
!     double precision, dimension(nsigmas):: kernel_sum2
! 
!     ! Internal counters
!     integer :: i, j
!     integer :: ni, nj
!     integer :: a, b
!     integer :: xyz, pm
!     integer :: xyz1, pm1
!     integer :: xyz2, pm2
! 
!     integer :: idx_a, idx_b
! 
!     ! Temporary variables necessary for parallelization
!     double precision :: s12
! 
!     ! Pre-computed terms in the full distance matrix
!     double precision, allocatable, dimension(:,:,:,:) :: self_scalar1_ef
!     double precision, allocatable, dimension(:,:,:,:) :: self_scalar2_ef
! 
!     ! Pre-computed two-body weights for nummerical differentation of electric field
!     double precision, allocatable, dimension(:,:,:,:,:) :: ksi1_ef
!     double precision, allocatable, dimension(:,:,:,:,:) :: ksi2_ef
! 
!     ! ! Pre-computed terms for the Fourier expansion of the three-body term
!     double precision, allocatable, dimension(:,:,:,:,:,:,:) :: sinp1_ef
!     double precision, allocatable, dimension(:,:,:,:,:,:,:) :: sinp2_ef
!     double precision, allocatable, dimension(:,:,:,:,:,:,:) :: cosp1_ef
!     double precision, allocatable, dimension(:,:,:,:,:,:,:) :: cosp2_ef
! 
!     ! Max index in the periodic table
!     integer :: pmax1
!     integer :: pmax2
! 
!     ! Angular normalization constant
!     double precision :: ang_norm2
! 
!     ! Max number of neighbors
!     integer :: maxneigh1
!     integer :: maxneigh2
! 
!     ! Variables to calculate time
!     double precision :: t_start, t_end
! 
!     write (*,*) "CLEARING KERNEL MEM"
!     kernels(:,:,:) = 0.0d0
! 
!     ! Get max number of neighbors
!     maxneigh1 = maxval(nneigh1)
!     maxneigh2 = maxval(nneigh2)
! 
!     ! Calculate angular normalization constant
!     ang_norm2 = get_angular_norm2(t_width)
! 
!     ! pmax = max nuclear charge
!     pmax1 = get_pmax(x1, n1)
!     pmax2 = get_pmax(x2, n2)
! 
! 
!     ! Get two-body weight function
!     ksi1_ef = get_ksi_ef(x1, n1, nneigh1, two_body_power, cut_start, cut_distance, ef_scale, df)
!     ksi2_ef = get_ksi_ef(x2, n2, nneigh2, two_body_power, cut_start, cut_distance, ef_scale, df)
! 
! 
!     ! Allocate three-body Fourier terms
!     allocate(cosp1_ef(nm1, 3, 2, maxval(n1), pmax1, order, maxneigh1))
!     allocate(sinp1_ef(nm1, 3, 2, maxval(n1), pmax1, order, maxneigh1))
! 
!     ! Initialize and pre-calculate three-body Fourier terms
!     call init_cosp_sinp_ef(x1, n1, nneigh1, three_body_power, order, cut_start,  cut_distance, &
!         & cosp1_ef, sinp1_ef, ef_scale, df)
! 
! 
!     ! Allocate three-body Fourier terms
!     allocate(cosp2_ef(nm2, 3, 2, maxval(n2), pmax2, order, maxneigh2))
!     allocate(sinp2_ef(nm2, 3, 2, maxval(n2), pmax2, order, maxneigh2))
! 
!     ! Initialize and pre-calculate three-body Fourier terms
!     call init_cosp_sinp_ef(x2, n2, nneigh2, three_body_power, order, cut_start, cut_distance, &
!         & cosp2_ef, sinp2_ef, ef_scale, df)
! 
! 
!     allocate(self_scalar1_ef(nm1, 3,2, maxval(n1)))
!     do a = 1, nm1
!         ni = n1(a)
!         do xyz = 1, 3
!             do pm = 1, 2
!                 do i = 1, ni
! 
!                     self_scalar1_ef(a,xyz,pm,i) = scalar(x1(a,i,:,:), x1(a,i,:,:), nneigh1(a,i), nneigh1(a,i), &
!                         & ksi1_ef(a,xyz,pm,i,:), ksi1_ef(a,xyz,pm,i,:), &
!                         & sinp1_ef(a,xyz,pm,i,:,:,:), sinp1_ef(a,xyz,pm,i,:,:,:), &
!                         & cosp1_ef(a,xyz,pm,i,:,:,:), cosp1_ef(a,xyz,pm,i,:,:,:), &
!                         & t_width, d_width, cut_distance, order, &
!                         & pd, ang_norm2, distance_scale, angular_scale, alchemy)
! 
!                 enddo
!             enddo
!         enddo
!     enddo
! 
! 
!     allocate(self_scalar2_ef(nm2, 3,2, maxval(n2)))
!     do a = 1, nm2
!         ni = n2(a)
!         do xyz = 1, 3
!             do pm = 1, 2
!                 do i = 1, ni
! 
!                     self_scalar2_ef(a,xyz,pm,i) = scalar(x2(a,i,:,:), x2(a,i,:,:), nneigh2(a,i), nneigh2(a,i), &
!                         & ksi2_ef(a,xyz,pm,i,:), ksi2_ef(a,xyz,pm,i,:), &
!                         & sinp2_ef(a,xyz,pm,i,:,:,:), sinp2_ef(a,xyz,pm,i,:,:,:), &
!                         & cosp2_ef(a,xyz,pm,i,:,:,:), cosp2_ef(a,xyz,pm,i,:,:,:), &
!                         & t_width, d_width, cut_distance, order, &
!                         & pd, ang_norm2, distance_scale, angular_scale, alchemy)
! 
!                 enddo
!             enddo
!         enddo
!     enddo
! 
!     t_start = omp_get_wtime()
!     write (*,"(A)", advance="no") "KERNEL"
! 
!     !$OMP PARALLEL DO schedule(dynamic) PRIVATE(s12,ni,nj,idx_a,idx_b,kernel_sum1,kernel_sum2)
!     do b = 1, nm2
!     nj = n2(b)
!     do j = 1, nj
! 
!         do a = 1, nm1
!         ni = n1(a)
!         do i = 1, ni
! 
!             do xyz1 = 1, 3
!             do xyz2 = 1, 3
! 
!                 idx_a = (a - 1) * 3 + xyz1
!                 idx_b = (b - 1) * 3 + xyz2
! 
!                 do pm1 = 1, 2
!                 do pm2 = 1, 2
! 
!                     s12 = scalar(x1(a,i,:,:), x2(b,j,:,:), &
!                         & nneigh1(a,i), nneigh2(b,j), &
!                         & ksi1_ef(a,xyz1,pm1,i,:), ksi2_ef(b,xyz2,pm2,j,:), &
!                         & sinp1_ef(a,xyz1,pm1,i,:,:,:), sinp2_ef(b,xyz2,pm2,j,:,:,:), &
!                         & cosp1_ef(a,xyz1,pm1,i,:,:,:), cosp2_ef(b,xyz2,pm2,j,:,:,:), &
!                         & t_width, d_width, cut_distance, order, &
!                         & pd, ang_norm2, distance_scale, angular_scale, alchemy)
! 
! 
!                     if (pm1 == pm2) then
! 
!                         kernels(:, idx_a, idx_b) = kernels(:, idx_a, idx_b) &
!                             & + kernel(self_scalar1_ef(a,xyz1,pm1,i),  self_scalar2_ef(b,xyz2,pm2,j), s12, &
!                             & kernel_idx, parameters)
!                     else
! 
!                         kernels(:, idx_a, idx_b) = kernels(:, idx_a, idx_b) &
!                             & - kernel(self_scalar1_ef(a,xyz1,pm1,i),  self_scalar2_ef(b,xyz2,pm2,j), s12, &
!                             & kernel_idx, parameters)
! 
!                     endif
!             enddo
!             enddo
!         enddo
!         enddo
!         enddo
!     enddo
!     enddo
!     enddo
!     !$OMP END PARALLEL DO
! 
!     kernels = kernels / (4 * df**2)
! 
!     t_end = omp_get_wtime()
!     write (*,"(A,F12.4,A)") "                                  Time = ", t_end - t_start, " s"
! 
! end subroutine fget_ef_local_hessian_kernels_fchl



!
! TODO: Fix, experimental code for polarizability
!
! subroutine fget_kernels_fchl_ef_2ndderiv(x1, x2, n1, n2, nneigh1, nneigh2, nm1, nm2, na1, nsigmas, &
!        & t_width, d_width, cut_start, cut_distance, order, pd, &
!        & distance_scale, angular_scale, alchemy, two_body_power, three_body_power, ef_scale,&
!        & df, kernel_idx, parameters, kernels)
!
!     use ffchl_module, only: scalar, get_angular_norm2, get_pmax, get_ksi, init_cosp_sinp, &
!         & get_selfscalar, get_ksi_pol, init_cosp_sinp_pol
!
!     use ffchl_kernels, only: kernel
!
!     use omp_lib, only: omp_get_wtime
!
!     implicit none
!
!     ! fchl descriptors for the training set, format (i,maxatoms,5,maxneighbors)
!     double precision, dimension(:,:,:,:), intent(in) :: x1
!     double precision, dimension(:,:,:,:), intent(in) :: x2
!
!     ! List of numbers of atoms in each molecule
!     integer, dimension(:), intent(in) :: n1
!     integer, dimension(:), intent(in) :: n2
!
!     ! Number of molecules
!     integer, intent(in) :: nm1
!     integer, intent(in) :: nm2
!
!     ! Number of atoms in set 1
!     integer, intent(in) :: na1
!
!     ! Number of sigmas
!     integer, intent(in) :: nsigmas
!
!     ! Number of neighbors for each atom in each compound
!     integer, dimension(:,:), intent(in) :: nneigh1
!     integer, dimension(:,:), intent(in) :: nneigh2
!
!     ! Angular Gaussian width
!     double precision, intent(in) :: t_width
!
!     ! Distance Gaussian width
!     double precision, intent(in) :: d_width
!
!     ! Fraction of cut_distance at which cut-off starts
!     double precision, intent(in) :: cut_start
!     double precision, intent(in) :: cut_distance
!
!     ! Truncation order for Fourier terms
!     integer, intent(in) :: order
!
!     ! Periodic table distance matrix
!     double precision, dimension(:,:), intent(in) :: pd
!
!     ! Scaling for angular and distance terms
!     double precision, intent(in) :: distance_scale
!     double precision, intent(in) :: angular_scale
!
!     ! Switch alchemy on or off
!     logical, intent(in) :: alchemy
!
!     ! Decaying power laws for two- and three-body terms
!     double precision, intent(in) :: two_body_power
!     double precision, intent(in) :: three_body_power
!
!
!     double precision, intent(in) :: ef_scale
!     double precision, intent(in) :: df
!
!     ! Kernel ID and corresponding parameters
!     integer, intent(in) :: kernel_idx
!     double precision, dimension(:,:), intent(in) :: parameters
!
!     ! Resulting alpha vector
!     ! double precision, dimension(nsigmas,nm1,nm2), intent(out) :: kernels
!     ! double precision, dimension(nsigmas,na1,nm2*3), intent(out) :: kernels
!     double precision, dimension(nsigmas,na1,na1*3), intent(out) :: kernels
!
!     ! Internal counters
!     integer :: i, j
!     integer :: ni, nj
!     integer :: a, b
!     integer :: xyz, pm
!     integer :: idx_a
!     integer :: idx_b
!     integer :: qidx
!
!     ! Temporary variables necessary for parallelization
!     double precision :: s12
!
!     ! Pre-computed terms in the full distance matrix
!     double precision, allocatable, dimension(:,:) :: self_scalar1
!     double precision, allocatable, dimension(:,:,:) :: self_scalar2_pol
!
!     ! Pre-computed two-body weights for nummerical differentation of electric field
!     double precision, allocatable, dimension(:,:,:) :: ksi1
!     double precision, allocatable, dimension(:,:,:,:) :: ksi2_pol
!
!     ! Pre-computed terms for the Fourier expansion of the three-body term
!     double precision, allocatable, dimension(:,:,:,:,:) :: sinp1
!     double precision, allocatable, dimension(:,:,:,:,:) :: cosp1
!     double precision, allocatable, dimension(:,:,:,:,:,:) :: sinp2_pol
!     double precision, allocatable, dimension(:,:,:,:,:,:) :: cosp2_pol
!
!     double precision, dimension(nsigmas) :: unperturbed
!     double precision, dimension(nsigmas) :: test_plus, test_minus
!
!     integer, dimension(3,2) :: idx
!
!     ! Max index in the periodic table
!     integer :: pmax1
!     integer :: pmax2
!
!     ! Angular normalization constant
!     double precision :: ang_norm2
!
!     ! Max number of neighbors
!     integer :: maxneigh1
!     integer :: maxneigh2
!
!     ! Variables to calculate time
!     double precision :: t_start, t_end
!
!     write (*,*) "CLEARING KERNEL MEM"
!     kernels(:,:,:) = 0.0d0
!
!     ! Get max number of neighbors
!     maxneigh1 = maxval(nneigh1)
!     maxneigh2 = maxval(nneigh2)
!
!     ! Calculate angular normalization constant
!     ang_norm2 = get_angular_norm2(t_width)
!
!     ! pmax = max nuclear charge
!     pmax1 = get_pmax(x1, n1)
!     pmax2 = get_pmax(x2, n2)
!
!     ! Get two-body weight function
!     ksi1 = get_ksi(x1, n1, nneigh1, two_body_power, cut_start, cut_distance)
!     ksi2_pol = get_ksi_pol(x2, n2, nneigh2, two_body_power, cut_start, cut_distance, ef_scale, df)
!
!     ! Allocate three-body Fourier terms
!     allocate(cosp1(nm1, maxval(n1), pmax1, order, maxneigh1))
!     allocate(sinp1(nm1, maxval(n1), pmax1, order, maxneigh1))
!
!     ! Initialize and pre-calculate three-body Fourier terms
!     call init_cosp_sinp(x1, n1, nneigh1, three_body_power, order, cut_start,  cut_distance, &
!         & cosp1, sinp1)
!
!     ! Allocate three-body Fourier terms
!     allocate(cosp2_pol(nm2, 19, maxval(n2), pmax2, order, maxneigh2))
!     allocate(sinp2_pol(nm2, 19, maxval(n2), pmax2, order, maxneigh2))
!
!     ! Initialize and pre-calculate three-body Fourier terms
!     call init_cosp_sinp_pol(x2, n2, nneigh2, three_body_power, order, cut_start, cut_distance, &
!         & cosp2_pol, sinp2_pol, ef_scale, df)
!
!     ! Pre-calculate self-scalar terms
!     self_scalar1 = get_selfscalar(x1, nm1, n1, nneigh1, ksi1, sinp1, cosp1, t_width, d_width, &
!          & cut_distance, order, pd, ang_norm2,distance_scale, angular_scale, alchemy)
!
!     allocate(self_scalar2_pol(nm2, 19, maxval(n2)))
!     do a = 1, nm2
!         ni = n2(a)
!         do xyz = 1, 19
!             do i = 1, ni
!
!                 self_scalar2_pol(a,xyz,i) = scalar(x2(a,i,:,:), x2(a,i,:,:), &
!                     & nneigh2(a,i), nneigh2(a,i), &
!                     & ksi2_pol(a,xyz,i,:), ksi2_pol(a,xyz,i,:), &
!                     & sinp2_pol(a,xyz,i,:,:,:), sinp2_pol(a,xyz,i,:,:,:), &
!                     & cosp2_pol(a,xyz,i,:,:,:), cosp2_pol(a,xyz,i,:,:,:), &
!                     & t_width, d_width, cut_distance, order, &
!                     & pd, ang_norm2, distance_scale, angular_scale, alchemy)
!
!             enddo
!         enddo
!     enddo
!
!     t_start = omp_get_wtime()
!     write (*,"(A)", advance="no") "KERNEL EF DERIVATIVE"
!
!     idx(1,:) = (/ 17,  3 /) !xx = (17) + (3) - 2*10
!     idx(2,:) = (/ 11,  9 /) !yy = (11) + (9) - 2*10
!     idx(3,:) = (/ 13,  7 /) !zz = (13) + (7) - 2*10
!
!     ! Loop over atoms/basis functions
!     ! !$OMP PARALLEL DO schedule(dynamic) PRIVATE(s12,ni,nj,idx_a,idx_b)
!     do a = 1, nm1
!     ni = n1(a)
!     do i = 1, ni
!     idx_a = sum(n1(:a)) - ni + i
!
!
!         do b = 1, nm2
!             nj = n2(b)
!
!             do j = 1, nj
!
!                 ! Precalculate L2 for unperturbed kernel (charge index = 10)
!                 qidx = 10
!
!                 ! write(*,*) qidx
!                 ! isotropic (XX, YY and ZZ)
!                 s12 = scalar(x1(a,i,:,:), x2(b,j,:,:), &
!                 & nneigh1(a,i), nneigh2(b,j), &
!                 & ksi1(a,i,:), ksi2_pol(b,qidx,j,:), &
!                     & sinp1(a,i,:,:,:), sinp2_pol(b,qidx,j,:,:,:), &
!                     & cosp1(a,i,:,:,:), cosp2_pol(b,qidx,j,:,:,:), &
!                     & t_width, d_width, cut_distance, order, &
!                     & pd, ang_norm2, distance_scale, angular_scale, alchemy)
!
!                 unperturbed(:) = kernel(self_scalar1(a,i),  self_scalar2_pol(b,qidx,j), s12, &
!                     & kernel_idx, parameters)
!                 ! write(*,*) s12, unperturbed(:)
!
!                 do xyz = 1, 3
!
!                     ! idx_b = (b - 1) * 3 + xyz
!                     idx_b = (sum(n2(:b)) - nj + j) * 3 + xyz
!
!                     qidx = idx(xyz,1)
!                     ! write(*,*) xyz, qidx
!
!                     s12 = scalar(x1(a,i,:,:), x2(b,j,:,:), &
!                         & nneigh1(a,i), nneigh2(b,j), &
!                         & ksi1(a,i,:), ksi2_pol(b,qidx,j,:), &
!                         & sinp1(a,i,:,:,:), sinp2_pol(b,qidx,j,:,:,:), &
!                         & cosp1(a,i,:,:,:), cosp2_pol(b,qidx,j,:,:,:), &
!                         & t_width, d_width, cut_distance, order, &
!                         & pd, ang_norm2, distance_scale, angular_scale, alchemy)
!
!                     kernels(:, idx_a, idx_b) = kernels(:, idx_a, idx_b) &
!                         & + kernel(self_scalar1(a,i),  self_scalar2_pol(b,qidx,j), s12, &
!                         & kernel_idx, parameters)
!
!                     test_plus(:) = kernel(self_scalar1(a,i),  self_scalar2_pol(b,qidx,j), s12, &
!                         & kernel_idx, parameters)
!
!                     qidx = idx(xyz,2)
!                     ! write(*,*) qidx
!                     ! write(*,*) xyz, qidx
!
!                     ! write(*,*) s12, kernel(self_scalar1(a,i),  self_scalar2_pol(b,qidx,j), s12, kernel_idx, parameters)
!
!                     s12 = scalar(x1(a,i,:,:), x2(b,j,:,:), &
!                         & nneigh1(a,i), nneigh2(b,j), &
!                         & ksi1(a,i,:), ksi2_pol(b,qidx,j,:), &
!                         & sinp1(a,i,:,:,:), sinp2_pol(b,qidx,j,:,:,:), &
!                         & cosp1(a,i,:,:,:), cosp2_pol(b,qidx,j,:,:,:), &
!                         & t_width, d_width, cut_distance, order, &
!                         & pd, ang_norm2, distance_scale, angular_scale, alchemy)
!
!                     kernels(:, idx_a, idx_b) = kernels(:, idx_a, idx_b) &
!                         & + kernel(self_scalar1(a,i),  self_scalar2_pol(b,qidx,j), s12, &
!                         & kernel_idx, parameters)
!
!                     test_minus(:) = kernel(self_scalar1(a,i),  self_scalar2_pol(b,qidx,j), s12, &
!                         & kernel_idx, parameters)
!
!                     ! write(*,*) s12, kernel(self_scalar1(a,i),  self_scalar2_pol(b,qidx,j), s12, kernel_idx, parameters)
!
!                     write(*,*) kernels(:, idx_a, idx_b)
!                     kernels(:, idx_a, idx_b) = kernels(:, idx_a, idx_b) - 2* unperturbed(:)
!                     write(*,*) kernels(:, idx_a, idx_b)
!                     !  write(*,*) unperturbed(:)
!                     !  write(*,*) - 2* unperturbed(:)
!                     !  write(*,*) test_plus(:)
!                     !  write(*,*) test_minus(:)
!                     !  write(*,*) kernels(:, idx_a, idx_b)
!
!                 enddo
!             enddo
!         enddo
!
!     enddo
!     enddo
!     ! !$OMP END PARALLEL DO
!
!     t_end = omp_get_wtime()
!     write (*,"(A,F12.4,A)") "                    Time = ", t_end - t_start, " s"
!
!     write(*,*) "DF = ", df
!
!     write (*,*) kernels(1,1,1)
!     kernels(:,:,:) = kernels(:,:,:) / (df * df)
!     write (*,*) kernels(1,1,1)
!
!     deallocate(self_scalar1)
!     deallocate(self_scalar2_pol)
!     deallocate(ksi1)
!     deallocate(ksi2_pol)
!     deallocate(cosp1)
!     deallocate(cosp2_pol)
!     deallocate(sinp1)
!     deallocate(sinp2_pol)
!
! end subroutine fget_kernels_fchl_ef_2ndderiv


