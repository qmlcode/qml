subroutine fget_gaussian_process_kernels_fchl(x1, x2, verbose, n1, n2, nneigh1, nneigh2, &
       & nm1, nm2, naq2, nsigmas, &
       & t_width, d_width, cut_start, cut_distance, order, pd, &
       & distance_scale, angular_scale, alchemy, two_body_power, three_body_power, dx, &
       & kernel_idx, parameters, kernels)

    use ffchl_module, only: scalar, get_angular_norm2, &
        & get_pmax, get_ksi, init_cosp_sinp, get_selfscalar, &
        & get_pmax_displaced, get_ksi_displaced, init_cosp_sinp_displaced, get_selfscalar_displaced

    use ffchl_kernels, only: kernel
    
    use omp_lib, only: omp_get_wtime

    implicit none

    ! fchl descriptors for the training set, format (nm1,maxatoms,5,maxneighbors)
    double precision, dimension(:,:,:,:), intent(in) :: x1
    
    ! fchl descriptors for the training set, format (nm2,3,2,maxatoms,maxatoms,5,maxneighbors)
    double precision, dimension(:,:,:,:,:,:,:), intent(in) :: x2

    ! Whether to be verbose with output
    logical, intent(in) :: verbose
    
    ! List of numbers of atoms in each molecule
    integer, dimension(:), intent(in) :: n1
    integer, dimension(:), intent(in) :: n2

    ! Number of neighbors for each atom in each compound
    integer, dimension(:,:), intent(in) :: nneigh1
    integer, dimension(:,:,:,:,:), intent(in) :: nneigh2

    ! Number of molecules
    integer, intent(in) :: nm1
    integer, intent(in) :: nm2
   
    ! Total number of force components 
    integer, intent(in) :: naq2

    ! Number of kernels
    integer, intent(in) :: nsigmas
    
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

    ! Displacement for numerical differentiation
    double precision, intent(in) :: dx

    ! Kernel ID and corresponding parameters
    integer, intent(in) :: kernel_idx
    double precision, dimension(:,:), intent(in) :: parameters

    ! Resulting kernel matrix
    double precision, dimension(nsigmas,nm1+naq2,nm1+naq2), intent(out) :: kernels

    ! Internal counters
    integer :: i1, i2, j1, j2
    integer :: na, nb
    integer :: a, b

    ! Temporary variables necessary for parallelization
    double precision :: s12

    ! Pre-computed terms in the full distance matrix
    double precision, allocatable, dimension(:,:) :: self_scalar1
    double precision, allocatable, dimension(:,:,:,:,:) :: self_scalar2

    ! Pre-computed two-body weights
    double precision, allocatable, dimension(:,:,:) :: ksi1
    double precision, allocatable, dimension(:,:,:,:,:,:) :: ksi2

    ! Pre-computed terms for the Fourier expansion of the three-body term
    double precision, allocatable, dimension(:,:,:,:,:) :: sinp1
    double precision, allocatable, dimension(:,:,:,:,:) :: cosp1

    ! Pre-computed terms for the Fourier expansion of the three-body term
    double precision, allocatable, dimension(:,:,:,:,:,:,:) :: sinp2
    double precision, allocatable, dimension(:,:,:,:,:,:,:) :: cosp2

    ! Indexes for numerical differentiation 
    integer :: xyz_pm1
    integer :: xyz_pm2
    integer :: idx1, idx2
    integer :: xyz1, pm1
    integer :: xyz2, pm2

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

    ! Angular normalization constant
    ang_norm2 = get_angular_norm2(t_width)

    if (verbose) write (*,*) "CLEARING KERNEL MEM"
    kernels = 0.0d0

    ! Max number of neighbors in the representations
    maxneigh1 = maxval(nneigh1)
    maxneigh2 = maxval(nneigh2)

    ! pmax = max nuclear charge
    pmax1 = get_pmax(x1, n1)
    pmax2 = get_pmax_displaced(x2, n2)

    ! Get two-body weight function
    ksi1 = get_ksi(x1, n1, nneigh1, two_body_power, cut_start, cut_distance, verbose)
    ksi2 = get_ksi_displaced(x2, n2, nneigh2, two_body_power, cut_start, cut_distance, verbose)
    
    ! Allocate three-body Fourier terms
    allocate(cosp1(nm1, maxval(n1), pmax1, order, maxneigh1))
    allocate(sinp1(nm1, maxval(n1), pmax1, order, maxneigh1))

    ! Initialize and pre-calculate three-body Fourier terms
    call init_cosp_sinp(x1, n1, nneigh1, three_body_power, order, cut_start,  cut_distance, &
        & cosp1, sinp1, verbose)
    
    ! Allocate three-body Fourier terms
    allocate(cosp2(nm2, 3*2, maxval(n2), maxval(n2), pmax2, order, maxneigh2))
    allocate(sinp2(nm2, 3*2, maxval(n2), maxval(n2), pmax2, order, maxneigh2))
    
    ! Initialize and pre-calculate three-body Fourier terms
    call init_cosp_sinp_displaced(x2, n2, nneigh2, three_body_power, order, cut_start, &
        & cut_distance, cosp2, sinp2, verbose)
   
    ! Pre-calculate self-scalar terms 
    self_scalar1 = get_selfscalar(x1, nm1, n1, nneigh1, ksi1, sinp1, cosp1, t_width, d_width, &
         & cut_distance, order, pd, ang_norm2,distance_scale, angular_scale, alchemy, verbose)

    ! Pre-calculate self-scalar terms 
    self_scalar2 = get_selfscalar_displaced(x2, nm2, n2, nneigh2, ksi2, sinp2, cosp2, t_width, &
    & d_width, cut_distance, order, pd, ang_norm2,distance_scale, angular_scale, alchemy, verbose)
 
    t_start = omp_get_wtime()
    if (verbose) write (*,"(A)", advance="no") "KERNEL"
    
    !$OMP PARALLEL DO schedule(dynamic) PRIVATE(na,nb,s12)
    do a = 1, nm1
    na = n1(a)
    do j1 = 1, na
            
        do b = 1, nm1
        nb = n1(b)
        do j2 = 1, nb

            s12 = scalar(x1(a,j1,:,:), x1(b,j2,:,:), &
                & nneigh1(a,j1), nneigh1(b,j2), &
                & ksi1(a,j1,:), ksi1(b,j2,:), &
                & sinp1(a,j1,:,:,:), sinp1(b,j2,:,:,:), &
                & cosp1(a,j1,:,:,:), cosp1(b,j2,:,:,:), &
                & t_width, d_width, cut_distance, order, &
                & pd, ang_norm2,distance_scale, angular_scale, alchemy)

            kernels(:, a, b) = kernels(:, a, b) &
                & + kernel(self_scalar1(a,j1),  self_scalar1(b,j2), s12, &
                & kernel_idx, parameters)

        enddo
        enddo
    enddo
    enddo
    !$OMP END PARALLEL do
    
    t_end = omp_get_wtime()
    if (verbose) write (*,"(A,F12.4,A)") "                                  Time = ", t_end - t_start, " s"

    t_start = omp_get_wtime()
    if (verbose) write (*,"(A)", advance="no") "KERNEL GRADIENT"
    !$OMP PARALLEL DO schedule(dynamic) PRIVATE(na,nb,xyz_pm2,s12),&
    !$OMP& PRIVATE(idx1,idx2)
    do a = 1, nm1
    na = n1(a)
    idx1 = a
    do j1 = 1, na
            
        do b = 1, nm2
        nb = n2(b)
        do xyz2 = 1, 3
        do pm2 = 1, 2
        xyz_pm2 = 2*xyz2 + pm2 - 2
        do i2 = 1, nb
        idx2 = (sum(n2(:b)) - n2(b))* 3 + (i2 - 1) * 3  + xyz2 + nm1
        do j2 = 1, nb

            s12 = scalar(x1(a,j1,:,:), x2(b,xyz2,pm2,i2,j2,:,:), &
                & nneigh1(a,j1), nneigh2(b,xyz2,pm2,i2,j2), &
                & ksi1(a,j1,:), ksi2(b,xyz2,pm2,i2,j2,:), &
                & sinp1(a,j1,:,:,:), sinp2(b,xyz_pm2,i2,j2,:,:,:), &
                & cosp1(a,j1,:,:,:), cosp2(b,xyz_pm2,i2,j2,:,:,:), &
                & t_width, d_width, cut_distance, order, &
                & pd, ang_norm2,distance_scale, angular_scale, alchemy)

            if (pm2 == 2) then

                kernels(:, idx1, idx2) = kernels(:, idx1,idx2) &
                        & + kernel(self_scalar1(a,j1),  self_scalar2(b,xyz2,pm2,i2,j2), s12, &
                        & kernel_idx, parameters)

                kernels(:,idx2,idx1) = kernels(:,idx1,idx2)

            else
                
                kernels(:, idx1, idx2) = kernels(:, idx1,idx2) &
                        & - kernel(self_scalar1(a,j1),  self_scalar2(b,xyz2,pm2,i2,j2), s12, &
                        & kernel_idx, parameters)

                kernels(:,idx2,idx1) = kernels(:,idx1,idx2)

            end if

        enddo
        enddo
        enddo
        enddo
        enddo
    enddo
    enddo
    !$OMP END PARALLEL do

    kernels(:,:nm1,nm1+1:) = kernels(:,:nm1,nm1+1:) / (2 * dx)
    kernels(:,nm1+1:,:nm1) = kernels(:,nm1+1:,:nm1) / (2 * dx)
    
    t_end = omp_get_wtime()
    if (verbose) write (*,"(A,F12.4,A)") "                         Time = ", t_end - t_start, " s"
         
    t_start = omp_get_wtime()
    if (verbose) write (*,"(A)", advance="no") "KERNEL HESSIAN"
    
    !$OMP PARALLEL DO schedule(dynamic) PRIVATE(na,nb,xyz_pm1,xyz_pm2,s12),&
    !$OMP& PRIVATE(idx1,idx2)
    do a = 1, nm1
    na = n1(a)
    do xyz1 = 1, 3
    do pm1 = 1, 2
    xyz_pm1 = 2*xyz1 + pm1 - 2
    do i1 = 1, na
    idx1 = (sum(n1(:a)) - n1(a))* 3 + (i1 - 1) * 3  + xyz1 + nm1
    do j1 = 1, na
            
        do b = a, nm1
        nb = n1(b)
        do xyz2 = 1, 3
        do pm2 = 1, 2
        xyz_pm2 = 2*xyz2 + pm2 - 2
        do i2 = 1, nb
        idx2 = (sum(n1(:b)) - n1(b))* 3 + (i2 - 1) * 3  + xyz2 + nm1
        do j2 = 1, nb


            s12 = scalar(x2(a,xyz1,pm1,i1,j1,:,:), x2(b,xyz2,pm2,i2,j2,:,:), &
                & nneigh2(a,xyz1,pm1,i1,j1), nneigh2(b,xyz2,pm2,i2,j2), &
                & ksi2(a,xyz1,pm1,i1,j1,:), ksi2(b,xyz2,pm2,i2,j2,:), &
                & sinp2(a,xyz_pm1,i1,j1,:,:,:), sinp2(b,xyz_pm2,i2,j2,:,:,:), &
                & cosp2(a,xyz_pm1,i1,j1,:,:,:), cosp2(b,xyz_pm2,i2,j2,:,:,:), &
                & t_width, d_width, cut_distance, order, &
                & pd, ang_norm2,distance_scale, angular_scale, alchemy)

            if (pm1 == pm2) then

                kernels(:, idx1, idx2) = kernels(:, idx1, idx2) &
                    & + kernel(self_scalar2(a,xyz1,pm1,i1,j1), self_scalar2(b,xyz2,pm2,i2,j2), s12,&
                    & kernel_idx, parameters)
                
                if (a /= b) then
                    kernels(:, idx2, idx1) = kernels(:, idx2, idx1) &
                        & + kernel(self_scalar2(a,xyz1,pm1,i1,j1), self_scalar2(b,xyz2,pm2,i2,j2), s12,&
                        & kernel_idx, parameters)
                endif
                
            else
                kernels(:, idx1, idx2) = kernels(:, idx1, idx2) &
                    & - kernel(self_scalar2(a,xyz1,pm1,i1,j1), self_scalar2(b,xyz2,pm2,i2,j2), s12,&
                    & kernel_idx, parameters)
                
                if (a /= b) then
                    kernels(:, idx2, idx1) = kernels(:, idx2, idx1) &
                        & - kernel(self_scalar2(a,xyz1,pm1,i1,j1), self_scalar2(b,xyz2,pm2,i2,j2), s12,&
                        & kernel_idx, parameters)
                endif

            end if


        enddo
        enddo
        enddo
        enddo
        enddo
    enddo
    enddo
    enddo
    enddo
    enddo
    !$OMP END PARALLEL do

    kernels(:,nm1+1:,nm1+1:) = kernels(:,nm1+1:,nm1+1:) / (4 * dx**2)
    
    t_end = omp_get_wtime()
    if (verbose) write (*,"(A,F12.4,A)") "                          Time = ", t_end - t_start, " s"

end subroutine fget_gaussian_process_kernels_fchl


subroutine fget_local_gradient_kernels_fchl(x1, x2, verbose, n1, n2, nneigh1, nneigh2, &
       & nm1, nm2, naq2, nsigmas, &
       & t_width, d_width, cut_start, cut_distance, order, pd, &
       & distance_scale, angular_scale, alchemy, two_body_power, three_body_power, dx, &
       & kernel_idx, parameters, kernels)

    use ffchl_module, only: scalar, get_angular_norm2, &
        & get_pmax, get_ksi, init_cosp_sinp, get_selfscalar, &
        & get_pmax_displaced, get_ksi_displaced, init_cosp_sinp_displaced, get_selfscalar_displaced

    use ffchl_kernels, only: kernel
    
    use omp_lib, only: omp_get_wtime

    implicit none

    ! fchl descriptors for the training set, format (nm1,maxatoms,5,maxneighbors)
    double precision, dimension(:,:,:,:), intent(in) :: x1
    
    ! fchl descriptors for the training set, format (nm2,3,2,maxatoms,maxatoms,5,maxneighbors)
    double precision, dimension(:,:,:,:,:,:,:), intent(in) :: x2

    ! Whether to be verbose with output
    logical, intent(in) :: verbose
    
    ! List of numbers of atoms in each molecule
    integer, dimension(:), intent(in) :: n1
    integer, dimension(:), intent(in) :: n2

    ! Number of neighbors for each atom in each compound
    integer, dimension(:,:), intent(in) :: nneigh1
    integer, dimension(:,:,:,:,:), intent(in) :: nneigh2

    ! Number of molecules
    integer, intent(in) :: nm1
    integer, intent(in) :: nm2
   
    ! Total number of force components 
    integer, intent(in) :: naq2

    ! Number of kernels
    integer, intent(in) :: nsigmas
    
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

    ! Displacement for numerical differentiation
    double precision, intent(in) :: dx

    ! Kernel ID and corresponding parameters
    integer, intent(in) :: kernel_idx
    double precision, dimension(:,:), intent(in) :: parameters

    ! Resulting alpha vector
    double precision, dimension(nsigmas,nm1,naq2), intent(out) :: kernels

    ! Internal counters
    integer :: i2, j1, j2
    integer :: na, nb
    integer :: a, b
    
    ! Temporary variables necessary for parallelization
    double precision :: s12

    ! Pre-computed terms in the full distance matrix
    double precision, allocatable, dimension(:,:) :: self_scalar1
    double precision, allocatable, dimension(:,:,:,:,:) :: self_scalar2

    ! Pre-computed two-body weights
    double precision, allocatable, dimension(:,:,:) :: ksi1
    double precision, allocatable, dimension(:,:,:,:,:,:) :: ksi2

    ! Pre-computed terms for the Fourier expansion of the three-body term
    double precision, allocatable, dimension(:,:,:,:,:) :: sinp1
    double precision, allocatable, dimension(:,:,:,:,:) :: cosp1

    ! Pre-computed terms for the Fourier expansion of the three-body term
    double precision, allocatable, dimension(:,:,:,:,:,:,:) :: sinp2
    double precision, allocatable, dimension(:,:,:,:,:,:,:) :: cosp2

    ! Indexes for numerical differentiation 
    integer :: idx1, idx2
    integer :: xyz_pm2
    integer :: xyz2, pm2

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

    if (verbose) write (*,*) "INIT, dx =", dx

    if (verbose) write (*,*) "CLEARING KERNEL MEM"
    kernels = 0.0d0

    ! Angular normalization constant
    ang_norm2 = get_angular_norm2(t_width)

    ! Max number of neighbors in the representations
    maxneigh1 = maxval(nneigh1)
    maxneigh2 = maxval(nneigh2)

    ! pmax = max nuclear charge
    pmax1 = get_pmax(x1, n1)
    pmax2 = get_pmax_displaced(x2, n2)

    ! Get two-body weight function
    ksi1 = get_ksi(x1, n1, nneigh1, two_body_power, cut_start, cut_distance, verbose)
    ksi2 = get_ksi_displaced(x2, n2, nneigh2, two_body_power, cut_start, cut_distance, verbose)
    
    ! Allocate three-body Fourier terms
    allocate(cosp1(nm1, maxval(n1), pmax1, order, maxneigh1))
    allocate(sinp1(nm1, maxval(n1), pmax1, order, maxneigh1))
    
    ! Initialize and pre-calculate three-body Fourier terms
    call init_cosp_sinp(x1, n1, nneigh1, three_body_power, order, cut_start,  cut_distance, &
        & cosp1, sinp1, verbose)
    
    ! Allocate three-body Fourier terms
    allocate(cosp2(nm2, 3*2, maxval(n2), maxval(n2), pmax2, order, maxneigh2))
    allocate(sinp2(nm2, 3*2, maxval(n2), maxval(n2), pmax2, order, maxneigh2))
    
    ! Initialize and pre-calculate three-body Fourier terms
    call init_cosp_sinp_displaced(x2, n2, nneigh2, three_body_power, order, cut_start, &
        & cut_distance, cosp2, sinp2, verbose)
    
    ! Pre-calculate self-scalar terms 
    self_scalar1 = get_selfscalar(x1, nm1, n1, nneigh1, ksi1, sinp1, cosp1, t_width, d_width, &
         & cut_distance, order, pd, ang_norm2,distance_scale, angular_scale, alchemy, verbose)

    ! Pre-calculate self-scalar terms 
    self_scalar2 = get_selfscalar_displaced(x2, nm2, n2, nneigh2, ksi2, sinp2, cosp2, t_width, &
    & d_width, cut_distance, order, pd, ang_norm2,distance_scale, angular_scale, alchemy, verbose)

    t_start = omp_get_wtime()
    if (verbose) write (*,"(A)", advance="no") "KERNEL GRADIENT"

    !$OMP PARALLEL DO schedule(dynamic) PRIVATE(na,nb,xyz_pm2,s12),&
    !$OMP& PRIVATE(idx1,idx2)
    do a = 1, nm1
    na = n1(a)
    idx1 = a
    do j1 = 1, na
            
        do b = 1, nm2
        nb = n2(b)
        do xyz2 = 1, 3
        do pm2 = 1, 2
        xyz_pm2 = 2*xyz2 + pm2 - 2
        do i2 = 1, nb
        idx2 = (sum(n2(:b)) - n2(b))* 3 + (i2 - 1) * 3  + xyz2
        do j2 = 1, nb


            s12 = scalar(x1(a,j1,:,:), x2(b,xyz2,pm2,i2,j2,:,:), &
                & nneigh1(a,j1), nneigh2(b,xyz2,pm2,i2,j2), &
                & ksi1(a,j1,:), ksi2(b,xyz2,pm2,i2,j2,:), &
                & sinp1(a,j1,:,:,:), sinp2(b,xyz_pm2,i2,j2,:,:,:), &
                & cosp1(a,j1,:,:,:), cosp2(b,xyz_pm2,i2,j2,:,:,:), &
                & t_width, d_width, cut_distance, order, &
                & pd, ang_norm2,distance_scale, angular_scale, alchemy)
            if (pm2 == 2) then

                kernels(:,idx1,idx2) = kernels(:,idx1,idx2) & 
                    & + kernel(self_scalar1(a,j1), self_scalar2(b,xyz2,pm2,i2,j2), s12, &
                    & kernel_idx, parameters)
            else
                kernels(:,idx1,idx2) = kernels(:,idx1,idx2) & 
                    & - kernel(self_scalar1(a,j1), self_scalar2(b,xyz2,pm2,i2,j2), s12, &
                    & kernel_idx, parameters)
            end if


        enddo
        enddo
        enddo
        enddo
        enddo
    enddo
    enddo
    !$OMP END PARALLEL do

    kernels = kernels / (2 * dx)
    
    t_end = omp_get_wtime()
    if (verbose) write (*,"(A,F12.4,A)") "                         Time = ", t_end - t_start, " s"

end subroutine fget_local_gradient_kernels_fchl


subroutine fget_local_hessian_kernels_fchl(x1, x2, verbose, n1, n2, nneigh1, nneigh2, &
       &  nm1, nm2, naq1, naq2, nsigmas, &
       & t_width, d_width, cut_start, cut_distance, order, pd, &
       & distance_scale, angular_scale, alchemy, two_body_power, three_body_power, dx, &
       & kernel_idx, parameters, kernels)

    use ffchl_module, only: scalar, get_angular_norm2, &
        & get_pmax_displaced, get_ksi_displaced, init_cosp_sinp_displaced, get_selfscalar_displaced

    use ffchl_kernels, only: kernel
    
    use omp_lib, only: omp_get_wtime

    implicit none

    ! fchl descriptors for the training set, format (nm1,3,2,maxatoms,maxatoms,5,maxneighbors)
    double precision, dimension(:,:,:,:,:,:,:), intent(in) :: x1
    double precision, dimension(:,:,:,:,:,:,:), intent(in) :: x2

    ! Whether to be verbose with output
    logical, intent(in) :: verbose
    
    ! List of numbers of atoms in each molecule
    integer, dimension(:), intent(in) :: n1
    integer, dimension(:), intent(in) :: n2

    ! Number of neighbors for each atom in each compound
    integer, dimension(:,:,:,:,:), intent(in) :: nneigh1
    integer, dimension(:,:,:,:,:), intent(in) :: nneigh2

    ! Number of molecules
    integer, intent(in) :: nm1
    integer, intent(in) :: nm2
    
    ! Total number of force components 
    integer, intent(in) :: naq1
    integer, intent(in) :: naq2
    
    ! Number of kernels
    integer, intent(in) :: nsigmas
    
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

    ! Displacement for numerical differentiation
    double precision, intent(in) :: dx

    ! Kernel ID and corresponding parameters
    integer, intent(in) :: kernel_idx
    double precision, dimension(:,:), intent(in) :: parameters

    ! Resulting alpha vector
    double precision, dimension(nsigmas,naq1,naq2), intent(out) :: kernels

    ! Internal counters
    integer :: i1, i2, j1, j2
    integer :: na, nb
    integer :: a, b

    ! Temporary variables necessary for parallelization
    double precision :: s12

    ! Pre-computed terms in the full distance matrix
    double precision, allocatable, dimension(:,:,:,:,:) :: self_scalar1
    double precision, allocatable, dimension(:,:,:,:,:) :: self_scalar2

    ! Pre-computed two-body weights
    double precision, allocatable, dimension(:,:,:,:,:,:) :: ksi1
    double precision, allocatable, dimension(:,:,:,:,:,:) :: ksi2

    ! Pre-computed terms for the Fourier expansion of the three-body term
    double precision, allocatable, dimension(:,:,:,:,:,:,:) :: sinp1
    double precision, allocatable, dimension(:,:,:,:,:,:,:) :: cosp1
    
    ! Pre-computed terms for the Fourier expansion of the three-body term
    double precision, allocatable, dimension(:,:,:,:,:,:,:) :: sinp2
    double precision, allocatable, dimension(:,:,:,:,:,:,:) :: cosp2

    ! Indexes for numerical differentiation 
    integer :: xyz_pm1
    integer :: xyz_pm2
    integer :: idx1, idx2
    integer :: xyz1, pm1
    integer :: xyz2, pm2
    
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

    ! Angular normalization constant
    ang_norm2 = get_angular_norm2(t_width)

    if (verbose) write (*,*) "INIT, dx =", dx

    if (verbose) write (*,*) "CLEARING KERNEL MEM"
    kernels = 0.0d0

    ! Max number of neighbors in the representations
    maxneigh1 = maxval(nneigh1)
    maxneigh2 = maxval(nneigh2)

    ! pmax = max nuclear charge
    pmax1 = get_pmax_displaced(x1, n1)
    pmax2 = get_pmax_displaced(x2, n2)

    ! Get two-body weight function
    ksi1 = get_ksi_displaced(x1, n1, nneigh1, two_body_power, cut_start, cut_distance, verbose)
    ksi2 = get_ksi_displaced(x2, n2, nneigh2, two_body_power, cut_start, cut_distance, verbose)
    
    ! Allocate three-body Fourier terms
    allocate(cosp1(nm1, 3*2, maxval(n1), maxval(n1), pmax1, order, maxval(nneigh1)))
    allocate(sinp1(nm1, 3*2, maxval(n1), maxval(n1), pmax1, order, maxval(nneigh1)))
    
    ! Initialize and pre-calculate three-body Fourier terms
    call init_cosp_sinp_displaced(x1, n1, nneigh1, three_body_power, order, cut_start,  cut_distance, &
        & cosp1, sinp1, verbose)
    
    ! Initialize and pre-calculate three-body Fourier terms
    allocate(cosp2(nm2, 3*2, maxval(n2), maxval(n2), pmax2, order, maxneigh2))
    allocate(sinp2(nm2, 3*2, maxval(n2), maxval(n2), pmax2, order, maxneigh2))
    
    ! Initialize and pre-calculate three-body Fourier terms
    call init_cosp_sinp_displaced(x2, n2, nneigh2, three_body_power, order, cut_start, &
        & cut_distance, cosp2, sinp2, verbose)
    
    ! Pre-calculate self-scalar terms 
    self_scalar1 = get_selfscalar_displaced(x1, nm1, n1, nneigh1, ksi1, sinp1, cosp1, t_width, &
    & d_width, cut_distance, order, pd, ang_norm2,distance_scale, angular_scale, alchemy, verbose)

    ! Pre-calculate self-scalar terms 
    self_scalar2 = get_selfscalar_displaced(x2, nm2, n2, nneigh2, ksi2, sinp2, cosp2, t_width, &
    & d_width, cut_distance, order, pd, ang_norm2,distance_scale, angular_scale, alchemy, verbose)

    t_start = omp_get_wtime()
    if (verbose) write (*,"(A)", advance="no") "KERNEL HESSIAN"
    
    !$OMP PARALLEL DO schedule(dynamic) PRIVATE(na,nb,xyz_pm1,xyz_pm2,s12),&
    !$OMP& PRIVATE(idx1,idx2)
    do a = 1, nm1
    na = n1(a)
    do xyz1 = 1, 3
    do pm1 = 1, 2
    xyz_pm1 = 2*xyz1 + pm1 - 2
    do i1 = 1, na
    idx1 = (sum(n1(:a)) - n1(a))* 3 + (i1 - 1) * 3  + xyz1
    do j1 = 1, na
            
        do b = 1, nm2
        nb = n2(b)
        do xyz2 = 1, 3
        do pm2 = 1, 2
        xyz_pm2 = 2*xyz2 + pm2 - 2
        do i2 = 1, nb
        idx2 = (sum(n2(:b)) - n2(b))* 3 + (i2 - 1) * 3  + xyz2
        do j2 = 1, nb


            s12 = scalar(x1(a,xyz1,pm1,i1,j1,:,:), x2(b,xyz2,pm2,i2,j2,:,:), &
                & nneigh1(a,xyz1,pm1,i1,j1), nneigh2(b,xyz2,pm2,i2,j2), &
                & ksi1(a,xyz1,pm1,i1,j1,:), ksi2(b,xyz2,pm2,i2,j2,:), &
                & sinp1(a,xyz_pm1,i1,j1,:,:,:), sinp2(b,xyz_pm2,i2,j2,:,:,:), &
                & cosp1(a,xyz_pm1,i1,j1,:,:,:), cosp2(b,xyz_pm2,i2,j2,:,:,:), &
                & t_width, d_width, cut_distance, order, &
                & pd, ang_norm2,distance_scale, angular_scale, alchemy)

            if (pm1 == pm2) then

                kernels(:, idx1, idx2) = kernels(:, idx1, idx2) &
                    & + kernel(self_scalar1(a,xyz1,pm1,i1,j1), self_scalar2(b,xyz2,pm2,i2,j2), s12,&
                    & kernel_idx, parameters)
                
            else
                kernels(:, idx1, idx2) = kernels(:, idx1, idx2) &
                    & - kernel(self_scalar1(a,xyz1,pm1,i1,j1), self_scalar2(b,xyz2,pm2,i2,j2), s12,&
                    & kernel_idx, parameters)

            end if


        enddo
        enddo
        enddo
        enddo
        enddo
    enddo
    enddo
    enddo
    enddo
    enddo
    !$OMP END PARALLEL do

    kernels = kernels / (4 * dx**2)
    
    t_end = omp_get_wtime()
    if (verbose) write (*,"(A,F12.4,A)") "                          Time = ", t_end - t_start, " s"

end subroutine fget_local_hessian_kernels_fchl


subroutine fget_local_symmetric_hessian_kernels_fchl(x1, verbose, n1, nneigh1, &
       & nm1, naq1, nsigmas, &
       & t_width, d_width, cut_start, cut_distance, order, pd, &
       & distance_scale, angular_scale, alchemy, two_body_power, three_body_power, dx, &
       & kernel_idx, parameters, kernels)

    use ffchl_module, only: scalar, get_angular_norm2, &
        & get_pmax_displaced, get_ksi_displaced, init_cosp_sinp_displaced, get_selfscalar_displaced

    use ffchl_kernels, only: kernel
    
    use omp_lib, only: omp_get_wtime

    implicit none

    ! fchl descriptors for the training set, format (nm1,3,2,maxatoms,maxatoms,5,maxneighbors)
    double precision, dimension(:,:,:,:,:,:,:), intent(in) :: x1

    ! Whether to be verbose with output
    logical, intent(in) :: verbose
    
    ! List of numbers of atoms in each molecule
    integer, dimension(:), intent(in) :: n1

    ! Number of neighbors for each atom in each compound
    integer, dimension(:,:,:,:,:), intent(in) :: nneigh1

    ! Number of molecules
    integer, intent(in) :: nm1
    
    ! Total number of force components 
    integer, intent(in) :: naq1

    ! Number of kernels
    integer, intent(in) :: nsigmas
    
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

    ! Displacement for numerical differentiation
    double precision, intent(in) :: dx

    ! Kernel ID and corresponding parameters
    integer, intent(in) :: kernel_idx
    double precision, dimension(:,:), intent(in) :: parameters

    ! Resulting alpha vector
    double precision, dimension(nsigmas,naq1,naq1), intent(out) :: kernels

    ! Internal counters
    integer :: i1, i2, j1, j2
    integer :: na, nb
    integer :: a, b

    ! Temporary variables necessary for parallelization
    double precision :: s12 

    ! Pre-computed terms in the full distance matrix
    double precision, allocatable, dimension(:,:,:,:,:) :: self_scalar1

    ! Pre-computed two-body weights
    double precision, allocatable, dimension(:,:,:,:,:,:) :: ksi1

    ! Pre-computed terms for the Fourier expansion of the three-body term
    double precision, allocatable, dimension(:,:,:,:,:,:,:) :: sinp1
    double precision, allocatable, dimension(:,:,:,:,:,:,:) :: cosp1

    ! Indexes for numerical differentiation 
    integer :: xyz_pm1
    integer :: xyz_pm2
    integer :: xyz1, pm1
    integer :: xyz2, pm2
    integer :: idx1, idx2

    ! Max index in the periodic table
    integer :: pmax1

    ! Angular normalization constant
    double precision :: ang_norm2
   
    ! Max number of neighbors 
    integer :: maxneigh1

    ! Variables to calculate time 
    double precision :: t_start, t_end

    ! Angular normalization constant
    ang_norm2 = get_angular_norm2(t_width)

    if (verbose) write (*,*) "INIT, dx =", dx

    if (verbose) write (*,*) "CLEARING KERNEL MEM"
    kernels = 0.0d0

    ! Max number of neighbors 
    maxneigh1 = maxval(nneigh1)

    ! pmax = max nuclear charge
    pmax1 = get_pmax_displaced(x1, n1)

    ! Get two-body weight function
    ksi1 = get_ksi_displaced(x1, n1, nneigh1, two_body_power, cut_start, cut_distance, verbose)

    ! Allocate three-body Fourier terms
    allocate(cosp1(nm1, 3*2, maxval(n1), maxval(n1), pmax1, order, maxval(nneigh1)))
    allocate(sinp1(nm1, 3*2, maxval(n1), maxval(n1), pmax1, order, maxval(nneigh1)))
    
    ! Initialize and pre-calculate three-body Fourier terms
    call init_cosp_sinp_displaced(x1, n1, nneigh1, three_body_power, order, cut_start,  cut_distance, &
        & cosp1, sinp1, verbose)
    
    ! Pre-calculate self-scalar terms 
    self_scalar1 = get_selfscalar_displaced(x1, nm1, n1, nneigh1, ksi1, sinp1, cosp1, t_width,&
    & d_width, cut_distance, order, pd, ang_norm2,distance_scale, angular_scale, alchemy, verbose)

    t_start = omp_get_wtime()
    if (verbose) write (*,"(A)", advance="no") "KERNEL HESSIAN"

    !$OMP PARALLEL DO schedule(dynamic) PRIVATE(na,nb,xyz_pm1,xyz_pm2,s12),&
    !$OMP& PRIVATE(idx1,idx2)
    do a = 1, nm1
    na = n1(a)
    do xyz1 = 1, 3
    do pm1 = 1, 2
    xyz_pm1 = 2*xyz1 + pm1 - 2
    do i1 = 1, na
    idx1 = (sum(n1(:a)) - n1(a))* 3 + (i1 - 1) * 3  + xyz1
    do j1 = 1, na
            
        do b = a, nm1
        nb = n1(b)
        do xyz2 = 1, 3
        do pm2 = 1, 2
        xyz_pm2 = 2*xyz2 + pm2 - 2
        do i2 = 1, nb
        idx2 = (sum(n1(:b)) - n1(b))* 3 + (i2 - 1) * 3  + xyz2
        do j2 = 1, nb


            s12 = scalar(x1(a,xyz1,pm1,i1,j1,:,:), x1(b,xyz2,pm2,i2,j2,:,:), &
                & nneigh1(a,xyz1,pm1,i1,j1), nneigh1(b,xyz2,pm2,i2,j2), &
                & ksi1(a,xyz1,pm1,i1,j1,:), ksi1(b,xyz2,pm2,i2,j2,:), &
                & sinp1(a,xyz_pm1,i1,j1,:,:,:), sinp1(b,xyz_pm2,i2,j2,:,:,:), &
                & cosp1(a,xyz_pm1,i1,j1,:,:,:), cosp1(b,xyz_pm2,i2,j2,:,:,:), &
                & t_width, d_width, cut_distance, order, &
                & pd, ang_norm2,distance_scale, angular_scale, alchemy)

            if (pm1 == pm2) then

                kernels(:, idx1, idx2) = kernels(:, idx1, idx2) &
                    & + kernel(self_scalar1(a,xyz1,pm1,i1,j1), self_scalar1(b,xyz2,pm2,i2,j2), s12,&
                    & kernel_idx, parameters)
                    if (a /= b) then
                        kernels(:,idx2,idx1) = kernels(:,idx2,idx1) & 
                            & + kernel(self_scalar1(a,xyz1,pm1,i1,j1), self_scalar1(b,xyz2,pm2,i2,j2), s12,&
                            & kernel_idx, parameters)
                    endif
                
            else
                kernels(:, idx1, idx2) = kernels(:, idx1, idx2) &
                    & - kernel(self_scalar1(a,xyz1,pm1,i1,j1), self_scalar1(b,xyz2,pm2,i2,j2), s12,&
                    & kernel_idx, parameters)

                    if (a /= b) then
                        kernels(:,idx2,idx1) = kernels(:,idx2,idx1) & 
                            & - kernel(self_scalar1(a,xyz1,pm1,i1,j1), self_scalar1(b,xyz2,pm2,i2,j2), s12,&
                            & kernel_idx, parameters)
                    endif

            end if

        enddo
        enddo
        enddo
        enddo
        enddo
    enddo
    enddo
    enddo
    enddo
    enddo
    !$OMP END PARALLEL do

    kernels = kernels / (4 * dx**2)
    
    t_end = omp_get_wtime()
    if (verbose) write (*,"(A,F12.4,A)") "                          Time = ", t_end - t_start, " s"

end subroutine fget_local_symmetric_hessian_kernels_fchl


subroutine fget_force_alphas_fchl(x1, x2, verbose, forces, energies, n1, n2, &
        & nneigh1, nneigh2,  nm1, nm2, na1, nsigmas, &
        & t_width, d_width, cut_start, cut_distance, order, pd, &
        & distance_scale, angular_scale, alchemy, two_body_power, three_body_power, dx, &
        & kernel_idx, parameters, llambda, alphas)

    use ffchl_module, only: scalar, get_angular_norm2, &
        & get_pmax, get_ksi, init_cosp_sinp, get_selfscalar, &
        & get_pmax_displaced, get_ksi_displaced, init_cosp_sinp_displaced, get_selfscalar_displaced

    use ffchl_kernels, only: kernel
    
    use omp_lib, only: omp_get_wtime

    implicit none

    ! fchl descriptors for the training set, format (nm1,3,2,maxatoms,maxatoms,5,maxneighbors)
    double precision, dimension(:,:,:,:), intent(in) :: x1
    double precision, dimension(:,:,:,:,:,:,:), intent(in) :: x2

    ! Whether to be verbose with output
    logical, intent(in) :: verbose
    
    double precision, dimension(:,:), intent(in) :: forces
    double precision, dimension(:), intent(in) :: energies

    ! List of numbers of atoms in each molecule
    integer, dimension(:), intent(in) :: n1
    integer, dimension(:), intent(in) :: n2

    ! Number of neighbors for each atom in each compound
    integer, dimension(:,:), intent(in) :: nneigh1
    integer, dimension(:,:,:,:,:), intent(in) :: nneigh2

    ! Number of molecules
    integer, intent(in) :: nm1
    integer, intent(in) :: nm2

    ! Number of atoms (and force components in each direction)    
    integer, intent(in) :: na1
    
    ! Number of kernels
    integer, intent(in) :: nsigmas
    
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

    ! Displacement for numerical differentiation
    double precision, intent(in) :: dx

    ! Kernel ID and corresponding parameters
    integer, intent(in) :: kernel_idx
    double precision, dimension(:,:), intent(in) :: parameters

    ! Regularization Lambda
    double precision, intent(in) :: llambda
    
    double precision, dimension(nsigmas,na1), intent(out) :: alphas

    ! Internal counters
    integer :: i, j, k, i2, j1, j2
    integer :: na, nb, ni, nj
    integer :: a, b

    ! Temporary variables necessary for parallelization
    double precision :: s12

    ! Pre-computed terms in the full distance matrix
    double precision, allocatable, dimension(:,:) :: self_scalar1
    double precision, allocatable, dimension(:,:,:,:,:) :: self_scalar2

    ! Pre-computed terms
    double precision, allocatable, dimension(:,:,:) :: ksi1
    double precision, allocatable, dimension(:,:,:,:,:,:) :: ksi2

    double precision, allocatable, dimension(:,:,:,:,:) :: sinp1
    double precision, allocatable, dimension(:,:,:,:,:) :: cosp1

    double precision, allocatable, dimension(:,:,:,:,:,:,:) :: sinp2
    double precision, allocatable, dimension(:,:,:,:,:,:,:) :: cosp2

    ! Indexes for numerical differentiation 
    integer :: xyz_pm2
    integer :: xyz2, pm2
    integer :: idx1
    integer :: idx2
    integer :: idx1_start
    integer :: idx2_start

    ! 1/(2*dx) 
    double precision :: inv_2dx

    ! Max index in the periodic table
    integer :: pmax1
    integer :: pmax2

    ! Angular normalization constant
    double precision :: ang_norm2
   
    ! Max number of neighbors 
    integer :: maxneigh1
    integer :: maxneigh2

    ! Info variable for BLAS/LAPACK calls
    integer :: info

    ! Feature vector multiplied by the kernel derivatives
    double precision, allocatable, dimension(:,:) :: y
   
    ! Numerical derivatives of kernel 
    double precision, allocatable, dimension(:,:,:)  :: kernel_delta

    ! Scratch space for products of the kernel derivatives
    double precision, allocatable, dimension(:,:,:)  :: kernel_scratch
   
    ! Variables to calculate time 
    double precision :: t_start, t_end
   
    ! Kernel between molecules and atom 
    double precision, allocatable, dimension(:,:,:) :: kernel_ma

    inv_2dx = 1.0d0 / (2.0d0 * dx)

    ! Angular normalization constant
    ang_norm2 = get_angular_norm2(t_width)

    if (verbose) write (*,*) "INIT, DX =", dx

    ! Max number of neighbors in the representations
    maxneigh1 = maxval(nneigh1)
    maxneigh2 = maxval(nneigh2)

    ! pmax = max nuclear charge
    pmax1 = get_pmax(x1, n1)
    pmax2 = get_pmax_displaced(x2, n2)

    ! Get two-body weight function
    ksi1 = get_ksi(x1, n1, nneigh1, two_body_power, cut_start, cut_distance, verbose)
    ksi2 = get_ksi_displaced(x2, n2, nneigh2, two_body_power, cut_start, cut_distance, verbose)
    
    ! Allocate three-body Fourier terms
    allocate(cosp1(nm1, maxval(n1), pmax1, order, maxneigh1))
    allocate(sinp1(nm1, maxval(n1), pmax1, order, maxneigh1))
    
    ! Initialize and pre-calculate three-body Fourier terms
    call init_cosp_sinp(x1, n1, nneigh1, three_body_power, order, cut_start,  cut_distance, &
        & cosp1, sinp1, verbose)
    
    ! Allocate three-body Fourier terms
    allocate(cosp2(nm2, 3*2, maxval(n2), maxval(n2), pmax2, order, maxneigh2))
    allocate(sinp2(nm2, 3*2, maxval(n2), maxval(n2), pmax2, order, maxneigh2))
    
    ! Initialize and pre-calculate three-body Fourier terms
    call init_cosp_sinp_displaced(x2, n2, nneigh2, three_body_power, order, cut_start, &
        & cut_distance, cosp2, sinp2, verbose)
    
    ! Pre-calculate self-scalar terms 
    self_scalar1 = get_selfscalar(x1, nm1, n1, nneigh1, ksi1, sinp1, cosp1, t_width, d_width, &
         & cut_distance, order, pd, ang_norm2,distance_scale, angular_scale, alchemy, verbose)

    ! Pre-calculate self-scalar terms 
    self_scalar2 = get_selfscalar_displaced(x2, nm2, n2, nneigh2, ksi2, sinp2, cosp2, t_width, &
    & d_width, cut_distance, order, pd, ang_norm2,distance_scale, angular_scale, alchemy, verbose)


    if (verbose) write (*,*) "CLEARING KERNEL MEM"

    allocate(kernel_delta(na1,na1,nsigmas))
    allocate(y(na1,nsigmas))
    y = 0.0d0

    allocate(kernel_scratch(na1,na1,nsigmas))
    kernel_scratch = 0.0d0

    ! Calculate kernel derivatives and add to kernel matrix
    do xyz2 = 1, 3
        
        if (verbose) write (*,"(A,I3,A)", advance="no") "KERNEL GRADIENT", xyz2, " / 3"
        t_start = omp_get_wtime()
        
        kernel_delta = 0.0d0

        !$OMP PARALLEL DO schedule(dynamic) PRIVATE(na,nb,xyz_pm2,s12), &
        !$OMP& PRIVATE(idx1,idx2,idx1_start,idx2_start)
        do a = 1, nm1
        na = n1(a)
        idx1_start = sum(n1(:a)) - na
        do j1 = 1, na
        idx1 = idx1_start + j1
                
            do b = 1, nm2
            nb = n2(b)
            idx2_start = (sum(n2(:b)) - nb)

            do pm2 = 1, 2
            xyz_pm2 = 2*xyz2 + pm2 - 2
            do i2 = 1, nb
            idx2 = idx2_start + i2
            do j2 = 1, nb


                s12 = scalar(x1(a,j1,:,:), x2(b,xyz2,pm2,i2,j2,:,:), &
                    & nneigh1(a,j1), nneigh2(b,xyz2,pm2,i2,j2), &
                    & ksi1(a,j1,:), ksi2(b,xyz2,pm2,i2,j2,:), &
                    & sinp1(a,j1,:,:,:), sinp2(b,xyz_pm2,i2,j2,:,:,:), &
                    & cosp1(a,j1,:,:,:), cosp2(b,xyz_pm2,i2,j2,:,:,:), &
                    & t_width, d_width, cut_distance, order, &
                    & pd, ang_norm2,distance_scale, angular_scale, alchemy)

                if (pm2 == 2) then

                    kernel_delta(idx1,idx2,:) = kernel_delta(idx1,idx2,:) & 
                        & + kernel(self_scalar1(a,j1), self_scalar2(b,xyz2,pm2,i2,j2), s12,&
                         kernel_idx, parameters) * inv_2dx
                else
                    kernel_delta(idx1,idx2,:) = kernel_delta(idx1,idx2,:) & 
                        & - kernel(self_scalar1(a,j1), self_scalar2(b,xyz2,pm2,i2,j2), s12,&
                         kernel_idx, parameters) * inv_2dx

                end if


            enddo
            enddo
            enddo
            enddo
            enddo
        enddo
        !$OMP END PARALLEL do
        
        t_end = omp_get_wtime()
        if (verbose) write (*,"(A,F12.4,A)") "                  Time = ", t_end - t_start, " s"

        do k = 1, nsigmas

            if (verbose) write (*,"(A,I12)", advance="no") "     DSYRK()    sigma =", k
            t_start = omp_get_wtime()
            
            call dsyrk("U", "N", na1, na1, 1.0d0, kernel_delta(1,1,k), na1, &
               & 1.0d0, kernel_scratch(1,1,k), na1)

            ! kernel_scratch(:,:,k) = kernel_scratch(:,:,k) &
            !    & + matmul(kernel_delta(:,:,k),transpose(kernel_delta(:,:,k)))! * inv_2dx*inv_2dx

            t_end = omp_get_wtime()
            if (verbose) write (*,"(A,F12.4,A)") "     Time = ", t_end - t_start, " s"

            if (verbose) write (*,"(A,I12)", advance="no") "     DGEMV()    sigma =", k
            t_start = omp_get_wtime()
            
            call dgemv("N", na1, na1, 1.0d0, kernel_delta(:,:,k), na1, &
                & forces(:,xyz2), 1, 1.0d0, y(:,k), 1)
            
            ! y(:,k) = y(:,k) + matmul(kernel_delta(:,:,k), forces(:,xyz2))!* inv_2dx

            t_end = omp_get_wtime()
            if (verbose) write (*,"(A,F12.4,A)") "     Time = ", t_end - t_start, " s"

        enddo

    enddo

    deallocate(kernel_delta)
    deallocate(self_scalar2)
    deallocate(ksi2)
    deallocate(cosp2)
    deallocate(sinp2)

    allocate(kernel_MA(nm1,na1,nsigmas))
    kernel_MA = 0.0d0
 
    if (verbose) write (*,"(A)", advance="no") "KERNEL"

    !$OMP PARALLEL DO schedule(dynamic) PRIVATE(ni,nj,idx1,s12,idx1_start)
    do a = 1, nm1
        ni = n1(a)
        idx1_start = sum(n1(:a)) - ni
        do i = 1, ni
        
            idx1 = idx1_start + i
            
            do b = 1, nm1
                nj = n1(b)
                do j = 1, nj
 
                    s12 = scalar(x1(a,i,:,:), x1(b,j,:,:), &
                        & nneigh1(a,i), nneigh1(b,j), ksi1(a,i,:), ksi1(b,j,:), &
                        & sinp1(a,i,:,:,:), sinp1(b,j,:,:,:), &
                        & cosp1(a,i,:,:,:), cosp1(b,j,:,:,:), &
                        & t_width, d_width, cut_distance, order, &
                        & pd, ang_norm2, distance_scale, angular_scale, alchemy)
 
                    kernel_MA(b,idx1,:) = kernel_MA(b,idx1,:) & 
                        & + kernel(self_scalar1(a,i), self_scalar1(b,j), s12,&
                         kernel_idx, parameters)
 
                enddo
            enddo
 
        enddo
    enddo
    !$OMP END PARALLEL DO
    
    t_end = omp_get_wtime()
    if (verbose) write (*,"(A,F12.4,A)") "                                  Time = ", t_end - t_start, " s"
    
    deallocate(self_scalar1)
    deallocate(ksi1)
    deallocate(cosp1)
    deallocate(sinp1)
 
    do k = 1, nsigmas
        
        ! kernel_scratch(:,:,k) = kernel_scratch(:,:,k) &
        !    & + matmul(transpose(kernel_MA(:,:,k)),kernel_MA(:,:,k))
 
        ! y(:,k) = y(:,k) + matmul(transpose(kernel_MA(:,:,k)), energies(:))

        if (verbose) write (*,"(A,I12)", advance="no") "     DSYRK()    sigma =", k
        t_start = omp_get_wtime()

        call dsyrk("U", "T", na1, nm1, 1.0d0, kernel_MA(:,:,k), nm1, &
            & 1.0d0, kernel_scratch(:,:,k), na1)

        t_end = omp_get_wtime()
        if (verbose) write (*,"(A,F12.4,A)") "     Time = ", t_end - t_start, " s"
 
        if (verbose) write (*,"(A,I12)", advance="no") "     DGEMV()    sigma =", k
        t_start = omp_get_wtime()
            
        call dgemv("T", nm1, na1, 1.0d0, kernel_ma(:,:,k), nm1, &
                      & energies(:), 1, 1.0d0, y(:,k), 1)

        t_end = omp_get_wtime()
        if (verbose) write (*,"(A,F12.4,A)") "     Time = ", t_end - t_start, " s"
 
    enddo

    deallocate(kernel_ma)

    ! Add regularization
    do k = 1, nsigmas
        do i = 1, na1
            kernel_scratch(i,i,k) = kernel_scratch(i,i,k) + llambda
        enddo
    enddo

    alphas = 0.0d0

    ! Solve alphas
    if (verbose) write (*,"(A)") "CHOLESKY DECOMPOSITION"
    do k = 1, nsigmas

        if (verbose) write (*,"(A,I12)", advance="no") "     DPOTRF()   sigma =", k
        t_start = omp_get_wtime()

        call dpotrf("U", na1, kernel_scratch(:,:,k), na1, info)
        if (info > 0) then
            write (*,*) "QML WARNING: Error in LAPACK Cholesky decomposition DPOTRF()."
            write (*,*) "QML WARNING: The", info, "-th leading order is not positive definite."
        else if (info < 0) then
            write (*,*) "QML WARNING: Error in LAPACK Cholesky decomposition DPOTRF()."
            write (*,*) "QML WARNING: The", -info, "-th argument had an illegal value."
        endif

        t_end = omp_get_wtime()
        if (verbose) write (*,"(A,F12.4,A)") "     Time = ", t_end - t_start, " s"

        if (verbose) write (*,"(A,I12)", advance="no") "     DPOTRS()   sigma =", k
        t_start = omp_get_wtime()

        call dpotrs("U", na1, 1, kernel_scratch(:,:,k), na1, y(:,k), na1, info)
        if (info < 0) then
            write (*,*) "QML WARNING: Error in LAPACK Cholesky solver DPOTRS()."
            write (*,*) "QML WARNING: The", -info, "-th argument had an illegal value."
        endif

        t_end = omp_get_wtime()
        if (verbose) write (*,"(A,F12.4,A)") "     Time = ", t_end - t_start, " s"

        alphas(k,:) = y(:,k)
    enddo

    deallocate(y)
    deallocate(kernel_scratch)

end subroutine fget_force_alphas_fchl


subroutine fget_atomic_local_gradient_kernels_fchl(x1, x2, verbose, n1, n2, nneigh1, nneigh2, &
       & nm1, nm2, na1, naq2, nsigmas, &
       & t_width, d_width, cut_start, cut_distance, order, pd, &
       & distance_scale, angular_scale, alchemy, two_body_power, three_body_power, dx, &
       & kernel_idx, parameters, kernels)

    use ffchl_module, only: scalar, get_angular_norm2, &
        & get_pmax, get_ksi, init_cosp_sinp, get_selfscalar, &
        & get_pmax_displaced, get_ksi_displaced, init_cosp_sinp_displaced, get_selfscalar_displaced

    use ffchl_kernels, only: kernel
    
    use omp_lib, only: omp_get_wtime

    implicit none

    ! fchl descriptors for the training set, format (nm1,3,2,maxatoms,maxatoms,5,maxneighbors)
    double precision, dimension(:,:,:,:), intent(in) :: x1
    double precision, dimension(:,:,:,:,:,:,:), intent(in) :: x2

    ! Whether to be verbose with output
    logical, intent(in) :: verbose

    ! Number of neighbors for each atom in each compound
    integer, dimension(:,:), intent(in) :: nneigh1
    integer, dimension(:,:,:,:,:), intent(in) :: nneigh2

    ! Number of molecules
    integer, intent(in) :: nm1
    integer, intent(in) :: nm2
    
    integer, intent(in) :: na1
    integer, intent(in) :: naq2
    
    ! List of numbers of atoms in each molecule
    integer, dimension(:), intent(in) :: n1
    integer, dimension(:), intent(in) :: n2

    ! Number of kernels
    integer, intent(in) :: nsigmas
    
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

    ! Displacement for numerical differentiation
    double precision, intent(in) :: dx

    ! Kernel ID and corresponding parameters
    integer, intent(in) :: kernel_idx
    double precision, dimension(:,:), intent(in) :: parameters

    ! Resulting alpha vector
    double precision, dimension(nsigmas,na1,naq2), intent(out) :: kernels

    ! Internal counters
    integer :: i2, j1, j2
    integer :: na, nb
    integer :: a, b

    ! Temporary variables necessary for parallelization
    double precision :: s12 

    ! Pre-computed terms in the full distance matrix
    double precision, allocatable, dimension(:,:) :: self_scalar1
    double precision, allocatable, dimension(:,:,:,:,:) :: self_scalar2

    ! Pre-computed two-body weights
    double precision, allocatable, dimension(:,:,:) :: ksi1
    double precision, allocatable, dimension(:,:,:,:,:,:) :: ksi2

    ! Pre-computed terms for the Fourier expansion of the three-body term
    double precision, allocatable, dimension(:,:,:,:,:) :: sinp1
    double precision, allocatable, dimension(:,:,:,:,:) :: cosp1

    ! Pre-computed terms for the Fourier expansion of the three-body term
    double precision, allocatable, dimension(:,:,:,:,:,:,:) :: sinp2
    double precision, allocatable, dimension(:,:,:,:,:,:,:) :: cosp2

    ! Indexes for numerical differentiation 
    integer :: xyz_pm2
    integer :: xyz2, pm2
    integer :: idx1, idx2
    integer :: idx1_start,idx1_end
    integer :: idx2_start,idx2_end

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

    if (verbose) write (*,*) "INIT, dx =", dx

    if (verbose) write (*,*) "CLEARING KERNEL MEM"
    kernels = 0.0d0

    ! Angular normalization constant
    ang_norm2 = get_angular_norm2(t_width)

    ! Max number of neighbors in the representations
    maxneigh1 = maxval(nneigh1)
    maxneigh2 = maxval(nneigh2)

    ! pmax = max nuclear charge
    pmax1 = get_pmax(x1, n1)
    pmax2 = get_pmax_displaced(x2, n2)

    ! Get two-body weight function
    ksi1 = get_ksi(x1, n1, nneigh1, two_body_power, cut_start, cut_distance, verbose)
    ksi2 = get_ksi_displaced(x2, n2, nneigh2, two_body_power, cut_start, cut_distance, verbose)
    
    ! Allocate three-body Fourier terms
    allocate(cosp1(nm1, maxval(n1), pmax1, order, maxneigh1))
    allocate(sinp1(nm1, maxval(n1), pmax1, order, maxneigh1))
    
    ! Initialize and pre-calculate three-body Fourier terms
    call init_cosp_sinp(x1, n1, nneigh1, three_body_power, order, cut_start,  cut_distance, &
        & cosp1, sinp1, verbose)
    
    ! Allocate three-body Fourier terms
    allocate(cosp2(nm2, 3*2, maxval(n2), maxval(n2), pmax2, order, maxneigh2))
    allocate(sinp2(nm2, 3*2, maxval(n2), maxval(n2), pmax2, order, maxneigh2))
    
    ! Initialize and pre-calculate three-body Fourier terms
    call init_cosp_sinp_displaced(x2, n2, nneigh2, three_body_power, order, cut_start, &
        & cut_distance, cosp2, sinp2, verbose)

    ! Pre-calculate self-scalar terms 
    self_scalar1 = get_selfscalar(x1, nm1, n1, nneigh1, ksi1, sinp1, cosp1, t_width, d_width, &
         & cut_distance, order, pd, ang_norm2,distance_scale, angular_scale, alchemy, verbose)

    ! Pre-calculate self-scalar terms 
    self_scalar2 = get_selfscalar_displaced(x2, nm2, n2, nneigh2, ksi2, sinp2, cosp2, t_width, &
    & d_width, cut_distance, order, pd, ang_norm2,distance_scale, angular_scale, alchemy, verbose)

    t_start = omp_get_wtime()
    if (verbose) write (*,"(A)", advance="no") "KERNEL GRADIENT"

    !$OMP PARALLEL DO schedule(dynamic) PRIVATE(na,nb,xyz_pm2,s12),&
    !$OMP& PRIVATE(idx1,idx2,idx1_start,idx1_end,idx2_start,idx2_end)
    do a = 1, nm1
    na = n1(a)
    
    idx1_end = sum(n1(:a))
    idx1_start = idx1_end - na + 1 
    
    do j1 = 1, na
        idx1 = idx1_start - 1 + j1
            
        do b = 1, nm2
        nb = n2(b)

        idx2_end = sum(n2(:b))
        idx2_start = idx2_end - nb + 1 

        do xyz2 = 1, 3
        do pm2 = 1, 2
        xyz_pm2 = 2*xyz2 + pm2 - 2
        do i2 = 1, nb

        idx2 = (idx2_start-1)*3 + (i2-1)*3 + xyz2

        do j2 = 1, nb

            s12 = scalar(x1(a,j1,:,:), x2(b,xyz2,pm2,i2,j2,:,:), &
                & nneigh1(a,j1), nneigh2(b,xyz2,pm2,i2,j2), &
                & ksi1(a,j1,:), ksi2(b,xyz2,pm2,i2,j2,:), &
                & sinp1(a,j1,:,:,:), sinp2(b,xyz_pm2,i2,j2,:,:,:), &
                & cosp1(a,j1,:,:,:), cosp2(b,xyz_pm2,i2,j2,:,:,:), &
                & t_width, d_width, cut_distance, order, &
                & pd, ang_norm2,distance_scale, angular_scale, alchemy)

            if (pm2 == 2) then

                kernels(:,idx1,idx2) = kernels(:,idx1,idx2) & 
                    & + kernel(self_scalar1(a,j1), self_scalar2(b,xyz2,pm2,i2,j2), s12,&
                    & kernel_idx, parameters)
            else
                kernels(:,idx1,idx2) = kernels(:,idx1,idx2) & 
                    & - kernel(self_scalar1(a,j1), self_scalar2(b,xyz2,pm2,i2,j2), s12,&
                    & kernel_idx, parameters)

            end if

        enddo
        enddo
        enddo
        enddo
        enddo
    enddo
    enddo
    !$OMP END PARALLEL do

    kernels = kernels / (2 * dx)
    
    t_end = omp_get_wtime()
    if (verbose) write (*,"(A,F12.4,A)") "                         Time = ", t_end - t_start, " s"

end subroutine fget_atomic_local_gradient_kernels_fchl


subroutine fget_atomic_local_gradient_5point_kernels_fchl(x1, x2, verbose, n1, n2, nneigh1, nneigh2, &
       & nm1, nm2, na1, naq2, nsigmas, &
       & t_width, d_width, cut_start, cut_distance, order, pd, &
       & distance_scale, angular_scale, alchemy, two_body_power, three_body_power, dx, &
       & kernel_idx, parameters, kernels)

    use ffchl_module, only: scalar, get_angular_norm2, &
        & get_pmax, get_ksi, init_cosp_sinp, get_selfscalar, &
        & get_pmax_displaced, get_ksi_displaced, init_cosp_sinp_displaced, get_selfscalar_displaced

    use ffchl_kernels, only: kernel
    
    use omp_lib, only: omp_get_wtime

    implicit none

    ! fchl descriptors for the training set, format (nm1,3,2,maxatoms,maxatoms,5,maxneighbors)
    double precision, dimension(:,:,:,:), intent(in) :: x1
    double precision, dimension(:,:,:,:,:,:,:), intent(in) :: x2

    ! Whether to be verbose with output
    logical, intent(in) :: verbose

    ! Number of neighbors for each atom in each compound
    integer, dimension(:,:), intent(in) :: nneigh1
    integer, dimension(:,:,:,:,:), intent(in) :: nneigh2

    ! Number of molecules
    integer, intent(in) :: nm1
    integer, intent(in) :: nm2
    
    integer, intent(in) :: na1
    integer, intent(in) :: naq2
    
    ! List of numbers of atoms in each molecule
    integer, dimension(:), intent(in) :: n1
    integer, dimension(:), intent(in) :: n2

    ! Number of kernels
    integer, intent(in) :: nsigmas
    
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

    ! Displacement for numerical differentiation
    double precision, intent(in) :: dx

    ! Kernel ID and corresponding parameters
    integer, intent(in) :: kernel_idx
    double precision, dimension(:,:), intent(in) :: parameters

    ! Resulting alpha vector
    double precision, dimension(nsigmas,na1,naq2), intent(out) :: kernels

    ! Internal counters
    integer :: i2, j1, j2
    integer :: na, nb
    integer :: a, b

    ! Temporary variables necessary for parallelization
    double precision :: s12 

    ! Pre-computed terms in the full distance matrix
    double precision, allocatable, dimension(:,:) :: self_scalar1
    double precision, allocatable, dimension(:,:,:,:,:) :: self_scalar2

    ! Pre-computed two-body weights
    double precision, allocatable, dimension(:,:,:) :: ksi1
    double precision, allocatable, dimension(:,:,:,:,:,:) :: ksi2

    ! Pre-computed terms for the Fourier expansion of the three-body term
    double precision, allocatable, dimension(:,:,:,:,:) :: sinp1
    double precision, allocatable, dimension(:,:,:,:,:) :: cosp1

    ! Pre-computed terms for the Fourier expansion of the three-body term
    double precision, allocatable, dimension(:,:,:,:,:,:,:) :: sinp2
    double precision, allocatable, dimension(:,:,:,:,:,:,:) :: cosp2

    ! Indexes for numerical differentiation 
    integer :: xyz_pm2
    integer :: xyz2, pm2
    integer :: idx1, idx2
    integer :: idx1_start,idx1_end
    integer :: idx2_start,idx2_end

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

    ! For numerical differentiation    
    double precision, parameter, dimension(5) :: fact = (/ 1.0d0, -8.0d0, 0.0d0, 8.0d0, -1.0d0/)

    ! fact(1) =   1.0d0
    ! fact(2) =  -8.0d0
    ! fact(3) =   0.0d0
    ! fact(4) =   8.0d0
    ! fact(5) =  -1.0d0

    if (verbose) write (*,*) "INIT, dx =", dx

    if (verbose) write (*,*) "CLEARING KERNEL MEM"
    kernels = 0.0d0

    ! Angular normalization constant
    ang_norm2 = get_angular_norm2(t_width)

    ! Max number of neighbors in the representations
    maxneigh1 = maxval(nneigh1)
    maxneigh2 = maxval(nneigh2)

    ! pmax = max nuclear charge
    pmax1 = get_pmax(x1, n1)
    pmax2 = get_pmax_displaced(x2, n2)

    ! Get two-body weight function
    ksi1 = get_ksi(x1, n1, nneigh1, two_body_power, cut_start, cut_distance, verbose)
    ksi2 = get_ksi_displaced(x2, n2, nneigh2, two_body_power, cut_start, cut_distance, verbose)
    
    ! Allocate three-body Fourier terms
    allocate(cosp1(nm1, maxval(n1), pmax1, order, maxneigh1))
    allocate(sinp1(nm1, maxval(n1), pmax1, order, maxneigh1))
    
    ! Initialize and pre-calculate three-body Fourier terms
    call init_cosp_sinp(x1, n1, nneigh1, three_body_power, order, cut_start,  cut_distance, &
        & cosp1, sinp1, verbose)
    
    ! Allocate three-body Fourier terms
    allocate(cosp2(nm2, 3*5, maxval(n2), maxval(n2), pmax2, order, maxneigh2))
    allocate(sinp2(nm2, 3*5, maxval(n2), maxval(n2), pmax2, order, maxneigh2))
    
    ! Initialize and pre-calculate three-body Fourier terms
    call init_cosp_sinp_displaced(x2, n2, nneigh2, three_body_power, order, cut_start, &
        & cut_distance, cosp2, sinp2, verbose)

    ! Pre-calculate self-scalar terms 
    self_scalar1 = get_selfscalar(x1, nm1, n1, nneigh1, ksi1, sinp1, cosp1, t_width, d_width, &
         & cut_distance, order, pd, ang_norm2,distance_scale, angular_scale, alchemy, verbose)

    ! Pre-calculate self-scalar terms 
    self_scalar2 = get_selfscalar_displaced(x2, nm2, n2, nneigh2, ksi2, sinp2, cosp2, t_width, &
    & d_width, cut_distance, order, pd, ang_norm2,distance_scale, angular_scale, alchemy, verbose)

    t_start = omp_get_wtime()
    if (verbose) write (*,"(A)", advance="no") "KERNEL GRADIENT"

    !$OMP PARALLEL DO schedule(dynamic) PRIVATE(na,nb,xyz_pm2,s12),&
    !$OMP& PRIVATE(idx1,idx2,idx1_start,idx1_end,idx2_start,idx2_end)
    do a = 1, nm1
    na = n1(a)
    
    idx1_end = sum(n1(:a))
    idx1_start = idx1_end - na + 1 
    
    do j1 = 1, na
        idx1 = idx1_start - 1 + j1
            
        do b = 1, nm2
        nb = n2(b)

        idx2_end = sum(n2(:b))
        idx2_start = idx2_end - nb + 1 

        do xyz2 = 1, 3
        do pm2 = 1, 5

        if (pm2 /= 3) then
    
            ! xyz_pm2 = 2*xyz2 + pm2 - 2
            xyz_pm2 = 5*(xyz2 - 1) + pm2

            do i2 = 1, nb
            idx2 = (idx2_start-1)*3 + (i2-1)*3 + xyz2

            do j2 = 1, nb

                s12 = scalar(x1(a,j1,:,:), x2(b,xyz2,pm2,i2,j2,:,:), &
                    & nneigh1(a,j1), nneigh2(b,xyz2,pm2,i2,j2), &
                    & ksi1(a,j1,:), ksi2(b,xyz2,pm2,i2,j2,:), &
                    & sinp1(a,j1,:,:,:), sinp2(b,xyz_pm2,i2,j2,:,:,:), &
                    & cosp1(a,j1,:,:,:), cosp2(b,xyz_pm2,i2,j2,:,:,:), &
                    & t_width, d_width, cut_distance, order, &
                    & pd, ang_norm2,distance_scale, angular_scale, alchemy)

                kernels(:,idx1,idx2) = kernels(:,idx1,idx2) & 
                    & + kernel(self_scalar1(a,j1), self_scalar2(b,xyz2,pm2,i2,j2), s12,&
                    & kernel_idx, parameters) * fact(pm2)

            enddo
            enddo

        endif

        enddo
        enddo
        enddo
    enddo
    enddo
    !$OMP END PARALLEL do

    kernels = kernels / (12 * dx)
    
    t_end = omp_get_wtime()
    if (verbose) write (*,"(A,F12.4,A)") "                         Time = ", t_end - t_start, " s"

end subroutine fget_atomic_local_gradient_5point_kernels_fchl
