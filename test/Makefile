all: fchl-acsf gradient-kernel

fchl-acsf: ffchl_acsf.f90
	/home/andersx/.local/bin/f2py -c ffchl_acsf.f90 -m ffchl_acsf --opt='-O3 -m64 -march=native' --f90flags='-ffree-line-length-512'

gradient-kernel: fgradient_kernels.f90
	/home/andersx/.local/bin/f2py -c fgradient_kernels.f90 -m fgradient_kernels --opt='-O3 -m64 -march=native' --f90flags='-ffree-line-length-512'

clean:
	rm -f *.so
	rm -f *.pyc
	rm -f *.mod
