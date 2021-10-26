all: clean mesh fixed_mesh isotropic_dwr anisotropic_dwr weighted_hessian weighted_gradient

mesh:
	@for i in 0 1 2 3 4; do python3 meshgen.py $$i; done
	@cd resources && make mesh

fixed_mesh:
	@for i in 0 1 2 3 4; do for config in aligned offset; do python3 run_fixed_mesh.py $$config -level $$i -no_exports 1; done; done

# TODO: drop exports, etc.
isotropic_dwr:
	@for target in 1000 4000 16000 64000 512000; do for config in aligned offset; do python3 run_adapt.py $$config isotropic_dwr -target $$target; done; done

# TODO: drop exports, etc.
anisotropic_dwr:
	@for target in 1000 4000 16000 64000 512000; do for config in aligned offset; do python3 run_adapt.py $$config anisotropic_dwr -target $$target; done; done

# TODO: drop exports, etc.
weighted_hessian:
	@for target in 1000 4000 16000 64000 512000; do for config in aligned offset; do python3 run_adapt.py $$config weighted_hessian -target $$target; done; done

# TODO: drop exports, etc.
weighted_gradient:
	@for target in 1000 4000 16000 64000 512000; do for config in aligned offset; do python3 run_adapt.py $$config weighted_gradient -target $$target; done; done

clean:
	@rm -rf outputs/aligned/isotropic_dwr/cg1/level0/*.h5
	@rm -rf outputs/offset/isotropic_dwr/cg1/level0/*.h5
	@rm -rf outputs/aligned/anisotropic_dwr/cg1/level0/*.h5
	@rm -rf outputs/offset/anisotropic_dwr/cg1/level0/*.h5
	@rm -rf outputs/aligned/weighted_hessian/cg1/level0/*.h5
	@rm -rf outputs/offset/weighted_hessian/cg1/level0/*.h5
	@rm -rf outputs/aligned/weighted_gradient/cg1/level0/*.h5
	@rm -rf outputs/offset/weighted_gradient/cg1/level0/*.h5

plot:
	@python3 plot_convergence.py aligned
	@python3 plot_convergence.py offset
