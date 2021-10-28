all: clean mesh fixed_mesh isotropic_dwr anisotropic_dwr weighted_hessian weighted_gradient

mesh:
	@for i in 0 1 2 3 4; do python3 meshgen.py $$i; done
	@cd resources && make mesh

fixed_mesh:
	@for i in 0 1 2 3 4; do python3 run_fixed_mesh.py -level $$i -no_exports 1; done

# TODO: drop exports, etc.
convergence:
	@for target in 1000 4000 16000 64000 256000; do for approach in isotropic_dwr anisotropic_dwr weighted_hessian weighted_gradient; do python3 run_adapt.py $$approach -target $$target; done; done

# TODO: drop exports, etc.
isotropic_dwr:
	@for target in 1000 4000 16000 64000 256000; do python3 run_adapt.py isotropic_dwr -target $$target; done

# TODO: drop exports, etc.
anisotropic_dwr:
	@for target in 1000 4000 16000 64000 256000; do python3 run_adapt.py anisotropic_dwr -target $$target; done

# TODO: drop exports, etc.
weighted_hessian:
	@for target in 1000 4000 16000 64000 256000; do python3 run_adapt.py weighted_hessian -target $$target; done

# TODO: drop exports, etc.
weighted_gradient:
	@for target in 1000 4000 16000 64000 256000; do python3 run_adapt.py weighted_gradient -target $$target; done

clean:
	@for i in target1000 target4000 target16000 target64000 target256000; do rm -rf outputs/offset/isotropic_dwr/cg1/target$$i000/*.h5; done
	@for i in target1000 target4000 target16000 target64000 target256000; do rm -rf outputs/offset/anisotropic_dwr/cg1/target$$i000/*.h5; done
	@for i in target1000 target4000 target16000 target64000 target256000; do rm -rf outputs/offset/weighted_hessian/cg1/target$$i000/*.h5; done
	@for i in target1000 target4000 target16000 target64000 target256000; do rm -rf outputs/offset/weighted_gradient/cg1/target$$i000/*.h5; done

plot:
	@python3 plot_convergence.py aligned
	@python3 plot_convergence.py offset
