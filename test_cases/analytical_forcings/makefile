all: clean mesh fixed_mesh isotropic_dwr anisotropic_dwr weighted_hessian weighted_gradient

mesh:
	@for i in 0 1 2 3 4; do python3 meshgen.py $$i; done
	@cd resources && make mesh

fixed_mesh:
	@for i in 0 1 2 3 4; do python3 run_fixed_mesh.py -level $$i -no_exports 1; done

convergence: clean_h5
	@for target in 1000 2000 4000 8000 16000 32000; do for approach in isotropic_dwr anisotropic_dwr weighted_hessian weighted_gradient; do python3 run_adapt.py $$approach -target $$target -profile 1; done; done

isotropic_dwr:
	@for target in 1000 2000 4000 8000 16000 32000; do python3 run_adapt.py isotropic_dwr -target $$target -profile 1; done

anisotropic_dwr:
	@for target in 1000 2000 4000 8000 16000 32000; do python3 run_adapt.py anisotropic_dwr -target $$target -profile 1; done

weighted_hessian:
	@for target in 1000 2000 4000 8000 16000 32000; do python3 run_adapt.py weighted_hessian -target $$target -profile 1; done

weighted_gradient:
	@for target in 1000 2000 4000 8000 16000 32000; do python3 run_adapt.py weighted_gradient -target $$target -profile 1; done

clean: clean_vtu clean_h5

clean_lagged_vtu:
	@for i in target1000 target2000 target4000 target8000 target16000 target32000; do rm -rf outputs/offset/isotropic_dwr/cg1/$$i/AdjointNext2d*; done
	@for i in target1000 target2000 target4000 target8000 target16000 target32000; do rm -rf outputs/offset/isotropic_dwr/cg1/$$i/ForwardOld2d*; done
	@for i in target1000 target2000 target4000 target8000 target16000 target32000; do rm -rf outputs/offset/anisotropic_dwr/cg1/$$i/AdjointNext2d*; done
	@for i in target1000 target2000 target4000 target8000 target16000 target32000; do rm -rf outputs/offset/anisotropic_dwr/cg1/$$i/ForwardOld2d*; done
	@for i in target1000 target2000 target4000 target8000 target16000 target32000; do rm -rf outputs/offset/weighted_hessian/cg1/$$i/AdjointNext2d*; done
	@for i in target1000 target2000 target4000 target8000 target16000 target32000; do rm -rf outputs/offset/weighted_hessian/cg1/$$i/ForwardOld2d*; done
	@for i in target1000 target2000 target4000 target8000 target16000 target32000; do rm -rf outputs/offset/weighted_gradient/cg1/$$i/AdjointNext2d*; done
	@for i in target1000 target2000 target4000 target8000 target16000 target32000; do rm -rf outputs/offset/weighted_gradient/cg1/$$i/ForwardOld2d*; done

clean_vtu: clean_lagged_vtu
	@for i in target1000 target2000 target4000 target8000 target16000 target32000; do rm -rf outputs/offset/isotropic_dwr/cg1/$$i/Adjoint2d*; done
	@for i in target1000 target2000 target4000 target8000 target16000 target32000; do rm -rf outputs/offset/isotropic_dwr/cg1/$$i/Forward2d*; done
	@for i in target1000 target2000 target4000 target8000 target16000 target32000; do rm -rf outputs/offset/anisotropic_dwr/cg1/$$i/Adjoint2d*; done
	@for i in target1000 target2000 target4000 target8000 target16000 target32000; do rm -rf outputs/offset/anisotropic_dwr/cg1/$$i/Forward2d*; done
	@for i in target1000 target2000 target4000 target8000 target16000 target32000; do rm -rf outputs/offset/weighted_hessian/cg1/$$i/Adjoint2d*; done
	@for i in target1000 target2000 target4000 target8000 target16000 target32000; do rm -rf outputs/offset/weighted_hessian/cg1/$$i/Forward2d*; done
	@for i in target1000 target2000 target4000 target8000 target16000 target32000; do rm -rf outputs/offset/weighted_gradient/cg1/$$i/Adjoint2d*; done
	@for i in target1000 target2000 target4000 target8000 target16000 target32000; do rm -rf outputs/offset/weighted_gradient/cg1/$$i/Forward2d*; done

clean_h5:
	@for i in target1000 target2000 target4000 target8000 target16000 target32000; do rm -rf outputs/offset/isotropic_dwr/cg1/$$i/*.h5; done
	@for i in target1000 target2000 target4000 target8000 target16000 target32000; do rm -rf outputs/offset/anisotropic_dwr/cg1/$$i/*.h5; done
	@for i in target1000 target2000 target4000 target8000 target16000 target32000; do rm -rf outputs/offset/weighted_hessian/cg1/$$i/*.h5; done
	@for i in target1000 target2000 target4000 target8000 target16000 target32000; do rm -rf outputs/offset/weighted_gradient/cg1/$$i/*.h5; done

plot:
	@python3 plot_convergence.py -configuration offset
