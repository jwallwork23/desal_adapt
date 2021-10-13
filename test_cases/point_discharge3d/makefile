all: uniform hessian isotropic_dwr anisotropic_dwr weighted_hessian weighted_gradient plot

snapshot: snapshot_hessian snapshot_isotropic_dwr snapshot_anisotropic_dwr snapshot_weighted_hessian snapshot_weighted_gradient

snapshot_hessian:
	@python3 run_adapt.py aligned hessian

snapshot_isotropic_dwr:
	@python3 run_adapt.py aligned isotropic_dwr
	@python3 run_adapt.py offset isotropic_dwr

snapshot_anisotropic_dwr:
	@python3 run_adapt.py aligned anisotropic_dwr
	@python3 run_adapt.py offset anisotropic_dwr

snapshot_weighted_hessian:
	@python3 run_adapt.py aligned weighted_hessian
	@python3 run_adapt.py offset weighted_hessian

snapshot_weighted_gradient:
	@python3 run_adapt.py aligned weighted_gradient
	@python3 run_adapt.py offset weighted_gradient

uniform:
	@python3 run_uniform_convergence.py aligned -num_repetitions 5
	@python3 run_uniform_convergence.py offset -num_repetitions 5

hessian:
	@python3 run_adaptation_loop.py aligned hessian -num_repetitions 5
	@python3 run_adaptation_loop.py offset hessian -num_repetitions 5

isotropic_dwr:
	@python3 run_adaptation_loop.py aligned isotropic_dwr -num_repetitions 5
	@python3 run_adaptation_loop.py offset isotropic_dwr -num_repetitions 5

anisotropic_dwr:
	@python3 run_adaptation_loop.py aligned anisotropic_dwr -num_repetitions 5
	@python3 run_adaptation_loop.py offset anisotropic_dwr -num_repetitions 5

weighted_hessian:
	@python3 run_adaptation_loop.py aligned weighted_hessian -num_repetitions 5
	@python3 run_adaptation_loop.py offset weighted_hessian -num_repetitions 5

weighted_gradient:
	@python3 run_adaptation_loop.py aligned weighted_gradient -num_repetitions 5
	@python3 run_adaptation_loop.py offset weighted_gradient -num_repetitions 5

plot:
	@python3 plot_convergence.py aligned
	@python3 plot_convergence.py offset
