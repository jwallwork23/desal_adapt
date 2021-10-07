all: uniform hessian isotropic_dwr anisotropic_dwr weighted_hessian weighted_gradient plot

snapshot:
	@python3 run_adapt.py aligned hessian
	@python3 run_adapt.py aligned isotropic_dwr
	@python3 run_adapt.py offset isotropic_dwr
	@python3 run_adapt.py aligned anisotropic_dwr
	@python3 run_adapt.py offset anisotropic_dwr
	@python3 run_adapt.py aligned weighted_hessian
	@python3 run_adapt.py offset weighted_hessian
	@python3 run_adapt.py aligned weighted_gradient
	@python3 run_adapt.py offset weighted_gradient

uniform:
	@python3 run_uniform_convergence.py aligned
	@python3 run_uniform_convergence.py offset

hessian:
	@python3 run_adaptation_loop.py aligned hessian
	@python3 run_adaptation_loop.py offset hessian

isotropic_dwr:
	@python3 run_adaptation_loop.py aligned isotropic_dwr
	@python3 run_adaptation_loop.py offset isotropic_dwr

anisotropic_dwr:
	@python3 run_adaptation_loop.py aligned anisotropic_dwr
	@python3 run_adaptation_loop.py offset anisotropic_dwr

weighted_hessian:
	@python3 run_adaptation_loop.py aligned weighted_hessian
	@python3 run_adaptation_loop.py offset weighted_hessian

weighted_gradient:
	@python3 run_adaptation_loop.py aligned weighted_gradient
	@python3 run_adaptation_loop.py offset weighted_gradient

plot:
	@python3 plot_convergence.py aligned
	@python3 plot_convergence.py offset
