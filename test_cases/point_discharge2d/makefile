all: uniform hessian isotropic_dwr anisotropic_dwr weighted_hessian weighted_gradient plot

uniform:
	@python3 run_uniform_convergence.py aligned
	@python3 run_uniform_convergence.py offset

hessian:
	@python3 run_adaptation_loop.py aligned -approach hessian
	@python3 run_adaptation_loop.py offset -approach hessian

isotropic_dwr:
	@python3 run_adaptation_loop.py aligned -approach isotropic_dwr
	@python3 run_adaptation_loop.py offset -approach isotropic_dwr

anisotropic_dwr:
	@python3 run_adaptation_loop.py aligned -approach anisotropic_dwr
	@python3 run_adaptation_loop.py offset -approach anisotropic_dwr

weighted_hessian:
	@python3 run_adaptation_loop.py aligned -approach weighted_hessian
	@python3 run_adaptation_loop.py offset -approach weighted_hessian

weighted_gradient:
	@python3 run_adaptation_loop.py aligned -approach weighted_gradient
	@python3 run_adaptation_loop.py offset -approach weighted_gradient

plot:
	@python3 plot_convergence.py aligned
	@python3 plot_convergence.py offset
