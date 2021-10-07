all: uniform hessian plot

uniform:
	@python3 run_uniform_convergence.py aligned
	@python3 run_uniform_convergence.py offset

hessian:
	@python3 run_adaptation_loop.py aligned -approach hessian
	@python3 run_adaptation_loop.py offset -approach hessian

plot:
	@python3 plot_convergence.py aligned
	@python3 plot_convergence.py offset
