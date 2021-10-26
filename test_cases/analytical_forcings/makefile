all: mesh uniform

mesh:
	@for i in 0 1 2 3 4; do python3 meshgen.py $$i; done
	@cd resources && make mesh

uniform:
	@for i in 0 1 2 3 4; do for config in aligned offset; do python3 run_fixed_mesh.py $$config -level $$i -no_exports 1; done; done

plot:
	@python3 plot_convergence.py aligned
	@python3 plot_convergence.py offset
