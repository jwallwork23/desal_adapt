all: convergence

snapshot: snapshot_hessian snapshot_isotropic_dwr snapshot_anisotropic_dwr snapshot_weighted_hessian snapshot_weighted_gradient

snapshot_hessian:
	@python3 run_adapt.py aligned hessian

snapshot_isotropic_dwr:
	@python3 run_adapt.py aligned isotropic_dwr
	@python3 run_adapt.py offset isotropic_dwr

snapshot_anisotropic_dwr:
	@python3 run_adapt.py aligned anisotropic_dwr -target 8000
	@python3 run_adapt.py offset anisotropic_dwr -target 8000

snapshot_weighted_hessian:
	@python3 run_adapt.py aligned weighted_hessian -target 3000
	@python3 run_adapt.py offset weighted_hessian -target 3000

snapshot_weighted_gradient:
	@python3 run_adapt.py aligned weighted_gradient
	@python3 run_adapt.py offset weighted_gradient

convergence: uniform hessian isotropic_dwr anisotropic_dwr weighted_hessian weighted_gradient plot

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

time: time_uniform time_hessian time_isotropic_dwr time_anisotropic_dwr time_weighted_hessian time_weighted_gradient plot

time_uniform:
	@python3 run_uniform_convergence.py aligned -num_repetitions 5
	@python3 run_uniform_convergence.py offset -num_repetitions 5

time_hessian:
	@python3 run_adaptation_loop.py aligned hessian -num_repetitions 5
	@python3 run_adaptation_loop.py offset hessian -num_repetitions 5

time_isotropic_dwr:
	@python3 run_adaptation_loop.py aligned isotropic_dwr -num_repetitions 5
	@python3 run_adaptation_loop.py offset isotropic_dwr -num_repetitions 5

time_anisotropic_dwr:
	@python3 run_adaptation_loop.py aligned anisotropic_dwr -num_repetitions 5
	@python3 run_adaptation_loop.py offset anisotropic_dwr -num_repetitions 5

time_weighted_hessian:
	@python3 run_adaptation_loop.py aligned weighted_hessian -num_repetitions 5
	@python3 run_adaptation_loop.py offset weighted_hessian -num_repetitions 5

time_weighted_gradient:
	@python3 run_adaptation_loop.py aligned weighted_gradient -num_repetitions 5
	@python3 run_adaptation_loop.py offset weighted_gradient -num_repetitions 5

plot:
	@python3 plot_convergence.py aligned
	@python3 plot_convergence.py offset

profile: profile_fixed profile_hessian profile_isotropic_dwr profile_anisotropic_dwr profile_weighted_hessian profile_weighted_gradient

profile_fixed:
	@for config in aligned offset; do python3 run_fixed_mesh.py $$config -profile 1 -log_view :outputs/$$config/fixed_mesh/cg1/logview.txt:ascii_flamegraph; done
	@for config in aligned offset; do cd outputs/$$config/fixed_mesh/cg1 && flamegraph.pl logview.txt > flamegraph_$$config.svg && cd ../../../..; done

profile_hessian:
	@for config in aligned offset; do python3 run_adapt.py $$config hessian -profile 1 -target 8000 -log_view :outputs/$$config/hessian/cg1/logview.txt:ascii_flamegraph; done
	@for config in aligned offset; do cd outputs/$$config/hessian/cg1 && flamegraph.pl logview.txt > flamegraph_$$config.svg && cd ../../../..; done

profile_isotropic_dwr:
	@for config in aligned offset; do python3 run_adapt.py $$config isotropic_dwr -profile 1 -target 8000 -log_view :outputs/$$config/isotropic_dwr/cg1/logview.txt:ascii_flamegraph; done
	@for config in aligned offset; do cd outputs/$$config/isotropic_dwr/cg1 && flamegraph.pl logview.txt > flamegraph_$$config.svg && cd ../../../..; done

profile_anisotropic_dwr:
	@for config in aligned offset; do python3 run_adapt.py $$config anisotropic_dwr -profile 1 -target 16000 -log_view :outputs/$$config/anisotropic_dwr/cg1/logview.txt:ascii_flamegraph; done
	@for config in aligned offset; do cd outputs/$$config/anisotropic_dwr/cg1 && flamegraph.pl logview.txt > flamegraph_$$config.svg && cd ../../../..; done

profile_weighted_hessian:
	@for config in aligned offset; do python3 run_adapt.py $$config weighted_hessian -profile 1 -target 6000 -log_view :outputs/$$config/weighted_hessian/cg1/logview.txt:ascii_flamegraph; done
	@for config in aligned offset; do cd outputs/$$config/weighted_hessian/cg1 && flamegraph.pl logview.txt > flamegraph_$$config.svg && cd ../../../..; done

profile_weighted_gradient:
	@for config in aligned offset; do python3 run_adapt.py $$config weighted_gradient -profile 1 -target 8000 -log_view :outputs/$$config/weighted_gradient/cg1/logview.txt:ascii_flamegraph; done
	@for config in aligned offset; do cd outputs/$$config/weighted_gradient/cg1 && flamegraph.pl logview.txt > flamegraph_$$config.svg && cd ../../../..; done
