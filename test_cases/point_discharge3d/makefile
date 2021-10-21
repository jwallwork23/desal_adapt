all: uniform hessian isotropic_dwr anisotropic_dwr weighted_hessian weighted_gradient plot

calibrate:
	@for i in 0 1 2 3; do python3 calibrate.py $$i; done
	@for i in 0 1 2 3; do python3 plot_calibration.py $$i; done

analytical:
	@python3 options.py aligned
	@python3 options.py offset

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

plot:
	@python3 plot_convergence.py aligned
	@python3 plot_convergence.py offset

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

profile: profile_fixed profile_hessian profile_isotropic_dwr profile_anisotropic_dwr profile_weighted_hessian profile_weighted_gradient

profile_fixed:
	@python3 run_fixed_mesh.py aligned -profile 1 -log_view :outputs/aligned/fixed_mesh/cg1/logview.txt:ascii_flamegraph
	@cd outputs/aligned/fixed_mesh/cg1 && flamegraph.pl --title "Fixed mesh 3D" logview.txt > flamegraph_fixed_mesh.svg && rm logview.txt

profile_hessian:
	@python3 run_adapt.py aligned hessian -profile 1 -target 10000 -log_view :outputs/aligned/hessian/cg1/logview.txt:ascii_flamegraph
	@cd outputs/aligned/hessian/cg1 && flamegraph.pl --title "Hessian-based 3D" logview.txt > flamegraph_aligned.svg && rm logview.txt

profile_isotropic_dwr:
	@python3 run_adapt.py aligned isotropic_dwr -profile 1 -target 10000 -log_view :outputs/aligned/isotropic_dwr/cg1/logview.txt:ascii_flamegraph
	@cd outputs/aligned/isotropic_dwr/cg1 && flamegraph.pl --title "Isotropic DWR 3D" logview.txt > flamegraph_aligned.svg && rm logview.txt

profile_anisotropic_dwr:
	@python3 run_adapt.py aligned anisotropic_dwr -profile 1 -target 20000 -log_view :outputs/aligned/anisotropic_dwr/cg1/logview.txt:ascii_flamegraph
	@cd outputs/aligned/anisotropic_dwr/cg1 && flamegraph.pl --title "Anisotropic DWR 3D" logview.txt > flamegraph_aligned.svg && rm logview.txt

profile_weighted_hessian:
	@python3 run_adapt.py aligned weighted_hessian -profile 1 -target 7500 -log_view :outputs/aligned/weighted_hessian/cg1/logview.txt:ascii_flamegraph
	@cd outputs/aligned/weighted_hessian/cg1 && flamegraph.pl --title "Weighted Hessian 3D" logview.txt > flamegraph_aligned.svg && rm logview.txt

profile_weighted_gradient:
	@python3 run_adapt.py aligned weighted_gradient -profile 1 -target 10000 -log_view :outputs/aligned/weighted_gradient/cg1/logview.txt:ascii_flamegraph
	@cd outputs/aligned/weighted_gradient/cg1 && flamegraph.pl --title "Weighted gradient 3D" logview.txt > flamegraph_aligned.svg && rm logview.txt
