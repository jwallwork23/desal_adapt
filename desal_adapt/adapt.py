from desal_adapt import *
from desal_adapt.error_estimation import ErrorEstimator
from desal_adapt.utility import ramp_complexity
import pyadjoint
from thetis import print_output, get_functionspace
from time import perf_counter


__all__ = ["GoalOrientedDesalinationPlant"]


class GoalOrientedDesalinationPlant(GoalOrientedMeshSeq):
    """
    Class to facilitate goal-oriented metric-based mesh
    adaptive simulations of desalination outfall modelling problems.
    """
    def __init__(self, options, root_dir, num_subintervals, qoi='inlet_salinity_difference'):
        """
        :arg options: :class:`PlantOptions` encapsulating the problem
        :arg root_dir: directory where metrics should be stored
        """
        self.options = options
        self.root_dir = root_dir
        self.integrated_quantity = qoi

        self.test_functions = [None]*num_subintervals
        self.cell_sizes = [None]*num_subintervals

        # Partition time interval
        dt = options.timestep
        dt_per_export = [int(options.simulation_export_time/dt)]*num_subintervals
        time_partition = TimePartition(
            options.simulation_end_time, num_subintervals,
            dt, ['tracer_2d'], timesteps_per_export=dt_per_export,
        )

        # Set initial meshes to default
        initial_meshes = [
            Mesh(options.mesh2d)
            for subinterval in time_partition.subintervals
        ]

        # Create GoalOrientedMeshSeq
        super(GoalOrientedDesalinationPlant, self).__init__(
            time_partition, initial_meshes,
            None, None, None, None, qoi_type='time_integrated',
        )

    def get_function_spaces(self, mesh):
        if self.options.tracer_element_family == 'dg':
            return {'tracer_2d': get_functionspace(mesh, "DG", 1, name="Q_2d")}
        elif self.options.tracer_element_family == 'cg':
            return {'tracer_2d': get_functionspace(mesh, "CG", 1, name="Q_2d")}
        else:
            raise NotImplementedError

    def get_solver(self):
        options = self.options

        def solver(i, ic, **model_options):
            """
            Solve forward over time window (`t_start`, `t_end`).
            """
            t_start, t_end = self.time_partition.subintervals[i]
            mesh = ic.tracer_2d.function_space().mesh()
            options.rebuild_mesh_dependent_components(mesh)
            options.simulation_end_time = t_end
            i_export = int(np.round(t_start/options.simulation_export_time))
            options.tracer_old = ic.tracer_2d
            compute_salinity = model_options.pop('compute_salinity', False)
            model_options.setdefault('no_exports', True)
            options.update(model_options)
            if not options.no_exports:
                options.fields_to_export = ['tracer_2d']

            # Create a new solver object
            solver_obj = PlantSolver2d(options, mesh=mesh,
                                       i_export=i_export, t_start=t_start,
                                       compute_salinity=compute_salinity)

            # Setup QoI
            qoi = self.get_qoi(i)
            uf = options.get_update_forcings(solver_obj)

            def update_forcings(t):
                uf(t)
                self.J += qoi({'tracer_2d': solver_obj.fields.tracer_2d}, t)

            # Solve forward on current subinterval
            export_func = options.get_export_func(solver_obj)
            solver_obj.iterate(update_forcings=update_forcings, export_func=export_func)

            # Stash SUPG components
            self.test_functions[i] = options.test_function
            self.cell_sizes[i] = options.cell_size

            return AttrDict({'tracer_2d': solver_obj.fields.tracer_2d})

        return solver

    def get_initial_condition(self):
        """
        Background salinity value.
        """
        tracer_2d = Function(self.function_spaces.tracer_2d[0])
        tracer_2d.assign(self.options.background_salinity)
        return {'tracer_2d': tracer_2d}

    def get_qoi(self, i):
        """
        Currently supported QoIs:

        * 'inlet_salinity'
            - salinity at the inlet of the desalination plant
        * 'inlet_salinity_difference'
            - difference between the salinity at the inlet of the
              desalination plant and the background value
        """
        kernel = self.options.qoi_kernel
        if self.integrated_quantity == 'inlet_salinity':

            def qoi(sol, t):
                tracer_2d = sol['tracer_2d']
                j = assemble(kernel*tracer_2d*dx)
                if pyadjoint.tape.annotate_tape():
                    j.block_variable.adj_value = 1.0
                return j

        elif self.integrated_quantity == 'inlet_salinity_difference':
            bg = self.options.background_salinity

            def qoi(sol, t):
                tracer_2d = sol['tracer_2d']
                j = assemble(kernel*(tracer_2d - bg)*dx)
                if pyadjoint.tape.annotate_tape():
                    j.block_variable.adj_value = 1.0
                return j

        else:
            raise ValueError(f"Integrated quantity {self.integrated_quantity} not recognised")

        return qoi

    def fixed_point_iteration(self, **parsed_args):
        """
        Apply a goal-oriented metric-based mesh adaptation
        fixed point iteration loop for a desalination outfall
        modelling problem.
        """
        cpu_timestamp = perf_counter()
        parsed_args = AttrDict(parsed_args)
        options = self.options
        expected = {'miniter', 'maxiter', 'load_index', 'qoi_rtol', 'element_rtol',
                    'error_indicator', 'approach', 'h_min', 'h_max', 'a_max', 'profile',
                    'target', 'base_complexity', 'flux_form', 'norm_order', 'convergence_rate'}
        if not expected.issubset(set(parsed_args.keys())):
            raise ValueError("Missing required arguments"
                             f" {expected.difference(set(parsed_args.keys()))}")
        output_dir = options.output_directory
        end_time = options.simulation_end_time
        dt = options.timestep
        approach = parsed_args.approach
        target = end_time/dt*parsed_args.target  # Convert to space-time complexity
        base = end_time/dt*parsed_args.base_complexity
        num_subintervals = self.num_subintervals
        timesteps = [dt]*num_subintervals
        optimise = parsed_args.profile

        # Enter fixed point iteration
        miniter = parsed_args.miniter
        maxiter = parsed_args.maxiter
        if miniter > maxiter:
            miniter = maxiter
        qoi_rtol = parsed_args.qoi_rtol
        element_rtol = parsed_args.element_rtol
        converged = False
        converged_reason = None
        num_cells_old = None
        J_old = None
        load_index = parsed_args.load_index
        fp_iteration = load_index
        while fp_iteration <= maxiter:
            outfiles = AttrDict({})
            if fp_iteration < miniter:
                converged = False
            elif fp_iteration == maxiter:
                converged = True
                if converged_reason is None:
                    converged_reason = 'maximum number of iterations reached'

            # Ramp up the target complexity
            target_ramp = ramp_complexity(base, target, fp_iteration)

            # Load meshes, if requested
            if load_index > 0 and fp_iteration == load_index and not optimise:
                for i in range(num_subintervals):
                    mesh_fname = os.path.join(output_dir, f"mesh_fp{fp_iteration}_{i}")
                    if os.path.exists(mesh_fname + '.h5'):
                        print_output(f"\n--- Loading plex data for mesh {i+1}\n{mesh_fname}")
                    else:
                        raise IOError(f"Cannot load mesh file {mesh_fname}.")
                    plex = PETSc.DMPlex().create()
                    plex.createFromFile(mesh_fname + '.h5')
                    self.meshes[i] = Mesh(plex)

            # Create metric Functions
            metrics = [
                Function(TensorFunctionSpace(mesh, "CG", 1), name="Metric")
                for mesh in self.meshes
            ]

            # Load metric data, if available
            loaded = False
            if fp_iteration == load_index and not optimise:
                for i, metric in enumerate(metrics):
                    if load_index == 0:
                        metric_fname = os.path.join(self.root_dir, f'metric{i}')
                    else:
                        metric_fname = os.path.join(output_dir, f'metric{i}_fp{fp_iteration}')
                    if os.path.exists(metric_fname + '.h5'):
                        print_output(f"\n--- Loading metric data on mesh {i+1}\n{metric_fname}")
                        try:
                            with DumbCheckpoint(metric_fname, mode=FILE_READ) as chk:
                                chk.load(metric, name="Metric")
                            loaded = True
                        except Exception:
                            print_output(f"Cannot load metric data on mesh {i+1}")
                            loaded = False
                            break
                    else:
                        assert not loaded, "Only partial metric data available"
                        break

            # Otherwise, solve forward and adjoint
            if not loaded:

                # Solve forward and adjoint on each subinterval
                if converged:
                    with pyadjoint.stop_annotating():
                        print_output("\n--- Final forward run\n")
                        self.get_checkpoints(
                            solver_kwargs=dict(no_exports=False, compute_salinity=True),
                            run_final_subinterval=True,
                        )
                else:
                    print_output(f"\n--- Forward-adjoint sweep {fp_iteration}\n")
                    solutions = self.solve_adjoint()

                # Check for QoI convergence
                if J_old is not None:
                    if abs(self.J - J_old) < qoi_rtol*J_old and fp_iteration >= miniter:
                        converged = True
                        converged_reason = 'converged quantity of interest'
                        with pyadjoint.stop_annotating():
                            print_output("\n--- Final forward run\n")
                            self.get_checkpoints(
                                solver_kwargs=dict(no_exports=False, compute_salinity=True),
                                run_final_subinterval=True,
                            )
                J_old = self.J

                # Escape if converged
                if converged:
                    print_output(f"Termination due to {converged_reason} after {fp_iteration+1}"
                                 + f" iterations\nEnergy output: {self.J/3.6e+09} MWh")
                    break

                # Create vtu output files
                outfiles.forward = None if optimise else File(os.path.join(output_dir, 'Forward2d.pvd'))
                outfiles.forward_old = None if optimise else File(os.path.join(output_dir, 'ForwardOld2d.pvd'))
                outfiles.adjoint_next = None if optimise else File(os.path.join(output_dir, 'AdjointNext2d.pvd'))
                outfiles.adjoint = None if optimise else File(os.path.join(output_dir, 'Adjoint2d.pvd'))

                # Construct metric
                with pyadjoint.stop_annotating():
                    print_output(f"\n--- Error estimation {fp_iteration}\n")
                    for i, mesh in enumerate(self.meshes):
                        print_output(f"Constructing {approach} metric {i+1}...")
                        options.rebuild_mesh_dependent_components(mesh)
                        options.get_bnd_conditions(self.function_spaces.tracer_2d[i])

                        # Create error estimator
                        options.test_function = self.test_functions[i]
                        options.Q = self.function_spaces['tracer_2d'][i]
                        options.cell_size = self.cell_sizes[i]
                        ee = ErrorEstimator(options,
                                            mesh=mesh,
                                            error_estimator=parsed_args.error_indicator,
                                            metric=approach,
                                            recovery_method='Clement')
                        metrics[i].assign(0.0)
                        uv = Function(ee.P1_vec)

                        # Loop over all exported timesteps
                        N = len(solutions.tracer_2d.adjoint[i])
                        for j in range(N):
                            if i < num_subintervals-1 and j == N-1:
                                continue

                            # Plot fields
                            args = []
                            for f in outfiles:
                                if f in ('forward', 'adjoint_next'):
                                    t = i*end_time/num_subintervals + dt*j
                                else:
                                    t = i*end_time/num_subintervals + dt*(j + 1)
                                self.options.tc.assign(t)
                                uv.interpolate(self.options.forced_velocity)
                                args.append(uv.copy(deepcopy=True))
                                args.append(solutions.tracer_2d[f][i][j])
                                args[-1].rename("Adjoint salinity" if 'adjoint' in f else "Salinity")
                                if not optimise:
                                    outfiles[f].write(*args[-2:])

                            # Evaluate error indicator
                            metric_step = ee.metric(*args, target_complexity=parsed_args.base_complexity, **parsed_args)

                            # Apply trapezium rule
                            metric_step *= 0.5*dt if j in (0, N-1) else dt
                            metrics[i] += metric_step

                    # Store metrics
                    if not optimise:
                        for i, metric in enumerate(metrics):
                            print_output(f"Storing {approach} metric data on mesh {i+1}...")
                            metric_fname = os.path.join(output_dir, f'metric{i}_fp{fp_iteration}')
                            with DumbCheckpoint(metric_fname, mode=FILE_CREATE) as chk:
                                chk.store(metric, name="Metric")
                            if fp_iteration == 0:
                                metric_fname = os.path.join(self.root_dir, f'metric{i}')
                                with DumbCheckpoint(metric_fname, mode=FILE_CREATE) as chk:
                                    chk.store(metric, name="Metric")

            # Process metrics
            print_output(f"\n--- Metric processing {fp_iteration}\n")
            metrics = space_time_normalise(metrics, end_time, timesteps, target_ramp, parsed_args.norm_order)

            # Enforce element constraints
            metrics = enforce_element_constraints(
                metrics, parsed_args.h_min, parsed_args.h_max, parsed_args.a_max
            )

            # Plot metrics
            if not optimise:
                outfiles.metric = File(os.path.join(output_dir, 'Metric2d.pvd'))
                for metric in metrics:
                    metric.rename("Metric")
                    outfiles.metric.write(metric)

            # Adapt meshes
            print_output(f"\n--- Mesh adaptation {fp_iteration}\n")
            if not optimise:
                outfiles.mesh = File(os.path.join(output_dir, 'Mesh2d.pvd'))
            for i, metric in enumerate(metrics):
                self.meshes[i] = Mesh(adapt(self.meshes[i], metric))
                if not optimise:
                    outfiles.mesh.write(self.meshes[i].coordinates)
            num_cells = [mesh.num_cells() for mesh in self.meshes]

            # Check for convergence of element count
            elements_converged = False
            if num_cells_old is not None:
                elements_converged = True
                for nc, _nc in zip(num_cells, num_cells_old):
                    if abs(nc - _nc) > element_rtol*_nc:
                        elements_converged = False
            num_cells_old = num_cells
            if elements_converged:
                print_output(f"Mesh element count converged to rtol {element_rtol}")
                converged = True
                converged_reason = 'converged element counts'

            # Save mesh data to disk
            if COMM_WORLD.size == 1 and not opimise:
                for i, mesh in enumerate(self.meshes):
                    mesh_fname = os.path.join(output_dir, f"mesh_fp{fp_iteration+1}_{i}.h5")
                    viewer = PETSc.Viewer().createHDF5(mesh_fname, 'w')
                    viewer(mesh.topology_dm)

            # Increment
            fp_iteration += 1

        # Log convergence reason
        cpu_time = perf_counter() - cpu_timestamp
        return '\n'.join([
            f'Converged in {fp_iteration+1} iterations due to {converged_reason}',
            f'QoI = {self.J:.8e}',
            f'CPU time = {cpu_time:.1f} seconds', ''
        ])
