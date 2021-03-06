from desal_adapt import *
from desal_adapt.error_estimation import ErrorEstimator
from desal_adapt.utility import ramp_complexity
from firedrake_adjoint import *
from firedrake.adjoint.solving import get_solve_blocks
from time import perf_counter
from options import PointDischarge3dOptions


# Parse arguments
parser = Parser(prog='test_cases/point_discharge3d/run_adaptation_loop.py')
parser.add_argument('configuration', 'aligned', help="""
    Choose from 'aligned' and 'offset'.
    """)
parser.add_argument('approach', 'hessian')
parser.add_argument('-num_repetitions', 1, help="""
    Number of times to repeat the simulation (default 1).

    This is for timing purposes.
    """)
parser.add_argument('-family', 'cg')
parser.add_argument('-recovery_method', 'Clement')
parser.add_argument('-norm_order', 1.0)
parser.add_argument('-convergence_rate', 6.0)
parser.add_argument('-miniter', 3)
parser.add_argument('-maxiter', 35)
parser.add_argument('-element_rtol', 0.005)
parser.add_argument('-qoi_rtol', 0.005)
parser.add_argument('-h_min', 1.0e-06)
parser.add_argument('-h_max', 1.0e+02)
parser.add_argument('-a_max', 1.0e+05)
parser.add_argument('-boundary', False)
parser.add_argument('-flux_form', False)
parsed_args = parser.parse_args()
config = parsed_args.configuration
family = parsed_args.family
method = parsed_args.recovery_method
num_repetitions = parsed_args.num_repetitions
assert num_repetitions >= 1
approach = parsed_args.approach
p = parsed_args.norm_order
assert p >= 1.0
alpha = parsed_args.convergence_rate
assert alpha >= 1.0
miniter = parsed_args.miniter
assert miniter >= 0
maxiter = parsed_args.maxiter
assert maxiter > 0
element_rtol = parsed_args.element_rtol
assert element_rtol > 0.0
qoi_rtol = parsed_args.qoi_rtol
assert qoi_rtol > 0.0
h_min = parsed_args.h_min
assert h_min > 0.0
h_max = parsed_args.h_max
assert h_max > h_min
a_max = parsed_args.a_max
assert a_max > 1.0
boundary = parsed_args.boundary
flux_form = parsed_args.flux_form
cwd = os.path.join(os.path.dirname(__file__))
output_dir = create_directory(os.path.join(cwd, 'outputs', config, approach, 'cg1'))

# Set targets to get a relatively even spread
targets = {
    'isotropic_dwr': [4000, 8000, 16000, 32000, 70000],
    'anisotropic_dwr': [2000, 8000, 32000, 128000, 300000],
    'weighted_hessian': [500, 1000, 4000, 16000, 64000],
    'weighted_gradient': [500, 1000, 4000, 16000, 50000],
}
if num_repetitions > 1:
    for key in targets:
        targets[key] = [targets[key][-1]]
num_refinements = len(targets[approach]) - 1

# Loop over mesh refinement levels
lines = 'qois,dofs,elements,wallclock,iterations,wallclock_metric,converged_reason\n'
tape = get_working_tape()
if approach == 'hessian':
    stop_annotating()
converged_reason = None
for level, target in enumerate(targets[approach]):
    cpu_times = []
    cpu_times_metric = []
    converged_reason = None
    for rep in range(num_repetitions):
        if converged_reason == 'diverged':
            continue
        msg = f'Refinement {level}/{num_refinements}, repetition {rep+1}/{num_repetitions}' \
              + f' ({approach}, {config})'
        print_output('\n'.join(['\n', '*'*len(msg), msg, '*'*len(msg)]))
        cpu_timestamp = perf_counter()
        cpu_times_metric.append(0)

        # Adapt until mesh convergence is achieved
        mesh = None
        qoi_old = None
        elements_old = None
        converged_reason = None
        for i in range(maxiter):
            tape.clear_tape()

            # Ramp up the target complexity
            target_ramp = ramp_complexity(3000.0, target, i)

            # Setup
            options = PointDischarge3dOptions(configuration=config, family=family, mesh=mesh)
            mesh = options.mesh3d
            options.output_directory = output_dir
            options.fields_to_export = []
            solver_obj = PlantSolver3d(options, optimise=True)

            # Forward solve
            solver_obj.iterate()

            # Check for QoI convergence
            tracer_3d = solver_obj.fields.tracer_3d
            qoi = options.qoi(tracer_3d)
            if qoi_old is not None and i > miniter:
                if np.abs(qoi - qoi_old) < qoi_rtol*np.abs(qoi_old):
                    print_output(f'Converged after {i+1} iterations due to QoI convergence.')
                    converged_reason = 'QoI'
                    break

            # Construct metric
            cpu_timestamp_metric = perf_counter()
            ee = ErrorEstimator(options,
                                error_estimator='difference_quotient',
                                metric=approach,
                                recovery_method=method)
            uv = solver_obj.fields.uv_3d
            if approach == 'hessian':
                metric = ee.recover_hessian(tracer_3d)
            else:
                solve_blocks = get_solve_blocks()
                compute_gradient(qoi, Control(options.tracer['tracer_3d'].diffusivity))
                adjoint_tracer_3d = solve_blocks[-1].adj_sol
                with stop_annotating():
                    metric = ee.metric(uv, tracer_3d, uv, adjoint_tracer_3d,
                                       target_complexity=target_ramp,
                                       convergence_rate=alpha,
                                       norm_order=p,
                                       flux_form=flux_form,
                                       boundary=boundary)
            if approach not in ('anisotropic_dwr', 'weighted_gradient'):
                enforce_element_constraints(metric, 1.0e-30, 1.0e+30, 1.0e+12, optimise=True)
                space_normalise(metric, target_ramp, p)
            enforce_element_constraints(metric, h_min, h_max, a_max, optimise=True)
            cpu_times_metric[-1] += perf_counter() - cpu_timestamp_metric

            # Adapt mesh and check convergence
            mesh = adapt(mesh, metric)
            elements = mesh.num_cells()
            if elements_old is not None and i > miniter:
                if np.abs(elements - elements_old) < element_rtol*elements_old:
                    print_output(f'Converged after {i+1} iterations due to element count convergence.')
                    converged_reason = 'element'
                    break
            elements_old = elements
            qoi_old = qoi
        if converged_reason is None:
            converged_reason = 'maxiter'
            print_output(f'Failed to converge after {maxiter} iterations.')
            continue
        cpu_times.append(perf_counter() - cpu_timestamp)

    # Logging
    if converged_reason == 'element':
        options = PointDischarge3dOptions(configuration=config, family=family, mesh=mesh)
        mesh = options.mesh3d
        options.output_directory = output_dir
        options.fields_to_export = []
        solver_obj = PlantSolver3d(options, optimise=True)
        options.apply_boundary_conditions(solver_obj)
        options.apply_initial_conditions(solver_obj)
        solver_obj.iterate()
        qoi = options.qoi(solver_obj.fields.tracer_3d)
    dofs = solver_obj.function_spaces.Q_3d.dof_count
    wallclock = np.mean(cpu_times)
    wallclock_metric = np.mean(cpu_times_metric)
    lines += f'{qoi},{dofs},{elements},{wallclock},{i+1},{wallclock_metric},{converged_reason}\n'
    with open(os.path.join(output_dir, 'convergence.log'), 'w+') as log:
        log.write(lines)
