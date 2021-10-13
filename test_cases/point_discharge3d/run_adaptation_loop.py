from desal_adapt import *
from desal_adapt.error_estimation import ErrorEstimator
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
parser.add_argument('-num_refinements', 2, help="""
    Number of mesh refinements to consider (default 4).
    """)
parser.add_argument('-num_repetitions', 1, help="""
    Number of times to repeat the simulation (default 1).

    This is for timing purposes.
    """)
parser.add_argument('-family', 'cg')
parser.add_argument('-norm_order', 1.0)
parser.add_argument('-convergence_rate', 6.0)
parser.add_argument('-miniter', 3)
parser.add_argument('-maxiter', 35)
parser.add_argument('-element_rtol', 0.005)
parser.add_argument('-qoi_rtol', 0.005)
parser.add_argument('-h_min', 1.0e-10)
parser.add_argument('-h_max', 1.0e+02)
parser.add_argument('-a_max', 1.0e+05)
parser.add_argument('-flux_form', False)
parsed_args = parser.parse_args()
config = parsed_args.configuration
family = parsed_args.family
num_refinements = parsed_args.num_refinements
assert num_refinements >= 1
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
flux_form = parsed_args.flux_form
cwd = os.path.join(os.path.dirname(__file__))
output_dir = create_directory(os.path.join(cwd, 'outputs', config, approach, 'cg1'))

# Loop over mesh refinement levels
lines = 'qois,dofs,elements,wallclock,iterations\n'
tape = get_working_tape()
if approach == 'hessian':
    stop_annotating()
for level in range(num_refinements + 1):
    target = 125.0*8.0**level
    cpu_times = []
    for rep in range(num_repetitions):
        msg = f'Refinement {level}/{num_refinements}, repetition {rep+1}/{num_repetitions}' \
              + f' ({approach}, {config})'
        print_output('\n'.join(['\n', '*'*len(msg), msg, '*'*len(msg)]))
        cpu_timestamp = perf_counter()

        # Adapt until mesh convergence is achieved
        mesh = None
        qoi_old = None
        elements_old = None
        for i in range(maxiter):
            tape.clear_tape()

            # Set parameters
            options = PointDischarge3dOptions(configuration=config, family=family, mesh=mesh)
            mesh = options.mesh3d
            options.output_directory = output_dir
            options.fields_to_export = []

            # Create solver
            solver_obj = PlantSolver3d(options, optimise=True)
            options.apply_boundary_conditions(solver_obj)
            options.apply_initial_conditions(solver_obj)

            # Solve
            try:
                solver_obj.iterate()
            except firedrake.ConvergenceError:
                print_output('Failed to converge with iterative solver parameters, trying direct.')
                options.tracer_timestepper_options.solver_parameters['pc_type'] = 'lu'
                solver_obj.iterate()
            tracer_3d = solver_obj.fields.tracer_3d
            qoi = options.qoi(tracer_3d)
            if qoi_old is not None and i > miniter:
                if np.abs(qoi - qoi_old) < qoi_rtol*np.abs(qoi_old):
                    print_output(f'Converged after {i+1} iterations due to QoI convergence.')
                    break

            # Construct metric
            ee = ErrorEstimator(options, error_estimator='difference_quotient', metric=approach)
            uv = solver_obj.fields.uv_3d
            if approach == 'hessian':
                metric = ee.recover_hessian(uv, tracer_3d)
            else:
                solve_blocks = get_solve_blocks()
                try:
                    compute_gradient(qoi, Control(options.tracer['tracer_3d'].diffusivity))
                except firedrake.ConvergenceError:
                    print_output('Failed to converge with iterative solver parameters, trying direct.')
                    solve_blocks[-1].adj_kwargs['solver_parameters']['pc_type'] = 'lu'
                    compute_gradient(qoi, Control(options.tracer['tracer_3d'].diffusivity))
                adjoint_tracer_3d = solve_blocks[-1].adj_sol
                with stop_annotating():
                    metric = ee.metric(uv, tracer_3d, uv, adjoint_tracer_3d,
                                       target_complexity=target,
                                       convergence_rate=alpha,
                                       norm_order=p,
                                       flux_form=flux_form)
            if approach not in ('anisotropic_dwr', 'weighted_gradient'):
                space_normalise(metric, target, p)
            enforce_element_constraints(metric, h_min, h_max, a_max)

            # Adapt mesh and check convergence
            mesh = adapt(mesh, metric)
            elements = mesh.num_cells()
            if elements_old is not None and i > miniter:
                if np.abs(elements - elements_old) < element_rtol*elements_old:
                    print_output(f"Converged after {i+1} iterations due to element count convergence.")
                    break
            elements_old = elements
            qoi_old = qoi
            if i + 1 == maxiter:
                raise ConvergenceError(f"Failed to converge after {maxiter} iterations.")
        cpu_times.append(perf_counter() - cpu_timestamp)

    # Logging
    qoi = options.qoi(solver_obj.fields.tracer_3d)
    dofs = solver_obj.function_spaces.Q_3d.dof_count
    elements = options.mesh3d.num_cells()
    wallclock = np.mean(cpu_times)
    lines += f'{qoi},{dofs},{elements},{wallclock},{i+1}\n'
    with open(os.path.join(output_dir, 'convergence.log'), 'w+') as log:
        log.write(lines)