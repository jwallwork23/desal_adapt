from desal_adapt import *
from desal_adapt.error_estimation import ErrorEstimator
from desal_adapt.plotting import *
from firedrake_adjoint import *
from firedrake.adjoint.solving import get_solve_blocks
from options import PointDischarge3dOptions


# Parse arguments
parser = Parser(prog='test_cases/point_discharge3d/run_adapt.py')
parser.add_argument('configuration', 'aligned', help="""
    Choose from 'aligned' and 'offset'.
    """)
parser.add_argument('approach', 'hessian')
parser.add_argument('-level', 0, help="""
    Base mesh resolution level (default 0).
    """)
parser.add_argument('-family', 'cg')
parser.add_argument('-recovery_method', 'Clement')
parser.add_argument('-target', 10000.0)
parser.add_argument('-norm_order', 1.0)
parser.add_argument('-convergence_rate', 6.0)
parser.add_argument('-miniter', 3)
parser.add_argument('-maxiter', 35)
parser.add_argument('-element_rtol', 0.005)
parser.add_argument('-qoi_rtol', 0.005)
parser.add_argument('-h_min', 1.0e-10)
parser.add_argument('-h_max', 1.0e+02)
parser.add_argument('-a_max', 1.0e+05)
parser.add_argument('-boundary', False)
parser.add_argument('-flux_form', False)
parser.add_argument('-profile', False)
parsed_args = parser.parse_args()
config = parsed_args.configuration
level = parsed_args.level
family = parsed_args.family
method = parsed_args.recovery_method
approach = parsed_args.approach
target = parsed_args.target
assert target > 0.0
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
profile = parsed_args.profile

# Adapt until mesh convergence is achieved
mesh = None
qoi_old = None
elements_old = None
converged_reason = None
tape = get_working_tape()
for i in range(maxiter):
    tape.clear_tape()
    msg = f'Iteration {i+1}/{maxiter} ({approach}, {config})'
    print_output('\n'.join(['\n', '*'*len(msg), msg, '*'*len(msg)]))

    # Ramp up the target complexity
    base = 30000.0
    if i == 0:
        target_ramp = base
    elif i == 1:
        target_ramp = (2*base + target)/3
    elif i == 2:
        target_ramp = (base + 2*target)/3
    else:
        target_ramp = target

    # Setup
    options = PointDischarge3dOptions(level=level, family=family, configuration=config, mesh=mesh)
    mesh = options.mesh3d
    output_dir = os.path.join(options.output_directory, config, approach, f'{family}1', f'target{target:.0f}')
    options.output_directory = create_directory(output_dir)
    options.no_exports = profile
    create_directory(os.path.join(output_dir, 'Tracer3d'))
    solver_obj = PlantSolver3d(options, optimise=profile)

    # Forward solve
    solver_obj.iterate()

    # Check for QoI convergence
    tracer_3d = solver_obj.fields.tracer_3d
    if not profile:
        File(os.path.join(output_dir, 'Tracer3d', 'tracer_3d.pvd')).write(tracer_3d)
    qoi = options.qoi(tracer_3d)
    print_output(f'QoI = {qoi:.8e}')
    if qoi_old is not None and i > miniter:
        if np.abs(qoi - qoi_old) < qoi_rtol*np.abs(qoi_old):
            converged_reason = 'QoI'
            break

    # Construct metric
    ee = ErrorEstimator(options, error_estimator='difference_quotient', metric=approach, recovery_method=method)
    uv = solver_obj.fields.uv_3d
    if approach == 'hessian':
        metric = ee.recover_hessian(tracer_3d)
    else:
        solve_blocks = get_solve_blocks()
        with firedrake.PETSc.Log.Event("solve_adjoint"):
            compute_gradient(qoi, Control(options.tracer['tracer_3d'].diffusivity))
        adjoint_tracer_3d = solve_blocks[-1].adj_sol
        if not profile:
            File(os.path.join(output_dir, 'Tracer3d', 'adjoint_3d.pvd')).write(adjoint_tracer_3d)
        with stop_annotating():
            metric = ee.metric(uv, tracer_3d, uv, adjoint_tracer_3d,
                               target_complexity=target,
                               convergence_rate=alpha,
                               norm_order=p,
                               flux_form=flux_form,
                               boundary=boundary)
            if not profile:
                File(os.path.join(output_dir, 'metric_3d.pvd')).write(metric)
    with stop_annotating():
        if approach not in ('anisotropic_dwr', 'weighted_gradient'):
            enforce_element_constraints(metric, 1.0e-30, 1.0e+30, 1.0e+12, optimise=profile)
            space_normalise(metric, target, p)
        enforce_element_constraints(metric, h_min, h_max, a_max, optimise=profile)

    # Adapt mesh and check convergence
    mesh = adapt(mesh, metric)
    elements = mesh.num_cells()
    if elements_old is not None and i > miniter:
        if np.abs(elements - elements_old) < element_rtol*elements_old:
            converged_reason = 'element count'
            break
    elements_old = elements
    qoi_old = qoi
    if i + 1 == maxiter:
        raise ConvergenceError(f"Failed to converge after {maxiter} iterations.")

# Print final QoI value
print_output(f"Converged after {i+1} iterations due to {converged_reason} convergence.")
if converged_reason == 'element_count':
    options = PointDischarge3dOptions(level=level, family=family, configuration=config, mesh=mesh)
    output_dir = os.path.join(options.output_directory, config, approach, f'{family}1', f'target{target:.0f}')
    options.output_directory = create_directory(output_dir)
    create_directory(os.path.join(output_dir, 'Tracer3d'))
    solver_obj = PlantSolver3d(options)
    options.apply_boundary_conditions(solver_obj)
    options.apply_initial_conditions(solver_obj)
    solver_obj.iterate()
    tracer_3d = solver_obj.fields.tracer_3d
    if not profile:
        File(os.path.join(output_dir, 'Tracer3d', 'tracer_3d.pvd')).write(tracer_3d)
    qoi = options.qoi(tracer_3d)
print_output(f'Converged QoI = {qoi:.8e}')
