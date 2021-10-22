from desal_adapt import *
from desal_adapt.error_estimation import ErrorEstimator
from desal_adapt.plotting import *
from firedrake_adjoint import *
from firedrake.adjoint.solving import get_solve_blocks
from options import PointDischarge2dOptions


# Parse arguments
parser = Parser(prog='test_cases/point_discharge2d/run_adapt.py')
parser.add_argument('configuration', 'aligned', help="""
    Choose from 'aligned' and 'offset'.
    """)
parser.add_argument('approach', 'hessian')
parser.add_argument('-level', 1)
parser.add_argument('-family', 'cg')
parser.add_argument('-recovery_method', 'Clement')
parser.add_argument('-mixed_L2', False)
parser.add_argument('-target', 4000.0)
parser.add_argument('-norm_order', 1.0)
parser.add_argument('-convergence_rate', 6.0)
parser.add_argument('-miniter', 3)
parser.add_argument('-maxiter', 35)
parser.add_argument('-element_rtol', 0.005)
parser.add_argument('-qoi_rtol', 0.001)
parser.add_argument('-h_min', 1.0e-10)
parser.add_argument('-h_max', 1.0e+01)
parser.add_argument('-a_max', 1.0e+05)
parser.add_argument('-flux_form', False)
parser.add_argument('-profile', False)
parser.add_argument('-debug', False)
parsed_args = parser.parse_args()
config = parsed_args.configuration
level = parsed_args.level
family = parsed_args.family
method = parsed_args.recovery_method
mixed_L2 = parsed_args.mixed_L2
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
flux_form = parsed_args.flux_form
profile = parsed_args.profile
if parsed_args.debug:
    set_log_level(DEBUG)
method_str = method if method != 'L2' or not mixed_L2 else method + '_mixed'
cwd = os.path.dirname(__file__)
output_dir = os.path.join(cwd, 'outputs', config, approach, f'{family}1', method_str, f'target{target:.0f}')

# Adapt until mesh convergence is achieved
mesh = None
qoi_old = None
elements_old = None
tape = get_working_tape()
for i in range(maxiter):
    tape.clear_tape()

    # Setup
    options = PointDischarge2dOptions(level=level, family=family, configuration=config, mesh=mesh)
    mesh = options.mesh2d
    options.output_directory = create_directory(output_dir)
    options.no_exports = profile
    solver_obj = PlantSolver2d(options, optimise=profile)

    # Forward solve
    with firedrake.PETSc.Log.Event("solve_forward"):
        solver_obj.iterate()

    # Check for QoI convergence
    tracer_2d = solver_obj.fields.tracer_2d
    qoi = options.qoi(tracer_2d)
    if qoi_old is not None and i > miniter:
        if np.abs(qoi - qoi_old) < qoi_rtol*np.abs(qoi_old):
            print_output(f"Converged after {i+1} iterations due to QoI convergence.")
            break

    # Construct metric
    ee = ErrorEstimator(options, error_estimator='difference_quotient', metric=approach,
                        recovery_method=method, mixed_L2=mixed_L2)
    uv = solver_obj.fields.uv_2d
    if approach == 'hessian':
        debug("Recovering Hessian")
        metric = ee.recover_hessian(tracer_2d)
    else:
        debug("Solving adjoint problem")
        solve_blocks = get_solve_blocks()
        with firedrake.PETSc.Log.Event("solve_adjoint"):
            compute_gradient(qoi, Control(options.tracer['tracer_2d'].diffusivity))
        adjoint_tracer_2d = solve_blocks[-1].adj_sol
        with stop_annotating():
            debug("Computing metric")
            metric = ee.metric(uv, tracer_2d, uv, adjoint_tracer_2d,
                               target_complexity=target,
                               convergence_rate=alpha,
                               norm_order=p,
                               flux_form=flux_form)
    with stop_annotating():
        if approach not in ('anisotropic_dwr', 'weighted_gradient'):
            enforce_element_constraints(metric, 1.0e-30, 1.0e+30, 1.0e+12, optimise=profile)
            space_normalise(metric, target, p)
        debug("Enforcing element constraints")
        enforce_element_constraints(metric, h_min, h_max, a_max, optimise=profile)

    # Adapt mesh and check convergence
    debug("Adapting mesh")
    mesh = adapt(mesh, metric)
    debug("Checking for element convergence")
    elements = mesh.num_cells()
    if elements_old is not None and i > miniter:
        if np.abs(elements - elements_old) < element_rtol*elements_old:
            print_output(f"Converged after {i+1} iterations due to element count convergence.")
            break
    elements_old = elements
    qoi_old = qoi
    if i + 1 == maxiter:
        raise ConvergenceError(f"Failed to converge after {maxiter} iterations.")

# Plot
if not profile:
    fig, axes = plt.subplots(figsize=(8, 2.5))
    levels = np.linspace(0, 3, 50)
    triplot(mesh, axes=axes, interior_kw={'linewidth': 0.1}, boundary_kw={'color': 'k'})
    axes.axis(False)
    axes.set_xlim([0, 50])
    axes.set_ylim([0, 10])
    plt.tight_layout()
    cwd = os.path.join(os.path.dirname(__file__))
    plot_dir = create_directory(os.path.join(cwd, 'plots', config, f'{family}1', method_str))
    plt.savefig(os.path.join(plot_dir, f'{approach}_mesh_{config}_target{target:.0f}.jpg'), dpi=500)
