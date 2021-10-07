from desal_adapt import *
from desal_adapt.error_estimation import ErrorEstimator
from options import PointDischarge2dOptions


# Parse arguments
parser = Parser(prog='test_cases/point_discharge2d/run_fixed_mesh.py')
parser.add_argument('configuration', 'aligned', help="""
    Choose from 'aligned' and 'offset'.
    """)
parser.add_argument('-level', 0, help="""
    Base mesh resolution level (default 0).
    """)
parser.add_argument('-approach', 'hessian')
parser.add_argument('-target', 4000.0)
parser.add_argument('-norm_order', 1.0)
parser.add_argument('-miniter', 3)
parser.add_argument('-maxiter', 35)
parser.add_argument('-element_rtol', 0.005)
parser.add_argument('-qoi_rtol', 0.005)
parsed_args = parser.parse_args()
config = parsed_args.configuration
assert config in ['aligned', 'offset']
level = parsed_args.level
assert level >= 0
approach = parsed_args.approach
assert approach in ['hessian']
target = parsed_args.target
assert target > 0.0
p = parsed_args.norm_order
assert p >= 1.0
miniter = parsed_args.miniter
assert miniter >= 0
maxiter = parsed_args.maxiter
assert maxiter > 0
element_rtol = parsed_args.element_rtol
assert element_rtol > 0.0
qoi_rtol = parsed_args.qoi_rtol
assert qoi_rtol > 0.0

# Adapt until mesh convergence is achieved
mesh = None
qoi_old = None
elements_old = None
for i in range(maxiter):

    # Set parameters
    options = PointDischarge2dOptions(level=level, configuration=config, mesh=mesh)
    mesh = options.mesh2d
    output_dir = os.path.join(options.output_directory, config, approach, 'cg1', f'target{target:.0f}')
    options.output_directory = create_directory(output_dir)

    # Create solver
    solver_obj = PlantSolver(options)
    options.apply_boundary_conditions(solver_obj)
    options.apply_initial_conditions(solver_obj)

    # Solve
    solver_obj.iterate(update_forcings=options.update_forcings, export_func=options.export_func)
    tracer_2d = solver_obj.fields.tracer_2d
    qoi = options.qoi(tracer_2d)
    if qoi_old is not None and i > miniter:
        if np.abs(qoi - qoi_old) < qoi_rtol*np.abs(qoi_old):
            print_output(f"Converged after {i+1} iterations due to QoI convergence.")
            break

    # Construct metric  # TODO: ee can just get mesh from options
    ee = ErrorEstimator(options, mesh=mesh, error_estimator='difference_quotient')
    if approach == 'hessian':
        metric = ee.recover_hessian(tracer_2d)
    else:
        raise NotImplementedError  # TODO
    space_normalise(metric, target, p)
    enforce_element_constraints(metric, 1.0e-10, 1.0e+02, 1.0e+05)

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
