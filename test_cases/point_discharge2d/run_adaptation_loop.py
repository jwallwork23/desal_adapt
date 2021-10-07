from desal_adapt import *
from desal_adapt.error_estimation import ErrorEstimator
from pyadjoint import solve_adjoint, stop_annotating, get_working_tape
from time import perf_counter
from options import PointDischarge2dOptions


# Parse arguments
parser = Parser(prog='test_cases/point_discharge2d/run_adaptation_loop.py')
parser.add_argument('configuration', 'aligned', help="""
    Choose from 'aligned' and 'offset'.
    """)
parser.add_argument('-num_refinements', 4, help="""
    Number of mesh refinements to consider (default 4).
    """)
parser.add_argument('-family', 'cg')
parser.add_argument('-approach', 'hessian')
parser.add_argument('-norm_order', 1.0)
parser.add_argument('-miniter', 3)
parser.add_argument('-maxiter', 35)
parser.add_argument('-element_rtol', 0.005)
parser.add_argument('-qoi_rtol', 0.005)
parser.add_argument('-h_min', 1.0e-10)
parser.add_argument('-h_max', 1.0e+02)
parser.add_argument('-a_max', 1.0e+05)
parsed_args = parser.parse_args()
config = parsed_args.configuration
family = parsed_args.family
num_refinements = parsed_args.num_refinements
assert num_refinements >= 1
approach = parsed_args.approach
assert approach in ['hessian']
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
h_min = parsed_args.h_min
assert h_min > 0.0
h_max = parsed_args.h_max
assert h_max > h_min
a_max = parsed_args.a_max
assert a_max > 1.0
cwd = os.path.join(os.path.dirname(__file__))
output_dir = create_directory(os.path.join(cwd, 'outputs', config, approach, 'cg1'))

# Loop over mesh refinement levels
lines = 'qois,dofs,elements,wallclock,iterations\n'
tape = get_working_tape()
if approach == 'hessian':
    stop_annotating()
for level in range(num_refinements + 1):
    msg = f'Refinement {level}/{num_refinements}'
    print_output('\n'.join(['*'*len(msg), msg, '*'*len(msg)]))
    cpu_timestamp = perf_counter()
    target = 1000.0*4.0**level

    # Adapt until mesh convergence is achieved
    mesh = None
    qoi_old = None
    elements_old = None
    for i in range(maxiter):
        tape.clear_tape()

        # Set parameters
        options = PointDischarge2dOptions(configuration=config, family=family, mesh=mesh)
        mesh = options.mesh2d
        options.output_directory = output_dir
        options.fields_to_export = []

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

        # Construct metric
        ee = ErrorEstimator(options, error_estimator='difference_quotient', metric=approach)
        if approach == 'hessian':
            metric = ee.recover_hessian(tracer_2d)
        else:
            solve_adjoint(qoi)
            solve_blocks = get_solve_blocks()
            assert len(solve_blocks) == 1
            adjoint_tracer_2d = solve_blocks[0].adj_sol
            uv = solver_obj.fields.uv_2d
            with stop_annotating():
                metric = ee.metric(uv, tracer_2d, uv, tracer_2d, uv, adjoint_tracer_2d, uv, adjoint_tracer_2d)
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

    # Logging
    qoi = options.qoi(solver_obj.fields.tracer_2d)
    dofs = solver_obj.function_spaces.Q_2d.dof_count
    elements = options.mesh2d.num_cells()
    wallclock = perf_counter() - cpu_timestamp
    lines += f'{qoi},{dofs},{elements},{wallclock},{i+1}\n'
    with open(os.path.join(output_dir, 'convergence.log'), 'w+') as log:
        log.write(lines)
