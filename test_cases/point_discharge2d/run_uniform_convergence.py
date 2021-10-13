from desal_adapt import *
from time import perf_counter
from options import PointDischarge2dOptions


# Parse arguments
parser = Parser(prog='test_cases/point_discharge2d/run_uniform_convergence.py')
parser.add_argument('configuration', 'aligned', help="""
    Choose from 'aligned' and 'offset'.
    """)
parser.add_argument('-num_refinements', 5, help="""
    Number of mesh refinements to consider (default 5).
    """)
parser.add_argument('-num_repetitions', 1, help="""
    Number of times to repeat the simulation (default 1).

    This is for timing purposes.
    """)
parser.add_argument('-family', 'cg')
parsed_args = parser.parse_args()
config = parsed_args.configuration
num_refinements = parsed_args.num_refinements
assert num_refinements >= 1
num_repetitions = parsed_args.num_repetitions
assert num_repetitions >= 1
family = parsed_args.family
cwd = os.path.join(os.path.dirname(__file__))
output_dir = create_directory(os.path.join(cwd, 'outputs', config, 'fixed_mesh', f'{family}1'))

# Loop over mesh refinement levels
lines = 'qois,dofs,elements,wallclock\n'
for level in range(num_refinements + 1):
    cpu_times = []
    for rep in range(num_repetitions):
        msg = f'Refinement {level}/{num_refinements}, repetition {rep+1}/{num_repetitions}' \
              + f' (fixed_mesh, {config})'
        print_output('\n'.join(['\n', '*'*len(msg), msg, '*'*len(msg)]))
        cpu_timestamp = perf_counter()

        # Set parameters
        options = PointDischarge2dOptions(level=level, family=family, configuration=config)
        options.output_directory = output_dir
        options.fields_to_export = []

        # Setup solver
        solver_obj = PlantSolver2d(options)
        options.apply_boundary_conditions(solver_obj)
        options.apply_initial_conditions(solver_obj)
        solver_obj.iterate(update_forcings=options.update_forcings, export_func=options.export_func)
        cpu_times.append(perf_counter() - cpu_timestamp)

    # Logging
    qoi = options.qoi(solver_obj.fields.tracer_2d)
    dofs = solver_obj.function_spaces.Q_2d.dof_count
    elements = options.mesh2d.num_cells()
    wallclock = np.mean(cpu_times)
    lines += f'{qoi},{dofs},{elements},{wallclock}\n'
    with open(os.path.join(output_dir, 'convergence.log'), 'w+') as log:
        log.write(lines)
