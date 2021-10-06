from desal_adapt import *
from options import PointDischarge2dOptions


# Parse arguments
parser = Parser(prog='test_cases/point_discharge2d/run_fixed_mesh.py')
parser.add_argument('configuration', 'aligned', help="""
    Choose from 'aligned' and 'offset'.
    """)
parser.add_argument('-level', 0, help="""
    Mesh resolution level (default 0).
    """)
parsed_args = parser.parse_args()
config = parsed_args.configuration
assert config in ['aligned', 'offset']
level = parsed_args.level
assert level >= 0

# Set parameters
options = PointDischarge2dOptions(level=level, configuration=config)
output_dir = os.path.join(options.output_directory, config, 'fixed_mesh', f'level{level}')
options.output_directory = create_directory(output_dir)

# Create solver
solver_obj = PlantSolver(options)
options.apply_boundary_conditions(solver_obj)
options.apply_initial_conditions(solver_obj)

# Solve
solver_obj.iterate(update_forcings=options.update_forcings, export_func=options.export_func)
