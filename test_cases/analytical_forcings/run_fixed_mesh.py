from desal_adapt import *
from options import AnalyticalForcingsOptions


# Parse arguments
parser = Parser(prog='test_cases/analytical_forcings/run_fixed_mesh.py')
parser.add_argument('configuration', 'aligned', help="""
    Choose from 'aligned' and 'offset'.
    """)
parser.add_argument('-level', 0, help="""
    Mesh resolution level inside the refined region.
    Choose a value from [0, 1, 2, 3, 4, 5] (default 0).""")
parser.add_argument('-family', 'cg')
parser.add_argument('-num_tidal_cycles', 2.0)
parsed_args = parser.parse_args()
config = parsed_args.configuration
level = parsed_args.level
family = parsed_args.family
num_tidal_cycles = parsed_args.num_tidal_cycles
assert num_tidal_cycles > 0.0

# Set parameters
options = AnalyticalForcingsOptions(level=level, configuration=config, family=family)
options.simulation_end_time = num_tidal_cycles*options.tide_time
output_dir = os.path.join(options.output_directory, config, 'fixed_mesh', f'{family}1', f'level{level}')
options.output_directory = create_directory(output_dir)
options.fields_to_export = ['tracer_2d']

# Setup solver
solver_obj = PlantSolver2d(options)

# Solve
solver_obj.iterate(update_forcings=options.get_update_forcings(solver_obj))
