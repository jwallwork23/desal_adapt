from desal_adapt import *
from desal_adapt.adapt import GoalOrientedDesalinationPlant
from options import AnalyticalForcingsOptions


# Parse arguments
parser = Parser(prog='test_cases/analytical_forcings/run_adapt.py')
parser.add_argument('approach', 'isotropic_dwr')
parser.add_argument('-configuration', 'offset', help="""
    Choose from 'aligned' and 'offset'.
    """)
parser.add_argument('-level', 0, help="""
    Mesh resolution level inside the refined region.
    Choose a value from [0, 1, 2, 3, 4, 5] (default 0).""")
parser.add_argument('-family', 'cg')
parser.add_argument('-num_tidal_cycles', 2.0)
parser.add_argument('-num_meshes', 40)
parser.add_argument('-target', 4000.0)
parser.add_argument('-norm_order', 1.0)
parser.add_argument('-convergence_rate', 6.0)
parser.add_argument('-miniter', 3)
parser.add_argument('-maxiter', 8)
parser.add_argument('-element_rtol', 0.005)
parser.add_argument('-qoi_rtol', 0.001)
parser.add_argument('-h_min', 1.0e-04)
parser.add_argument('-h_max', 1.0e+02)
parser.add_argument('-a_max', 1.0e+05)
parser.add_argument('-flux_form', False)
parser.add_argument('-error_indicator', 'difference_quotient')
parser.add_argument('-load_index', 0)
parsed_args = parser.parse_args()
config = parsed_args.configuration
level = parsed_args.level
family = parsed_args.family
approach = parsed_args.approach
num_tidal_cycles = parsed_args.num_tidal_cycles
assert num_tidal_cycles > 0.0
num_subintervals = parsed_args.num_meshes
assert num_subintervals > 0
target = parsed_args.target
assert target > 0.0

# Set parameters
options = AnalyticalForcingsOptions(level=level, configuration=config, family=family)
options.simulation_end_time = num_tidal_cycles*options.tide_time
output_dir = os.path.join(options.output_directory, config, approach, f'{family}1', f'target{target:.0f}')
options.output_directory = create_directory(output_dir)
options.fields_to_export = ['tracer_2d']

# Setup solver
desal_plant = GoalOrientedDesalinationPlant(options, output_dir, num_subintervals)
conv = desal_plant.fixed_point_iteration(**parsed_args)
print(conv)
with open(os.path.join(output_dir, 'qoi.log'), 'w+') as f:
    f.write(conv)
