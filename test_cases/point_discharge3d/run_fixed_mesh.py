from desal_adapt import *
from options import PointDischarge3dOptions


# Parse arguments
parser = Parser(prog='test_cases/point_discharge3d/run_fixed_mesh.py')
parser.add_argument('configuration', 'aligned', help="""
    Choose from 'aligned' and 'offset'.
    """)
parser.add_argument('-level', 0, help="""
    Mesh resolution level (default 0).
    """)
parser.add_argument('-family', 'cg')
parsed_args = parser.parse_args()
config = parsed_args.configuration
level = parsed_args.level
family = parsed_args.family

# Set parameters
options = PointDischarge3dOptions(level=level, family=family, configuration=config)
output_dir = os.path.join(options.output_directory, config, 'fixed_mesh', f'{family}1', f'level{level}')
options.output_directory = create_directory(output_dir)

# Setup solver
solver_obj = PlantSolver3d(options)

# Solve
solver_obj.iterate()
output_dir = create_directory(os.path.join(output_dir, 'Tracer3d'))
File(os.path.join(output_dir, 'tracer_3d.pvd')).write(solver_obj.fields.tracer_3d)
