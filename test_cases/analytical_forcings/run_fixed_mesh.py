from desal_adapt.parse import Parser
from desal_adapt.solver import PlantSolver2d
from options import AnalyticalForcingsOptions
from thetis import create_directory, print_output, assemble, dx
import os
from time import perf_counter
from pyadjoint import pause_annotation


# Parse arguments
parser = Parser(prog='test_cases/analytical_forcings/run_fixed_mesh.py')
parser.add_argument('-configuration', 'offset', help="""
    Choose from 'aligned' and 'offset'.
    """)
parser.add_argument('-level', 0, help="""
    Mesh resolution level inside the refined region.
    Choose a value from [0, 1, 2, 3, 4, 5] (default 0).""")
parser.add_argument('-num_uniform_refinements', 0, help="""
    Number of uniform refinement to take across the
    whole domain (default 0).""")
parser.add_argument('-family', 'cg')
parser.add_argument('-num_tidal_cycles', 2.0)
parser.add_argument('-no_exports', False)
parsed_args = parser.parse_args()
config = parsed_args.configuration
level = parsed_args.level
family = parsed_args.family
num_tidal_cycles = parsed_args.num_tidal_cycles
assert num_tidal_cycles > 0.0
refs = parsed_args.num_uniform_refinements
cpu_timestamp = perf_counter()
pause_annotation()

# Set parameters
options = AnalyticalForcingsOptions(level=level, configuration=config, family=family,
                                    num_uniform_refinements=refs)
options.simulation_end_time = num_tidal_cycles*options.tide_time
output_dir = os.path.join(options.output_directory, config, 'fixed_mesh', f'{family}1', f'level{level}', 'refs{refs}')
options.output_directory = create_directory(output_dir)
options.fields_to_export = [] if parsed_args.no_exports else ['tracer_2d']

# Setup solver
solver_obj = PlantSolver2d(options)
uf = options.get_update_forcings(solver_obj)

# Setup QoI
sol = solver_obj.fields.tracer_2d
kernel = options.qoi_kernel
bg = options.background_salinity


def update_forcings(t):
    uf(t)
    current_qoi = assemble(kernel*(sol - bg)*dx)
    print_output(f'inlet salinity increase {current_qoi:.4e}')
    options.qoi_value += current_qoi


# Solve
solver_obj.iterate(update_forcings=update_forcings)
cpu_time = perf_counter() - cpu_timestamp
elements = options.mesh2d.num_cells()
dofs = solver_obj.function_spaces.Q_2d.dof_count
print_output(f'Fixed mesh level {level}, {config} QoI = {options.qoi_value:.8e}')
with open(os.path.join(output_dir, 'qoi.log'), 'w+') as log:
    log.write('dofs,elements,qoi,wallclock\n')
    log.write(f'{dofs},{elements},{options.qoi_value},{cpu_time}\n')
