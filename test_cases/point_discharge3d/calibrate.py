from desal_adapt import *
from firedrake_adjoint import *
from options import PointDischarge3dOptions


# Parse arguments
parser = Parser(prog='test_cases/point_discharge3d/calibrate.py')
parser.add_argument('level', 3, help="""
    Mesh resolution level (default 3).
    """)
parser.add_argument('-family', 'cg')
parser.add_argument('-initial_guess', 0.1, help="""
    Initial guess for the calibration experiment in metres (default 0.1).
    """)
parsed_args = parser.parse_args()
level = parsed_args.level
family = parsed_args.family
ig = parsed_args.initial_guess

# Set parameters
options = PointDischarge3dOptions(level=level, family=family, pipe_radius=ig)
options.no_exports = True
output_dir = create_directory(os.path.join(os.path.dirname(__file__), 'data'))

# Setup solver
solver_obj = PlantSolver3d(options)

# Solve
solver_obj.iterate()
approx = solver_obj.fields.tracer_3d

# Logging
options._isfrozen = False
options.m_progress = []
options.m = None
options.J_progress = []
options.J = None
options._isfrozen = True


def eval_cb(j, m):
    options.m_progress.append(m.dat.data[0])
    options.J_progress.append(j)


# Setup reduced functional
x, y, z = SpatialCoordinate(options.mesh3d)
x0 = options.source_x
y0 = options.source_y
z0 = options.source_z
r = options.source_r
analytical = options.analytical_solution_expression
cond = conditional((x - x0)**2 + (y - y0)**2 + (z - z0)**2 < r**2, 0, 1)
J = assemble(cond*(approx - analytical)**2*dx)
options.J_progress.append(J)
options.m_progress.append(r.dat.data[0])
Jhat = ReducedFunctional(J, Control(r), eval_cb_post=eval_cb)

# Run optimisation
r_cal = minimize(Jhat, method='L-BFGS-B', bounds=(0.001, np.Inf), options={'gtol': 1.0e-08})
print_output(f'Calibrated radius = {r_cal.dat.data[0]*100:.8f} cm')
np.save(os.path.join(output_dir, f'calibrated_radius_{level}.npy'), r_cal.dat.data)
np.save(os.path.join(output_dir, f'control_progress_{level}.npy'), options.m_progress)
np.save(os.path.join(output_dir, f'func_progress_{level}.npy'), options.J_progress)
