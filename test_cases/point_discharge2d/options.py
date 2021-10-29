from desal_adapt import *
from pyroteus.math import bessk0
from thetis.configuration import PositiveFloat, FiredrakeScalarExpression
import numpy as np


__all__ = ["PointDischarge2dOptions"]


class PointDischarge2dOptions(PlantOptions):
    """
    Problem specification for a simple
    advection-diffusion test case with a
    point source, from [Riadh et al. 2014].

    [Riadh et al. 2014] A. Riadh, G.
        Cedric, M. Jean, "TELEMAC modeling
        system: 2D hydrodynamics TELEMAC-2D
        software release 7.0 user manual."
        Paris: R&D, Electricite de France,
        p. 134 (2014).
    """
    domain_length = PositiveFloat(50.0).tag(config=False)
    domain_width = PositiveFloat(10.0).tag(config=False)
    tracer_old = FiredrakeScalarExpression(None, allow_none=True).tag(config=True)

    def __init__(self, configuration='aligned', level=1, source_level=5, pipe_radius=None,
                 family='cg', mesh=None, shift=1.0):
        """
        :kwarg configuration: choose from 'aligned and 'offset'
        :kwarg level: mesh resolution level
        :kwarg pipe_radius: optional value for source parametrisation
        :kwarg source_level: mesh resolution level for calibrated source data
        :kwarg family: choose from 'cg' and 'dg'
        :kwarg mesh: user-provided mesh
        :kwarg shift: number of units to shift the point source to the right
        """
        debug("Initialising PointDischarge2dOptions")
        super(PointDischarge2dOptions, self).__init__()
        assert configuration in ('aligned', 'offset')
        assert level >= 0
        assert source_level >= 0
        assert family in ('cg', 'dg')
        assert shift >= 0.0

        # Setup mesh
        self.mesh2d = mesh
        if self.mesh2d is None:
            self.mesh2d = RectangleMesh(50*2**level, 10*2**level, self.domain_length, self.domain_width)
        self.setup_mesh(self.mesh2d)

        # Physics
        self.tracer_only = True
        self.horizontal_velocity_scale = Constant(1.0)
        D = Constant(0.1)
        self.horizontal_diffusivity_scale = D
        self.bathymetry2d = Constant(1.0)  # arbitrary
        self.bnd_conditions = {
            1: {'value': Constant(0.0)},  # inflow
            2: {},                        # outflow
            3: {'diff_flux': Constant(0.0)},
            4: {'diff_flux': Constant(0.0)},
        }

        # Point source parametrisation
        self.source_value = 100.0
        self.source_x = Constant(1.0 + shift)
        self.source_y = Constant(5.0)
        cwd = os.path.dirname(__file__)
        fname = os.path.join(cwd, 'data', f'calibrated_radius_{source_level}.npy')
        if pipe_radius is not None:
            assert pipe_radius > 0.0
            self.source_r = Constant(pipe_radius)
        elif os.path.exists(fname):
            self.source_r = Constant(np.load(fname)[0])
        else:
            self.source_r = Constant(0.05605917)
        self.add_tracer_2d('tracer_2d',
                           'Depth averaged tracer',
                           'Tracer2d',
                           shortname='Tracer',
                           diffusivity=D,
                           source=self.source)

        # Receiver parametrisation
        self.receiver_x = 20.0
        self.receiver_y = 5.0 if configuration == 'aligned' else 7.5
        self.receiver_r = 0.5

        # Spatial discretisation
        self.tracer_element_family = family
        self.use_lax_friedrichs_tracer = family == 'dg'
        self.use_supg_tracer = family == 'cg'
        self.use_limiter_for_tracers = family == 'dg'

        # Temporal discretisation
        self.tracer_timestepper_type = 'SteadyState'
        self.timestep = 20.0
        self.simulation_export_time = 18.0
        self.simulation_end_time = 18.0

        # Solver parameters
        self.tracer_timestepper_options.solver_parameters.update({
            'mat_type': 'aij',
            'ksp_type': 'preonly',
            'pc_type': 'lu',
            'pc_factor_mat_solver_type': 'mumps',
        })

        # I/O
        self.fields_to_export = ['tracer_2d']
        self.fields_to_export_hdf5 = []

        self._isfrozen = True

    def apply_boundary_conditions(self, solver_obj):
        debug("Applying boundary conditions")
        if len(solver_obj.function_spaces.keys()) == 0:
            solver_obj.create_function_spaces()
        solver_obj.bnd_functions['tracer'] = self.bnd_conditions

    def apply_initial_conditions(self, solver_obj):
        debug("Applying initial conditions")
        if len(solver_obj.function_spaces.keys()) == 0:
            solver_obj.create_function_spaces()
        uv = Function(solver_obj.function_spaces.U_2d)
        uv.interpolate(as_vector([self.horizontal_velocity_scale.values()[0], 0.0]))
        if self.tracer_old is None:
            solver_obj.assign_initial_conditions(uv=uv)
        else:
            solver_obj.assign_initial_conditions(uv=uv, tracer=self.tracer_old)

    @property
    def source(self):
        x, y = SpatialCoordinate(self.mesh2d)
        x0 = self.source_x
        y0 = self.source_y
        r = self.source_r
        return self.source_value*exp(-((x - x0)**2 + (y - y0)**2)/r**2)

    @property
    def qoi_kernel(self):
        x, y = SpatialCoordinate(self.mesh2d)
        x0 = self.receiver_x
        y0 = self.receiver_y
        r = self.receiver_r
        ball = conditional((x - x0)**2 + (y - y0)**2 < r**2, 1, 0)
        area = assemble(ball*dx)
        scaling = 1.0 if np.isclose(area, 0.0) else pi*r**2/area
        return scaling*ball

    @PETSc.Log.EventDecorator("PointDischarge2dOptions.qoi")
    def qoi(self, solution, quadrature_degree=12):
        debug("Computing QoI")
        dx_qoi = dx(degree=quadrature_degree)
        return assemble(self.qoi_kernel*solution*dx_qoi)

    @property
    def analytical_solution_expression(self):
        x, y = SpatialCoordinate(self.mesh2d)
        x0 = self.source_x
        y0 = self.source_y
        r = self.source_r
        u = self.horizontal_velocity_scale
        D = self.horizontal_diffusivity_scale
        Pe = 0.5*u/D
        q = 1.0
        rr = max_value(sqrt((x - x0)**2 + (y - y0)**2), r)
        return 0.5*q/(pi*D)*exp(Pe*(x - x0))*bessk0(Pe*rr)

    @property
    def analytical_solution(self):
        fs = get_functionspace(self.mesh2d, self.tracer_element_family.upper(), 1)
        solution = Function(fs, name='Analytical solution')
        solution.interpolate(self.analytical_solution_expression)
        return solution

    def analytical_qoi(self, quadrature_degree=12):
        solution = self.analytical_solution_expression
        dx_qoi = dx(degree=quadrature_degree)
        return assemble(self.qoi_kernel*solution*dx_qoi)


if __name__ == '__main__':
    from desal_adapt.parse import Parser

    parser = Parser(prog='test_cases/point_discharge2d/options.py')
    parser.add_argument('configuration', 'aligned', help="""
        Choose from 'aligned' and 'offset'.
        """)
    parser.add_argument('-num_refinements', 6, help="""
        Number of mesh refinements (default 6).
        """)
    parser.add_argument('-quadrature_degree', 12)
    parsed_args = parser.parse_args()
    config = parsed_args.configuration
    num_refinements = parsed_args.num_refinements
    assert num_refinements >= 0
    degree = parsed_args.quadrature_degree
    assert degree >= 0
    lines = ''
    cwd = os.path.join(os.path.dirname(__file__))
    output_dir = create_directory(os.path.join(cwd, 'outputs', config, 'fixed_mesh', 'cg1'))
    for level in range(num_refinements+1):
        options = PointDischarge2dOptions(level=level, configuration=config)
        dofs = options.mesh2d.num_vertices()
        qoi = options.analytical_qoi(quadrature_degree=degree)
        line = f'DoF count {dofs:7d},   analytical QoI = {qoi:.8e}'
        print_output(line)
        lines = '\n'.join([lines, line])
        with open(os.path.join(output_dir, 'analytical_qoi.log'), 'w+') as log:
            log.write(lines)
